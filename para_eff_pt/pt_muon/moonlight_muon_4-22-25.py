import os
import math
import torch
import torch.distributed as dist


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Moonlight_Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_on='momentum',
        svd_every=None,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        
        self.muon_on = muon_on
        self.svd_every = svd_every
        
 
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"] # not used when doing bias correction 
            wd = group["wd"]
            momentum = group["momentum"]
            
            
            if self.muon_on == 'adam':
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if self.muon_on == 'adam':
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    state["step"] += 1
                    step = state["step"]
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())
                elif self.muon_on == 'momentum':
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                elif self.muon_on == 'sgd':
                    u = g
                elif self.muon_on == 'svd_adamw':
                    
                    
                    # --- Step 1. GradNorm: Row-wise gradient normalization ---
                    # Compute the RMS for each row (RMS)
                    # s shape: (m, 1)
                    
                    #s = g.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
                    #g = g / s                    
                    # ?????????????????
            
                    
                    U, s, Vh = torch.linalg.svd(g.data.float(), full_matrices = False) # full_matrices = False ???
 
                    
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(s) # ???
                        state["ema_sq"] = torch.zeros_like(s)
                        state["step"] = 0
                    buf1 = state["momentum_buffer"]
                    buf2 = state["ema_sq"]
                    state["step"] += 1
                    
                    #buf.mul_(momentum).add_(s / s.norm() , alpha=(1.0 - momentum))  # ??? / s.norm() ??? do those values become to small ???
                    
                    #buf.mul_(momentum).add_(s  , alpha=(1.0 - momentum))
                    
                    b1 = 0.9
                    
                    buf1.mul_(b1).add_(s  , alpha=(1.0 - b1))
                    
                    b2 = 0.99 # 0.999
                    
                    buf2.mul_(b2).addcmul_(s, s, value=1.0 - b2)
                    denom = buf2.sqrt().add_(1e-8)
                    
                    
                    lr = group["lr"]  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    
                    #alpha = 0.1
                    #lr *= alpha
                    
                    bias_correction1 = 1.0 - b1 ** state["step"]
                    bias_correction2 = 1.0 - b2 ** state["step"]
                    lr = lr * math.sqrt(bias_correction2) / bias_correction1              
                    


                    
                    #print(U.shape, torch.diag(buf).shape, Vh.shape)
                    
                    #print(buf, s.norm())
                    
                    #k = min(20, s.numel())
                    #u = U[:, :k] @ torch.diag(buf[:k]) @ Vh[:k, :] # ????
    
                    #u = U @ torch.diag(  s / buf1 + 1e-8) @ Vh # ???? 
                    
                    u = U @ torch.diag(  buf1 / denom + 1e-8) @ Vh

                    #u = U @ torch.diag(s) @ Vh
                    #u = g # ??????
                    
                    #update_norm = u.norm()
                    #grad_norm = g.data.norm()

                    #if update_norm > 0:
                    #    u = u * (grad_norm / update_norm)
                    
                    # Apply Norm-Growth Limiter in Fira (https://arxiv.org/abs/2410.01623) to avoid destructive gradient updates.
                    if False:
                        if "scaled_grad" in state:
                            scaled_grad_norm = torch.norm(u)
                            limiter = max(
                                    scaled_grad_norm / 
                                    (state["scaled_grad"] + 1e-8),
                                    1.01,
                                ) / 1.01
                            u = u / limiter
                            state["scaled_grad"] = scaled_grad_norm / limiter
                        else:
                            state["scaled_grad"] = torch.norm(u)  

                elif self.muon_on == 'svd_adamw_pre' or self.muon_on == 'svd_adamw_pre_ns' or self.muon_on == 'svd_adamw_pre_ns_on_g' or self.muon_on == "svd_adamw_pre_ns_min":
                   
                   
                    if "step" not in state:
                        state["step"] = 0

                    if state["step"] % self.svd_every == 0: #???
                    
                        U, s, Vh = torch.linalg.svd(g.data.float(), full_matrices = False) # full_matrices = False ???
 
                    
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(s) # ???
                        state["ema_sq"] = torch.zeros_like(s)
                        #state["step"] = 0
                        
         
                    buf1 = state["momentum_buffer"]
                    buf2 = state["ema_sq"]
                    #state["step"] += 1
                    
                    
                    
                    
                    b1 = 0.9
                    b2 = 0.99 # 0.999
                    
                    if  state["step"] % self.svd_every == 0: #???
                    
                        buf1.mul_(b1).add_(s  , alpha=(1.0 - b1))
                        buf2.mul_(b2).addcmul_(s, s, value=1.0 - b2)
                        
                    denom = buf2.sqrt().add_(1e-8)
                        
                    state["step"] += 1 # ??? should this be here ???
                    
                    
                    # maybe try without bias corr when svd every > 1 ???
                    
                    lr = group["lr"]  
                    bias_correction1 = 1.0 - b1 ** (state["step"] / self.svd_every) # ????
                    bias_correction2 = 1.0 - b2 ** (state["step"] / self.svd_every) # ????
                    lr = lr * math.sqrt(bias_correction2) / bias_correction1              
                    

                    
                    S = torch.diag(  buf1 / denom + 1e-8)
                    
                    # !!! g is bf16
                    # !!! S is float !!
                    
                    if self.muon_on == "svd_adamw_pre_ns_on_g":
                        o = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                        u = o.data.float() @ S     # ??? float here or on g ???
                    elif self.muon_on == "svd_adamw_pre_ns_min":
                        if g.shape[0] < g.shape[1]: # left multiply ??? ??? ???
                            
                            m,n = g.shape[0], g.shape[1] # n > m
                            
                            if False:
                                u = S @ g.data.float()  
                            else:
                                extra = torch.eye(n - m, dtype=S.dtype, device=S.device)
                                top_right = torch.zeros(m, n - m, dtype=S.dtype, device=S.device)
                                bottom_left = torch.zeros(n - m, m, dtype=S.dtype, device=S.device)

                                top = torch.cat([S, top_right], dim=1)
                                bottom = torch.cat([bottom_left, extra], dim=1)
                                S_expanded = torch.cat([top, bottom], dim=0)
                                
                                u = g.data.float()  @ S_expanded
                                
                                print(m,n )

                        else:
                            u = g.data.float() @ S      
                    else:
                        u = g.data.float() @ S 
                
                
                elif self.muon_on == 'swan': # ????
                    
                    # --- Step 1. GradNorm: Row-wise gradient normalization ---
                    # Compute the RMS for each row (RMS)
                    # s shape: (m, 1)
                    
                    s = g.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
                    g = g / s     
                    
                    #lr = group["lr"]  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    #lr *=0.05 # ???????   
                    
                    # not correct for same reason as bias correction !!!
                                                    
                            
                elif self.muon_on == 'none':
                    pass
                else:
                    raise ValueError(f"invalid choice for muon_on argmument...")

 
                if not (self.muon_on == 'svd_adamw' or self.muon_on == 'sgd' or self.muon_on == 'svd_adamw_pre' or self.muon_on == "svd_adamw_pre_ns_on_g"):  # dont do NS for those...
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    
                adjusted_lr = lr    
                if not (self.muon_on == 'swan' or self.muon_on == 'sgd'): # ???
                    # scale update
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)  # ????????????????

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]
 
            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
