import torch
import torch.nn as nn
import transformers
import bitsandbytes as bnb

from para_eff_pt.peft_pretraining import training_utils
from para_eff_pt.pt_low_rank import LoRaFaModel
from para_eff_pt.pt_sltrain import SpLoRaModel

from para_eff_pt.pt_relora import ReLoRaModel
from para_eff_pt.pt_galore import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
from para_eff_pt.pt_fourier import FourierModel
from para_eff_pt.pt_fourier import Fourier_Lowrank_Model

from para_eff_pt.pt_flora import Flora
from para_eff_pt.pt_golore import GoLoreAdamW,GoLoreAdamW8bit,GoLoreSGD
from para_eff_pt.pt_golore import GoloreReLoRaModel,GoloreReLoRaLinear
from para_eff_pt.pt_loro import LORO_optimizer

from para_eff_pt.pt_spam import SPAM_optimizer 

from para_eff_pt.pt_stable_spam import StableSPAM_optimizer

from para_eff_pt.pt_fira import Fira_AdamW

from para_eff_pt.pt_apollo import Apollo_AdamW

from para_eff_pt.pt_adamw_beta import adamw_beta
 

from para_eff_pt.pt_muon import Muon

from para_eff_pt.pt_muon import Moonlight_Muon

from para_eff_pt.pt_muon import Moonlight_Muon_Subspace
 



def build_model(model, args):
    if args.peft_model.lower() == "low-rank":
        model = LoRaFaModel(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            trainable_scaling=args.train_scaling,
        )
    elif args.peft_model.lower() == "sltrain":
        model = SpLoRaModel(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            trainable_scaling=args.train_scaling,
            sp_ratio=args.sp_ratio,
            sp_type=args.sp_type,
            random_subspace=args.random_subspace,
        )

    elif args.peft_model.lower() == "fourier":
        model = FourierModel(
            model,
            target_modules=args.target_modules,
            n_freq=args.n_freq,
            fourier_scale=args.fourier_scale,
        )
    elif args.peft_model.lower() == "lora":
        model = LoRaModel(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            trainable_scaling=args.train_scaling,
        )
    elif args.peft_model.lower() == "relora":
        model = ReLoRaModel(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            trainable_scaling=args.train_scaling,
        )
    elif args.peft_model.lower() == "fourier_low_rank":
        model = Fourier_Lowrank_Model(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            trainable_scaling=args.train_scaling,
            target_modules=args.target_modules,
            n_freq=args.n_freq,
            fourier_scale=args.fourier_scale,
        )
    elif args.peft_model.lower() == "restart_sltrain":
        model = SpLoRaModel_SVD(
            model,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            trainable_scaling=args.train_scaling,
        )
    
 
    elif args.peft_model.lower() == "golore":
        model = GoloreReLoRaModel(
            model,
            r=args.rank,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
    

    return model



def build_optimizer(model, trainable_params, args):   
    #if args.optimizer.lower() == "adamw_beta":
    #    optimizer = adamw_beta(trainable_params, lr=args.lr, weight_decay=args.weight_decay, cycle_length=args.cycle_length)
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "swan":


        if args.swan_only_mpl_att:
        
            # SWAN ONLY APPLIED TO ATT AND MLP !!!
            
            
            hidden_matrix_params = []
            target_modules_list = ["attn", "mlp","attention"]
            
            if args.add_embed_tokens:
                target_modules_list.append("embed_tokens")
            if args.add_lm_head:
                target_modules_list.append("lm_head")
                
            print(f"TARGET MODULES = {target_modules_list} !!!!")
            
            
            for module_name, module in model.named_modules():
                if not (isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):   # ?????? or isinstance(module, nn.Embedding) ????
                    continue

                if not any(target_key in module_name for target_key in target_modules_list):
                    continue
                
                hidden_matrix_params.append(module.weight)
                
                print("Hidden: ", module_name)
                
            id_hidden_matrix_params = [id(p) for p in hidden_matrix_params]
            # make parameters without "rank" to another group
            adamw_params = [p for p in model.parameters() if id(p) not in id_hidden_matrix_params]
            
            for module_name, module in model.named_modules():
                if hasattr(module, 'weight'):
                    p = module.weight
                    if id(p) not in id_hidden_matrix_params:
                        print("Adam: ", module_name)
        
        else:
            
            print("TARGET MODULES = ALL 2D WEIGHTS")
            hidden_matrix_params = [p for p in model.parameters() if p.ndim == 2]
            adamw_params = [p for p in model.parameters() if p.ndim != 2]        

            id_hidden_matrix_params = [id(p) for p in hidden_matrix_params]
            
            for module_name, module in model.named_modules():
                if hasattr(module, 'weight'):
                    p = module.weight
                    if id(p) not in id_hidden_matrix_params:
                        print("Adam: ", module_name)            
            
            
        
        
        # Create the optimizer
        optimizer = SWAN(        
                            lr=args.lr,
                            wd=args.weight_decay,
                            swan_version=args.swan_version,
                            muon_params=hidden_matrix_params,
                            momentum=args.momentum,
                            ns_steps=5,
                            adamw_params=adamw_params,
                            adamw_betas=(args.adam_beta_1, args.adam_beta_2),
                            adamw_eps=1e-8,
                                )
        
        
        
        
    elif args.optimizer.lower() == "muon":
        optimizer = Muon(
            params=trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1
        )        
        # ??? rank | word_size ??? try only on 1 GPU for now ... # lr 0.02 default ... weight_decay = 0.01 default ...
    
        
        
    elif args.optimizer.lower() == "moonlight_muon":
        
        #'lm_head'
        
        #if args.muon_on == 'svd_adamw_pre' or args.muon_on == 'svd_adamw_pre_ns' or args.muon_on == 'svd_adamw_pre_ns_on_g': # svd_adamw_pre_ns_min !
        #    hidden_matrix_params = [p for p in model.parameters() if (p.ndim == 2 and p.shape[0] == p.shape[1] )]
        #    adamw_params = [p for p in model.parameters() if (p.ndim != 2 or (p.ndim == 2 and p.shape[0] != p.shape[1]) )]
        #else:
            
        hidden_matrix_params = [p for p in model.parameters() if p.ndim == 2]
        adamw_params = [p for p in model.parameters() if p.ndim != 2]         
        
        # Create the optimizer
        optimizer = Moonlight_Muon(        
                                    lr=args.lr,
                                    wd=args.weight_decay,
                                    #muon_on=args.muon_on,
                                    #svd_every=args.svd_every,
                                    muon_params=hidden_matrix_params,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    ns_steps=5,
                                    adamw_params=adamw_params,
                                    adamw_betas=(args.adam_beta_1, args.adam_beta_2),
                                    adamw_eps=1e-8,
                                )
 
        """
        if args.muon_on == 'sgd':
            print(" !!! ")
            for module_name, module in model.named_modules():
                if 'lm_head' in module_name.lower():
                    if hasattr(module, 'weight'):
                        weight_param = module.weight
                        adamw_params.append(weight_param)

                        for i, p in enumerate(hidden_matrix_params):
                            if p is weight_param:
                                del hidden_matrix_params[i]
                                print("!!! removed :", module_name)
                                break
        """
                            
        #print("  !!!!!!!!!!!! ")
        #for module_name, module in model.named_modules():
        #    for param in module.parameters(recurse=False):  # Only direct parameters
        #        if param.ndim != 2:
        #            print(module_name)
        #            break  # Avoid duplicate prints per module
        #print("  !!!!!!!!!!!! ")
            
 
        #optimizer = Moonlight_Muon(muon_params=hidden_matrix_params, lr=args.lr, momentum=args.momentum,
        #                adamw_params=adamw_params, adamw_lr=args.adam_lr, adamw_betas=(args.adam_beta_1, args.adam_beta_2), adamw_wd=args.weight_decay)
 
 
    elif args.optimizer.lower() == "moonlight_muon_subspace":
        
        # is lm_head trainined here ???? yes, muon regular
        
        #hidden_matrix_params = [p for p in model.parameters() if p.ndim == 2]
        adamw_params = [p for p in model.parameters() if p.ndim != 2]
        
        
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        muon_params_low_rank = []
        target_modules_list = ["attn", "mlp"] # those are 2d always ?!
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):    
                continue
            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            #logger.info(f"Adding {module_name} to Moonlight_Muon_subspace parameters")

            muon_params_low_rank.append(module.weight)

        id_lowrank_params = [id(p) for p in muon_params_low_rank]
        
        # make parameters without "rank" to another group
        muon_params_regular = [p for p in model.parameters() if ((id(p) not in id_lowrank_params) and (p.ndim == 2) )]   # ??? remaining of 2d params -> regular params for muon (rest are in adamw_params)
        

        
        
        # Create the optimizer
        optimizer = Moonlight_Muon_Subspace(        
                                    lr=args.lr,
                                    wd=args.weight_decay,
                                    rank=args.rank,
                                    update_proj_gap=args.update_proj_gap,
                                    alpha_scale=args.galore_scale,
                                    proj_type=args.proj_type,
                                    disable_nl=args.disable_nl,
                                    muon_params_regular=muon_params_regular,
                                    muon_params_low_rank=muon_params_low_rank,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    ns_steps=5,
                                    adamw_params=adamw_params,
                                    adamw_betas=(args.adam_beta_1, args.adam_beta_2),
                                    adamw_eps=1e-8,
                                )
 
 
 
 
 
 
 
        
    elif args.optimizer.lower() == "adamw_beta":
        optimizer = adamw_beta(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
        print("using adam_beta !")
        
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.beta1,
        )
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "adam8bit_per_layer":
        optimizer = {}
        for p in model.parameters():
            if p.requires_grad:
                optimizer[p] = bnb.optim.Adam8bit(
                    [p], lr=args.lr, weight_decay=args.weight_decay
                )
        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer[p].step()
            optimizer[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
    elif args.optimizer.lower() == "flora_adam":
        optimizer = Flora(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay,rank=args.rank,seed=args.seed
        )
    
    elif args.optimizer.lower() == "loro_optimizer":
        optimizer = LORO_optimizer(
            trainable_params, lr=args.lr, K=10
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    return optimizer

'''
def build_optimizer_moonlight_muon_lora(model, param_groups,  args):
    if args.optimizer.lower() == "moonlight_muon_lora":

        hidden_matrix_params = [p for p in model.parameters() if p.ndim == 2]
        adamw_params = [p for p in model.parameters() if p.ndim != 2]
        # Create the optimizer
        optimizer = Moonlight_Muon_Lora(          # ???
                                    lr=args.lr,
                                    wd=args.weight_decay,
                                    muon_params=hidden_matrix_params,
                                    momentum=0.95,
                                    nesterov=True,
                                    ns_steps=5,
                                    adamw_params=adamw_params,
                                    adamw_betas=(args.adam_beta_1, args.adam_beta_2),
                                    adamw_eps=1e-8,
                                )
        
        
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    return optimizer 
    ''' 


def build_optimizer_apollo(model, param_groups, id_galore_params, args, param_to_name):
    if args.optimizer.lower() == "apollo_adamw":
        optimizer = Apollo_AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, scale_front=args.scale_front, disable_nl=args.disable_nl, param_to_name=param_to_name)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    return optimizer 


def build_optimizer_fira(model, param_groups, id_galore_params, args):
    if args.optimizer.lower() == "fira_adamw":
        optimizer = Fira_AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, disable_nl=args.disable_nl)
        print("using fira optimizer !!")
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    return optimizer 
    
    

def build_optimizer_galore(model, param_groups, id_galore_params, args):
    layer_wise_flag = False
    if args.optimizer.lower() == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(
            param_groups, lr=args.lr, weight_decay=args.weight_decay, galore_use_nl=args.galore_use_nl
        )
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "galore_adamw8bit_per_layer":
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit(
                        [
                            {
                                "params": [p],
                                "rank": args.rank,
                                "update_proj_gap": args.update_proj_gap * 2,
                                "scale": args.galore_scale,
                                "proj_type": args.proj_type,
                            }
                        ],
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                    )
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit(
                        [p], lr=args.lr, weight_decay=args.weight_decay
                    )

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        layer_wise_flag = True

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    return optimizer

def build_optimizer_golore(model, param_groups, args):
    layer_wise_flag = False
    if args.optimizer.lower() == "golore_adamw":

        optimizer = GoLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "golore_adamw8bit":
 
        optimizer = GoLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "golore_sgd":
     
        optimizer = GoLoreSGD(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        
    return optimizer


def build_optimizer_spam(model, param_groups, args):
    if args.optimizer.lower() == "spam_adamw":
        optimizer = SPAM_optimizer(param_groups, lr=args.lr,weight_decay=args.weight_decay,warmup_epoch=args.warmup_epoch,threshold=args.threshold)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
        
    return optimizer

def build_optimizer_stable_spam(model, param_groups, args):
    if args.optimizer.lower() == "stable_spam_adamw":
        optimizer = StableSPAM_optimizer(params=param_groups, lr = args.lr, weight_decay = args.weight_decay,gamma1=args.gamma1,gamma2=args.gamma2,gamma3=args.gamma3,eta_min=args.eta,update_proj_gap=args.update_gap,total_T=args.total_T)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
        
    return optimizer


def build_optimizer_loro(model, param_groups, args):
    """
    Build the LORO optimizer
    """
    if args.optimizer.lower() == "loro_optimizer":
        regular_group = param_groups[0]
        loro_group = param_groups[1]
        
        print(f"Regular params: {len(regular_group['params'])}")
        print(f"LORO params: {len(loro_group['params'])}")
        
        # Create AdamW optimizer for regular parameters
        adamw_optimizer = torch.optim.AdamW(
            regular_group['params'],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Create LORO optimizer for low-rank parameters
        loro_optimizer = LORO_optimizer(
            [loro_group],  # Pass the entire parameter group here
            lr=args.lr,
            weight_decay=args.weight_decay,
            update_k=args.cycle_length,
        )
        
        ## To test if the structure is correct
        # loro_adam_optimizer = torch.optim.AdamW(
        #     loro_group['params'],
        #     lr=args.lr,
        #     weight_decay=args.weight_decay
        # )
        
        class CombinedOptimizer(torch.optim.Optimizer):
            def __init__(self, adamw_opt, loro_opt, loro_adam_optimizer=None):
                self.adamw = adamw_opt
                self.loro = loro_opt
                self.loro_adam = loro_adam_optimizer
                self.step_count = 0
                
                # Collect all parameters
                params = []
                param_groups = []
                for group in self.adamw.param_groups:
                    params.extend(group['params'])
                    param_groups.append(group)
                for group in self.loro.param_groups:
                    params.extend(group['params'])
                    param_groups.append(group)
                # for group in self.loro_adam.param_groups:
                #     params.extend(group['params'])
                #     param_groups.append(group)
                    
                # Call the parent class constructor
                defaults = {
                    'lr': param_groups[0]['lr'],
                    'weight_decay': param_groups[0]['weight_decay']
                }
                super().__init__(params, defaults)
                
                # Restore the original param_groups
                self.param_groups = param_groups
                
            def zero_grad(self, set_to_none: bool = False):
                self.adamw.zero_grad(set_to_none=set_to_none)
                self.loro.zero_grad(set_to_none=set_to_none)
                # self.loro_adam.zero_grad(set_to_none=set_to_none)
                
            @torch.no_grad()
            def step(self, closure=None):
                # Update learning rates for each optimizer before stepping
                adamw_groups_len = len(self.adamw.param_groups)
                
                for i, group in enumerate(self.adamw.param_groups):
                    group['lr'] = self.param_groups[i]['lr']
                
                for i, group in enumerate(self.loro.param_groups):
                    group['lr'] = self.param_groups[i + adamw_groups_len]['lr']
                
                # for i, group in enumerate(self.loro_adam.param_groups):
                #     group['lr'] = self.param_groups[i + adamw_groups_len]['lr']
                    
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
        
                self.adamw.step()
         
                self.loro.step()
                # self.loro_adam.step()
                
                if self.loro.is_exact == True:
                    print(f"Resetting optimizer state after exact update at step {self.step_count + 1}")
                    
                    ## Reset AdamW optimizer
                    self.adamw = torch.optim.AdamW(
                        regular_group['params'],
                        lr=args.lr,
                        weight_decay=args.weight_decay
                    )
                    
                    ## Reset LORO optimizer
                    self.loro = LORO_optimizer(
                        [loro_group],  # Pass the entire parameter group here
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        update_k=args.cycle_length,
                    )
                    
                    self.loro.is_exact = False

                self.step_count += 1
                
                return loss
                
            def state_dict(self):
                return {
                    'adamw': self.adamw.state_dict(),
                    'loro': self.loro.state_dict(),
                    'param_groups': self.param_groups,
                    'state': self.state,
                }
                
            def load_state_dict(self, state_dict):
                self.adamw.load_state_dict(state_dict['adamw'])
                self.loro.load_state_dict(state_dict['loro'])
                self.param_groups = state_dict['param_groups']
                self.state = state_dict['state']
        
        optimizer = CombinedOptimizer(adamw_optimizer, loro_optimizer)
        
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    
    return optimizer
