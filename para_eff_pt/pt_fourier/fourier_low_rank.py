import os
import math
import json

from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoConfig


# Define a dataclass to store the configuration of the Fourier model
@dataclass
class Fourier_Lowrank_Config:
    r: int
    lora_alpha: float
    lora_dropout: float
    trainable_scaling: bool
    n_freq: int  # Number of frequencies
    fourier_scale: float  # Fourier scale
    target_modules: List[str]  # List of target module namesWW


# Define the Fourier model class
class Fourier_Lowrank_Model(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        trainable_scaling=False,
        target_modules,
        n_freq=1000,
        fourier_scale=100,
    ):
        super().__init__()
        self.wrapped_model: nn.Module = model  # The wrapped base model
        self.n_freq = n_freq  # Number of frequencies
        self.fourier_scale = fourier_scale  # Fourier scale
        self.target_modules = target_modules  # List of target module names
        self.parameterized_modules = []  # List to store parameterized modules
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.trainable_scaling = trainable_scaling

        # Create a FourierConfig object to store the configuration
        self._config = Fourier_Lowrank_Config(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            trainable_scaling=trainable_scaling,
            target_modules=target_modules,
            n_freq=n_freq,
            fourier_scale=fourier_scale,
        )

        # Patch the forward method to use the wrapped model's forward method
        self.forward = self.wrapped_model.forward

        # Convert target_modules to a list if it's a string
        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        # Iterate over the named modules of the wrapped model
        for module_name, module in self.wrapped_model.named_modules():
            # Skip if the module is not an instance of nn.Linear
            if not isinstance(module, nn.Linear):
                continue

            # Skip if the module name doesn't contain any of the target module names
            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"Reparameterized module: {module_name}")
            # Create a new FourierLinear module with the same parameters as the original module
            new_module = Fourier_Lowrank_Linear(
                module.in_features,
                module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                trainable_scaling=trainable_scaling,
                n_freq=n_freq,
                fourier_scale=fourier_scale,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

            # Remove the weight parameter from the original module and delete it
            module.weight = None
            del module

            # Get the parent module and replace the original module with the new FourierLinear module
            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        # Clear the GPU cache
        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        # Get the parent module given the module name
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def save_pretrained(self, path, max_shard_size="100GB"):
        # Save the pretrained model and configuration
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "fourier_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        # Load the pretrained model and configuration
        with open(os.path.join(path, "fourier_config.json"), "r") as f:
            fourier_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        if "keep_original" in fourier_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {fourier_config['keep_original']}")
            fourier_config["lora_only"] = not fourier_config.pop("keep_original")
            fourier_config["keep_original_weights"] = not fourier_config["lora_only"]

        model = cls(base_model, **fourier_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


# Define the FourierLinear module class
class Fourier_Lowrank_Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: float = 32,
        lora_dropout: float = 0.0,
        trainable_scaling: bool = False,
        n_freq,
        fourier_scale,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Initialize the bias parameter if bias is True
        if bias:
            self.bias = Parameter(
                torch.zeros(
                    out_features, device=device, dtype=dtype, requires_grad=True
                )
            )
            a = 1 / math.sqrt(out_features)
            nn.init.uniform_(self.bias, -a, a)
        else:
            self.register_parameter("bias", None)

        self.in_features = in_features  # Number of input features
        self.out_features = out_features  # Number of output features
        self.n_freq = n_freq  # Number of frequencies
        self.fourier_scale = fourier_scale  # Fourier scale
        self.device = device  # Device
        self.dtype = dtype  # Data type
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.trainable_scaling = trainable_scaling

        ## Low rank initialization
        if r <= 0:
            raise ValueError("r must be positive.")
        else:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.zeros_(self.lora_B.weight)

            if trainable_scaling:
                self.scaling = nn.Parameter(
                    torch.tensor([1.0], device=device, dtype=dtype), requires_grad=True
                )
            else:
                self.scaling = self.lora_alpha / self.r

        # Fourier layer initialization
        self.indices = torch.randperm(self.in_features * self.out_features)[
            : self.n_freq
        ]
        self.indices = torch.stack(
            [self.indices % self.in_features, self.indices % self.out_features], dim=0
        )
        self.spectrum = nn.Parameter(torch.randn(self.n_freq), requires_grad=True)

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling

    def forward(self, x: Tensor):
        """
        Forward pass of the FourierLinear module.
        Input x : [..., in_dim] and Output [..., out_dim]
        """
        out = 0

        spectrum = self.spectrum
        indices = self.indices
        fourier_scale = self.fourier_scale

        # Create a dense tensor to store the spectrum
        dense_s = torch.zeros(
            (self.in_features, self.out_features),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )

        # Assign the spectrum values to the corresponding indices in the dense tensor
        dense_s[indices[0, :], indices[1, :]] = spectrum

        # Convert the dense tensor to float32 if the spectrum is in bfloat16 format
        if spectrum.dtype == torch.bfloat16:
            dense_s = dense_s.to(torch.float32)

        # Perform inverse Fourier transform to obtain the weight matrix
        delta_w = torch.fft.ifft2(dense_s).real * fourier_scale

        # Move the input and weight matrix to the same device and data type as the spectrum
        x = x.to(device=spectrum.device, dtype=spectrum.dtype)
        delta_w = delta_w.to(device=spectrum.device, dtype=spectrum.dtype)

        # Perform matrix multiplication between the input and weight matrix
        out = torch.einsum("ijk,kl->ijl", x, delta_w)

        ## Low rank
        if self.r > 0:
            out += (
                self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()
            )

        return out

    def extra_repr(self) -> str:
        # Return a string representation of the module's parameters
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, n_freq={self.n_freq}, "
            f"bias={self.bias is not None}"
        )
