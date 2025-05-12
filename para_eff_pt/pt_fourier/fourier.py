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


@dataclass
class FourierConfig:
    n_freq: int
    fourier_scale: float
    target_modules: List[str]


class FourierModel(torch.nn.Module):
    def __init__(self, model, *, target_modules, n_freq, fourier_scale):

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.n_freq = n_freq
        self.fourier_scale = fourier_scale
        self.target_modules = target_modules
        self.parameterized_modules = []

        self._config = FourierConfig(
            target_modules=target_modules, n_freq=n_freq, fourier_scale=fourier_scale
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print(f"Reparameterized module: {module_name}")
            # self.parameterized_modules.append(module_name)
            new_module = FourierLinear(
                module.in_features,
                module.out_features,
                n_freq=n_freq,
                fourier_scale=fourier_scale,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

            module.weight = None
            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def save_pretrained(self, path, max_shard_size="100GB"):
        # TODO
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "fourier_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        # TODO
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


class FourierLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        n_freq,
        fourier_scale,
        bias=True,
        device=None,
        dtype=None,
    ):

        super().__init__()

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

        self.in_features = in_features
        self.out_features = out_features
        self.n_freq = n_freq
        self.fourier_scale = fourier_scale
        self.device = device
        self.dtype = dtype

        ## fourier layer
        self.indices = torch.randperm(self.in_features * self.out_features)[
            : self.n_freq
        ]
        self.indices = torch.stack(
            [self.indices % self.in_features, self.indices % self.out_features], dim=0
        )
        self.spectrum = nn.Parameter(torch.randn(self.n_freq), requires_grad=True)

    def forward(self, x: Tensor):
        """
        Input x : [..., in_dim] and Output [..., out_dim]
        """

        out = 0

        spectrum = self.spectrum
        indices = self.indices
        fourier_scale = self.fourier_scale

        # # sparse_coo_tensor will lead to similar GPU cost but longer training time, so it is not recommended
        # delta_w = (
        #     torch.fft.ifft2(
        #         torch.sparse_coo_tensor(
        #             indices,
        #             spectrum,
        #             [self.in_features, self.out_features],
        #             dtype=torch.complex64,
        #             device=spectrum.device,
        #         ).to_dense()
        #     ).real
        #     * fourier_scale
        # )

        dense_s = torch.zeros(
            (self.in_features, self.out_features),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )

        dense_s[indices[0, :], indices[1, :]] = spectrum

        if spectrum.dtype == torch.bfloat16:
            dense_s = dense_s.to(torch.float32)

        delta_w = torch.fft.ifft2(dense_s).real * fourier_scale

        x = x.to(device=spectrum.device, dtype=spectrum.dtype)
        delta_w = delta_w.to(device=spectrum.device, dtype=spectrum.dtype)

        out = torch.einsum("ijk,kl->ijl", x, delta_w)

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, n_freq={self.n_freq}, "
            f"bias={self.bias is not None}"
        )
