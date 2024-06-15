# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import pdb
import math
import warnings
from typing import Any, List, Optional, Union
from args import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

args_l = get_args()

def FFT_SHIFT(matrix):
        m_clone = matrix.clone()
        m,n = m_clone.shape
        m = int(m / 2)
        n = int(n / 2)

        for i in range(m):
            for j in range(n):
                m_clone[i][j] = matrix[m+i][n+j]
                m_clone[m+i][n+j] = matrix[i][j]
                m_clone[m+i][j] = matrix[i][j+n]
                m_clone[i][j+n] = matrix[m+i][j]
        return m_clone
        
class FourierLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ["spectrum"]
    # All names of other parameters that may contain adapter-related parameters
    # other_param_names = ("rank", "dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.n_frequency = {}
        # self.lora_alpha = {}
        self.scale = {}
        # self.dropout = nn.ModuleDict({})
        self.spectrum = nn.ParameterDict({})
        self.indices = {}
        # For Embedding layer
        # self.lora_embedding_A = nn.ParameterDict({})
        # self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        # elif isinstance(base_layer, nn.Conv2d):
        #     in_features, out_features = base_layer.in_channels, base_layer.out_channels
        # elif isinstance(base_layer, nn.Embedding):
        #     in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        # elif isinstance(base_layer, Conv1D):
        #     in_features, out_features = (
        #         base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
        #     )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features


    def update_layer(self, adapter_name, n_frequency, scale, init_fourier_weights=None):
        # print('\033 args in layer \033[0m', args)
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        self.n_frequency[adapter_name] = n_frequency
        self.scale[adapter_name] = scale
        if n_frequency > 0:  

            if args_l.set_bias:
                d = self.in_features
                center_frequency = args_l.fc  # D_0 
                width = args_l.width   # W
                order = 2 # 2n
                rows, cols = np.ogrid[:d, :d]
                distance = np.sqrt((rows - d / 2)**2 + (cols - d / 2)**2)
                mask_gs = torch.tensor(np.exp(-(distance * width / (distance**2 - center_frequency**2))**(-2)))
                mask_gs = FFT_SHIFT(mask_gs)
                samples = torch.multinomial(mask_gs.view(-1),1000, replacement=True)
                samples = torch.stack([samples // d, samples % d], dim=1).T
                self.indices[adapter_name] = samples
                print('\033[32m Using frequency bias... \033[0m')

            # print('\033[32m new_peft_official\033[0m')
            elif args_l.share_entry:
                self.indices[adapter_name] = torch.randperm(self.in_features * self.in_features,generator=torch.Generator().manual_seed(args_l.entry_seed))[:n_frequency]
                print('\033[32m Using shared entry... \033[0m')
            else:
                self.indices[adapter_name] = torch.randperm(self.in_features * self.in_features)[:n_frequency]
                
            self.indices[adapter_name] = torch.stack([self.indices[adapter_name] // self.in_features, self.indices[adapter_name] % self.in_features], dim=0)
            self.spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)
  
        # if init_fourier_weights == "loftq":
        #     self.loftq_init(adapter_name)
        # elif init_fourier_weights:
        #     self.reset_lora_parameters(adapter_name, init_fourier_weights)

        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def reset_fourier_parameters(self, adapter_name, init_fourier_weights):
        if init_fourier_weights is False:
            return

        if adapter_name in self.spectrum.keys():
            if init_fourier_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.spectrum[adapter_name].weight, a=math.sqrt(5))
            elif init_fourier_weights.lower() == "gaussian":
                nn.init.normal_(self.spectrum[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_fourier_weights=}")
        if adapter_name in self.spectrum.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.spectrum[adapter_name])

    # def loftq_init(self, adapter_name):
    #     from peft.utils.loftq_utils import loftq_init

    #     weight = self.get_base_layer().weight
    #     kwargs = {
    #         "num_bits": self.kwargs.get("loftq_bits", 4),
    #         "reduced_rank": self.r[adapter_name],
    #         "num_iter": self.kwargs.get("loftq_iter", 1),
    #     }

    #     qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
    #     if adapter_name in self.lora_A.keys():
    #         # initialize A the same way as the default for nn.Linear and B to zero
    #         self.lora_A[adapter_name].weight.data = lora_A
    #         self.lora_B[adapter_name].weight.data = lora_B
    #     if adapter_name in self.lora_embedding_A.keys():
    #         # initialize a the same way as the default for nn.linear and b to zero
    #         self.lora_embedding_A[adapter_name].weight.data = lora_A
    #         self.lora_embedding_B[adapter_name].weight.data = lora_B
    #     self.get_base_layer().weight.data = qweight

    # def set_scale(self, adapter, scale):
    #     if adapter not in self.scaling:
    #         # Ignore the case where the adapter is not in the layer
    #         return
    #     self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    # def scale_layer(self, scale: float) -> None:
    #     if scale == 1:
    #         return

    #     for active_adapter in self.active_adapters:
    #         if active_adapter not in self.lora_A.keys():
    #             continue

    #         self.scaling[active_adapter] *= scale

    # def unscale_layer(self, scale=None) -> None:
    #     for active_adapter in self.active_adapters:
    #         if active_adapter not in self.lora_A.keys():
    #             continue

    #         if scale is None:
    #             self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
    #         else:
    #             self.scaling[active_adapter] /= scale


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------



class Linear(nn.Module, FourierLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_frequency: int = 0,
        scale: float = 0.1,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_fourier_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        FourierLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_frequency, scale, init_fourier_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.spectrum.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.spectrum.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.spectrum[adapter].device
        dtype = self.spectrum[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        spectrum = self.spectrum[adapter]
        indices = self.indices[adapter].to(spectrum.device)
        

        weight = torch.fft.ifft2(torch.sparse.FloatTensor(indices, spectrum, [self.in_features, self.in_features]).to_dense()).real * 300
        if cast_to_fp32:
            weight = weight.float()

        output_tensor = weight

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.weight[adapter] = weight.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.spectrum.keys():
                    continue
                
                spectrum = self.spectrum[active_adapter]
                indices = self.indices[active_adapter].to(spectrum.device)
                scale = self.scale[active_adapter]

                ## sparse_coo_tensor will lead to similar GPU cost but longer training time, so it is not recommended
                # delta_w = torch.fft.ifft2(torch.sparse_coo_tensor(indices, spectrum, [self.in_features, self.in_features],  
                                            # dtype=spectrum.dtype, device=spectrum.device).to_dense()).real * scale

                dense_s = torch.zeros((self.in_features, self.in_features), dtype=spectrum.dtype, device='cuda')
                dense_s[indices[0, :], indices[1, :]] = spectrum
            
                if spectrum.dtype == torch.bfloat16:
                    dense_s = dense_s.to(torch.float16)

                delta_w = torch.fft.ifft2(dense_s).real * scale
                x, delta_w = x.to(spectrum.dtype), delta_w.to(spectrum.dtype)
                result += torch.einsum('ijk,kl->ijl', x, delta_w)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourier." + rep