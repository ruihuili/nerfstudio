# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Multi Layer Perceptron
"""
from typing import Optional, Set, Tuple

import torch
import torch.nn.utils.prune as prune
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.base_field_component import FieldComponent


class MLP(FieldComponent):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Ouput layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
    ) -> None:

        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def get_FI_scores(self, grad_dict, name_str="field.mlp_head", inverse=False):
        """Get pre-recorded gradients of a layer and compute FI score approximation"""
        f_score = []
        for l in range(self.num_layers):
            for n, g in grad_dict.items():
                # name of layers for the grad dict can be make better
                if name_str in n and str(l) == n.split(".")[-2] and "weight" == n.split(".")[-1]:
                    if inverse:
                        f_score.append(1 - g**2)
                    else:
                        f_score.append(g**2)
        return f_score

    def get_weight_as_scores(self, layers):
        return [i.weight.data.detach().cpu() for i in layers]

    def get_nn_wide_score(self, score_in, target_ratio):

        """Input score in original shape"""

        flatten_score = torch.cat([item.flatten() for item in score_in])

        # record original shape of input for later recovery
        score_ori_shape = [list(l.shape) for l in score_in]

        # getting the order of each element when sorted
        d = torch.searchsorted(flatten_score.sort().values, flatten_score)  # .reshape(layer.weight.shape)

        flatten_score = torch.where(d >= target_ratio * torch.numel(d), d, torch.zeros(d.shape))

        print("Proportion of non-pruned connections ", torch.count_nonzero(flatten_score) / len(flatten_score))

        # put back to the original shape of input scores

        ele_count = 0  # element counter to reshape back flattened scores to the same shape of weights/grads
        all_score = []

        assert len(score_in) == len(score_ori_shape)
        for i, s in enumerate(score_in):

            # count number of elements in this layer
            ele_add = 1
            for dim_i in score_ori_shape[i]:

                ele_add = dim_i * ele_add
                print("dim to add ", ele_add)
            all_score.append(flatten_score[ele_count : ele_count + ele_add].reshape(score_ori_shape[i]))

            ele_count += ele_add

        assert ele_count == len(flatten_score)

        return all_score

    def prune_mlp(self, grad_dict, param_name, ratio=0.6, scheme="InverseFI") -> None:
        """
        Prune multi-layer perceptron based on importance scores/L2/Uniform.

        Idea is to get scores for all weights and prune based on the scores

        2 Scoring regimes: layer-wise and network-wise i.e. pruning $target_ratio of a layer v.s the network

        Implementaiton flow: get all scores (weight/ FI) as a mask score_all, edit the mask to reflect target ratio,
        and then apply the mask on to pytorch pruning func as attribute importance_scores
        """
        if "FI" in scheme:
            inv = True if "InverseFI" in scheme else False
            print("FI inverse? ", inv, " ratio ", ratio)
            score_all = self.get_FI_scores(grad_dict, name_str=param_name, inverse=inv)
        else:
            print("magnitude based scoring")
            score_all = self.get_weight_as_scores(self.layers)

        for i, j in enumerate(score_all):
            print(" score_all items ", i, type(j))

        if "NN_wide" in scheme:
            """A scheme to select the top FI scores to preserve and prune the rest"""
            assert score_all is not None
            score_all = self.get_nn_wide_score(score_all, ratio)

        for i, layer in enumerate(self.layers):

            if "FI" in scheme or scheme == "NN_wide_magnitude":
                prune.l1_unstructured(layer, name="weight", amount=ratio, importance_scores=score_all[i])
            elif scheme == "Random":
                randp = torch.randperm(torch.numel(layer.weight)).reshape(layer.weight.shape)
                # importance scores used for pruning -- values are 0: prune; 1: no-prune
                score = torch.where(
                    randp >= ratio * torch.numel(layer.weight),
                    torch.ones(layer.weight.shape),
                    torch.zeros(layer.weight.shape),
                )
                prune.l1_unstructured(layer, name="weight", amount=ratio, importance_scores=score)

            elif "L1" in scheme:
                # removing the specified `amount` of (currently unpruned) units with the lowest L1-norm.
                # is this L1?
                prune.l1_unstructured(layer, name="weight", amount=ratio)
            elif "L2" in scheme:
                # TODO might need double check
                print("l2 norm")
                prune.ln_structured(layer, name="weight", amount=ratio, n=2, dim=1)
            else:
                print("Undefined scheme")

            prune.remove(layer, "weight")
            return

    def forward(self, in_tensor: TensorType["bs":..., "in_dim"]) -> TensorType["bs":..., "out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x
