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
Evaluation utils
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import yaml
from rich.console import Console

from nerfstudio.configs import base_config as cfg
from nerfstudio.pipelines.base_pipeline import Pipeline

CONSOLE = Console(width=120)


def eval_load_checkpoint(config: cfg.TrainerConfig, pipeline: Pipeline) -> Path:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    """
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path, loaded_state


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    prune_ratio: float = 0.6,
    prune_scheme: Literal["Uniform", "Fisher", "InverseFI", "L2_norm"] = "Uniform",
) -> Tuple[cfg.Config, Pipeline, Path, float, str]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test datset into memory
            'inference': does not load any dataset into memory


    Returns:
        Loaded config, pipeline module, and corresponding checkpoint.
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.trainer.load_dir = config.get_checkpoint_dir()
    config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    print("before loading checkpt")
    checkpoint_path, loaded_state = eval_load_checkpoint(config.trainer, pipeline)
    
    def get_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    # import torchvision.models as models

    # from ptflops import get_model_complexity_info

    # with torch.cuda.device(0):
    #     # net = models.densenet161()
    #     macs, params = get_model_complexity_info(pipeline.model, (2, 3), as_strings=True,
    #                                                 print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # for name, param, in pipeline.model.field.named_parameters():
        #     print(name, tuple(param.shape), type(param))
        #     macs, params = get_model_complexity_info(param, tuple(param.shape), as_strings=True,
        #                                             print_per_layer_stat=True, verbose=True)
        #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    # prune if its nerfacto
    for name, param, in pipeline.model.field.named_parameters():
        print(name, param.shape)
    # print(get_size(pipeline.model), get_size(pipeline.model.field), get_size(pipeline.model.proposal_networks))

    c_before = sum([torch.count_nonzero(l.weight) for l in pipeline.model.field.mlp_base.layers])
    
    pipeline.model.field.mlp_base.prune_mlp(loaded_state["grads"], "field.mlp_base", ratio=prune_ratio, scheme=prune_scheme)
    pipeline.model.field.mlp_head.prune_mlp(loaded_state["grads"], "field.mlp_head", ratio=prune_ratio, scheme=prune_scheme)
    
    c_after = sum([torch.count_nonzero(l.weight) for l in pipeline.model.field.mlp_base.layers])

    # from torchprofile import profile_macs
    # inputs = torch.randn(32768)
    # macs = profile_macs(pipeline.model, inputs)
    # print("after", get_size(pipeline.model),  get_size(pipeline.model.field), get_size(pipeline.model.proposal_networks))
    print("count non-zero before ", c_before, " after ", c_after)

    return config, pipeline, checkpoint_path
