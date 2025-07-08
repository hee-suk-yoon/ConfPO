# flake8: noqa

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# There is a circular import in the PPOTrainer if we let isort sort these
from typing import TYPE_CHECKING
from ..import_utils import _LazyModule, is_diffusers_available, OptionalDependencyNotAvailable

_import_structure = {
    "utils": [
        "AdaptiveKLController",
        "FixedKLController",
        "ConstantLengthDataset",
        "DataCollatorForCompletionOnlyLM",
        "RunningMoments",
        "disable_dropout_in_model",
        "peft_module_casting_to_bf16",
        "RichProgressCallback",
    ],
    "dpo_config": ["DPOConfig"],
    "dpo_trainer": ["DPOTrainer"],
    "cpo_config": ["CPOConfig"],
    "cpo_trainer": ["CPOTrainer"],
    "model_config": ["ModelConfig"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "base": ["BaseTrainer"],
    "simpo_config": ["SimPoConfig"], # added by esyoon 2024-07-04-16:55:14
    "simpo_trainer": ["SimPoTrainer"], # added by esyoon 2024-07-04-16:55:16
}

if TYPE_CHECKING:
    # isort: off
    from .utils import (
        AdaptiveKLController,
        FixedKLController,
        ConstantLengthDataset,
        DataCollatorForCompletionOnlyLM,
        RunningMoments,
        disable_dropout_in_model,
        peft_module_casting_to_bf16,
        RichProgressCallback,
    )

    # isort: on

    from .base import BaseTrainer

    from .dpo_config import DPOConfig
    from .dpo_trainer import DPOTrainer
    from .cpo_config import CPOConfig
    from .cpo_trainer import CPOTrainer
    from .model_config import ModelConfig
    from .sft_config import SFTConfig
    from .sft_trainer import SFTTrainer
    from .simpo_config import SimPoConfig 
    from .simpo_trainer import SimPoTrainer 

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
