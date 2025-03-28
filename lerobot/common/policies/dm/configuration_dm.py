from dataclasses import dataclass, field
from lerobot.common.optim.optimizers import AdamConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("dm")
@dataclass
class DMConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 1
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )


    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()


    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr = self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )


    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:#list:
        return None
        #return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None




