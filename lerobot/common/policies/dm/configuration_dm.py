from dataclasses import dataclass, field
from lerobot.common.optim.optimizers import AdamConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)


@PreTrainedConfig.register_subclass("dm")
@dataclass
class DMConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 2
    n_action_steps: int = 2

    action_time_proj_width: int = 1024
    max_action_dim: int = 32
    max_state_dim : int = 32
    num_steps : int = 10

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )


    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10


    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    optimizer_lr_backbone: float = 1e-5
    freeze_vision_encoder:bool = True

    def __post_init__(self):
        super().__post_init__()


    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr = self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )


    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:#list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None




