"""Typed configuration objects used by training scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class RandomTrainingConfig:
    """Configuration for random-network training runs."""

    model: str
    dataset: str
    targets: Optional[Sequence[str]]
    feature_scaler: str
    label_scaler: str
    use_log: bool
    use_area_root: bool
    n_hidden: int
    activation: str
    alpha: float
    gamma: float
    n_layers: int
    n_ensemble: int
    sc_mode: str
    rsc_prob: float
    n_folds: int
    n_blocks: int
    direct_link: bool
    base_dir: str
    output_dir: Optional[str]
    random_state: int
    n_seeds: int
    no_cuda: bool
    no_save_model: bool

    @classmethod
    def from_namespace(cls, args: Any) -> "RandomTrainingConfig":
        return cls(
            model=args.model,
            dataset=args.dataset,
            targets=args.targets,
            feature_scaler=args.feature_scaler,
            label_scaler=args.label_scaler,
            use_log=args.use_log,
            use_area_root=args.use_area_root,
            n_hidden=args.n_hidden,
            activation=args.activation,
            alpha=args.alpha,
            gamma=args.gamma,
            n_layers=args.n_layers,
            n_ensemble=args.n_ensemble,
            sc_mode=args.sc_mode,
            rsc_prob=args.rsc_prob,
            n_folds=args.n_folds,
            n_blocks=args.n_blocks,
            direct_link=args.direct_link,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            random_state=args.random_state,
            n_seeds=args.n_seeds,
            no_cuda=args.no_cuda,
            no_save_model=args.no_save_model,
        )

    def with_seed(self, seed: int) -> "RandomTrainingConfig":
        return replace(self, random_state=seed)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
