"""
Implementation of the DELiVR algorithm from

He, Ruoyu, Mingyang Liu, Zhaotong Lin, Zhong Zhuang, Xiaotong Shen, and Wei
Pan.  2023. “DeLIVR: A Deep Learning Approach to IV Regression for Testing
Nonlinear Causal Effects in Transcriptome-Wide Association Studies.”
Biostatistics, January. https://doi.org/10.1093/biostatistics/kxac051.

"""

import argparse
import os
import json
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split

from ..utils import _cat, default_validate_args, parse_project_and_run_name
from ..utils.linear import ridge_regression
from ..utils.models import MLP
from ..utils.training import train_model
from .core import (FullBatchDataLoader, IVDataset, IVDatasetWithGenotypes,
                   MREstimator)

DEFAULTS = {
    "hidden": [64, 32],
    "batch_size": 10_000,
    "max_epochs": 1000,
    "output_dir": "delivr_estimate",
    "learning_rate": 5e-3,
    "weight_decay": 1e-4,
    "validation_proportion": 0.2,
    "accelerator": "gpu" if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0
    ) else "cpu",
}


def fit_lin_exposure_model(dataset: Dataset) -> torch.Tensor:
    dl = FullBatchDataLoader(dataset)

    x, _, ivs, covars = next(iter(dl))

    return ridge_regression(_cat(ivs, covars), x, alpha=0)


def main(args: argparse.Namespace) -> None:
    default_validate_args(args)
    dataset = IVDatasetWithGenotypes.from_argparse_namespace(args)

    # Automatically add the model hyperparameters.
    kwargs = {k: v for k, v in vars(args).items() if k in DEFAULTS.keys()}

    fit_delivr(
        dataset=dataset,
        wandb_project=args.wandb_project,
        **kwargs,
    )


def fit_delivr(
    dataset: IVDataset,
    output_dir: str = DEFAULTS["output_dir"],  # type: ignore
    validation_proportion: float = DEFAULTS["validation_proportion"],  # type: ignore # noqa: E501
    hidden: List[int] = DEFAULTS["hidden"],  # type: ignore
    learning_rate: float = DEFAULTS["learning_rate"],  # type: ignore
    weight_decay: float = DEFAULTS["weight_decay"],  # type: ignore
    batch_size: int = DEFAULTS["batch_size"],  # type: ignore
    max_epochs: int = DEFAULTS["max_epochs"],  # type: ignore
    accelerator: str = DEFAULTS["accelerator"],  # type: ignore
    wandb_project: Optional[str] = None
):
    # Create output directory if needed.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Metadata dictionary that will be saved alongside the results.
    meta = dict(locals())
    meta["model"] = "delivr"
    meta.update(dataset.exposure_descriptive_statistics())
    del meta["dataset"]

    # Split here into train and val.
    train_dataset, val_dataset = random_split(
        dataset, [1 - validation_proportion, validation_proportion]
    )

    # Use linear first stage on whole training dataset.
    stg1_betas = fit_lin_exposure_model(train_dataset)

    outcome_val_loss = train_outcome_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        betas=stg1_betas,
        hidden=hidden,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        accelerator=accelerator,
        wandb_project=wandb_project
    )

    meta["outcome_val_loss"] = outcome_val_loss

    estimator = DeLIVREstimator.from_results(output_dir)
    with open(os.path.join(output_dir, "meta.json"), "wt") as f:
        json.dump(meta, f)

    if wandb_project is not None:
        import wandb
        _, run_name = parse_project_and_run_name(wandb_project)
        artifact = wandb.Artifact(
            "results" if run_name is None else f"{run_name}_results",
            type="results"
        )
        artifact.add_dir(output_dir)
        wandb.log_artifact(artifact)
        wandb.finish()

    return estimator


class OutcomeMLP(MLP):
    def __init__(
        self,
        input_size: int,
        betas: torch.Tensor,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float,
        activations: Iterable[nn.Module] = [nn.GELU()]
    ):
        super().__init__(
            input_size=input_size,
            hidden=hidden,
            out=1,
            activations=activations,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.betas = betas

    def on_fit_start(self) -> None:
        self.betas = self.betas.to(self.device)  # type: ignore
        return super().on_fit_start()

    def x_to_y(
        self, x: torch.Tensor, covars: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.mlp(_cat(x, covars))

    def _step(self, batch, batch_index, log_prefix):
        _, y, ivs, covars = batch

        # Get E[X|Z]
        x_hat = _cat(ivs, covars) @ self.betas

        # Get h_hat(x_hat)
        y_hat = self.x_to_y(x_hat, covars)

        loss = F.mse_loss(y_hat, y)

        self.log(f"outcome_{log_prefix}_loss", loss)

        return loss


def train_outcome_model(
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str,
        betas: torch.Tensor,
        hidden: List[int],
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        max_epochs: int,
        accelerator: Optional[str] = None,
        wandb_project: Optional[str] = None
) -> float:
    n_covars = train_dataset[0][3].numel()
    model = OutcomeMLP(
        input_size=n_covars + 1,
        betas=betas,
        hidden=hidden,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return train_model(
        train_dataset,
        val_dataset,
        model=model,
        monitored_metric="outcome_val_loss",
        output_dir=output_dir,
        checkpoint_filename="outcome_network.ckpt",
        batch_size=batch_size,
        max_epochs=max_epochs,
        accelerator=accelerator,
        wandb_project=wandb_project
    )


class DeLIVREstimator(MREstimator):
    def __init__(self, outcome_network: OutcomeMLP):
        self.outcome_network = outcome_network

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.outcome_network.x_to_y(x, covars)

    @classmethod
    def from_results(cls, dir_name: str) -> "DeLIVREstimator":
        outcome_network = OutcomeMLP.load_from_checkpoint(
            os.path.join(dir_name, "outcome_network.ckpt")
        ).to(torch.device("cpu"))

        outcome_network.eval()

        return cls(outcome_network)


def configure_argparse(parser) -> None:
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])

    MLP.add_mlp_arguments(
        parser,
        "Outcome Model Parameters",
        defaults={
            "hidden": DEFAULTS["hidden"],
            "batch-size": DEFAULTS["batch_size"],
        },
    )

    parser.add_argument(
        "--validation-proportion",
        type=float,
        default=DEFAULTS["validation_proportion"],
    )

    parser.add_argument(
        "--wandb-project",
        default=None,
        type=str,
        help="Activates the Weights and Biases logger using the provided "
             "project name. Patterns such as project:run_name are also "
             "allowed."
    )

    IVDatasetWithGenotypes.add_dataset_arguments(parser)


estimate = fit_delivr
load = DeLIVREstimator.from_results