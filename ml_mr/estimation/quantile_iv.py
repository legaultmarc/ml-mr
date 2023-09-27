"""
Implementation of an IV method based on estimating quantiles of the exposure
distribution.
"""

import argparse
import json
import os
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..logging import info
from ..utils import default_validate_args, parse_project_and_run_name
from ..utils.conformal import (
    NONCONFORMITY_MEASURES_TYPE, NONCONFORMITY_MEASURES,
    nonconformity_sqr, estimate_q_hat
)
from ..utils.models import MLP, OutcomeMLPBase
from ..utils.quantiles import QuantileLossMulti
from ..utils.training import train_model
from .core import (IVDataset, IVDatasetWithGenotypes, MREstimator,
                   MREstimatorWithUncertainty)

# Default values definitions.
# fmt: off
DEFAULTS = {
    "n_quantiles": 5,
    "conformal_score": "sqr",
    "conformal_alpha_level": 0.1,
    "exposure_hidden": [128, 64],
    "outcome_hidden": [64, 32],
    "exposure_learning_rate": 5e-4,
    "outcome_learning_rate": 5e-4,
    "exposure_batch_size": 10_000,
    "outcome_batch_size": 10_000,
    "exposure_max_epochs": 1000,
    "outcome_max_epochs": 1000,
    "exposure_weight_decay": 1e-4,
    "outcome_weight_decay": 1e-4,
    "exposure_add_input_batchnorm": False,
    "outcome_add_input_batchnorm": False,
    "accelerator": "gpu" if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0
    ) else "cpu",
    "validation_proportion": 0.2,
    "output_dir": "quantile_iv_estimate",
}
# fmt: on


class ExposureQuantileMLP(MLP):
    def __init__(
        self,
        n_quantiles: int,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()],
    ):
        """The model will predict q quantiles."""
        assert n_quantiles >= 3
        self.quantiles = torch.tensor([
            (i + 1) / (n_quantiles + 1) for i in range(n_quantiles)]
        )

        loss = QuantileLossMulti(self.quantiles)

        super().__init__(
            input_size=input_size,
            hidden=hidden,
            out=n_quantiles,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations,
            lr=lr,
            weight_decay=weight_decay,
            loss=loss
        )

    def on_fit_start(self) -> None:
        self.loss.quantiles = self.loss.quantiles.to(  # type: ignore
            device=self.device
        )
        return super().on_fit_start()

    def _step(self, batch, batch_index, log_prefix):
        x, _, ivs, covars = batch

        x_hat = self.forward(
            torch.hstack([tens for tens in (ivs, covars) if tens.numel() > 0])
        )

        loss = self.loss(x_hat, x)
        self.log(f"exposure_{log_prefix}_loss", loss)
        return loss


class OutcomeMLP(OutcomeMLPBase):
    def __init__(
        self,
        exposure_network: ExposureQuantileMLP,
        input_size: int,
        hidden: Iterable[int],
        lr: float,
        weight_decay: float = 0,
        sqr: bool = False,
        add_input_layer_batchnorm: bool = False,
        add_hidden_layer_batchnorm: bool = False,
        activations: Iterable[nn.Module] = [nn.GELU()]
    ):
        super().__init__(
            exposure_network=exposure_network,
            input_size=input_size,
            hidden=hidden,
            lr=lr,
            weight_decay=weight_decay,
            sqr=sqr,
            add_input_layer_batchnorm=add_input_layer_batchnorm,
            add_hidden_layer_batchnorm=add_hidden_layer_batchnorm,
            activations=activations
        )

    def forward(  # type: ignore
        self,
        ivs: torch.Tensor,
        covars: Optional[torch.Tensor],
        taus: Optional[torch.Tensor] = None
    ):
        """Forward pass throught the exposure and outcome models."""
        if self.hparams.sqr:  # type: ignore
            assert taus is not None, "Need quantile samples if SQR enabled."

        # x is the input to the exposure model.
        exposure_net_xs = torch.hstack(
            [tens for tens in (ivs, covars) if tens is not None]
        )

        with torch.no_grad():
            x_hat = torch.mean(  # type: ignore
                self.exposure_network.forward(exposure_net_xs),
                axis=1,
                keepdim=True
            )

        y_hat = self.mlp(
            torch.hstack([tens for tens in (
                x_hat, covars, taus
            ) if tens is not None])
        )

        return y_hat


class QuantileIVEstimator(MREstimator):
    def __init__(
        self,
        exposure_network: ExposureQuantileMLP,
        outcome_network: OutcomeMLP,
    ):
        self.exposure_network = exposure_network
        self.outcome_network = outcome_network

    def iv_reg_function(
        self, x: torch.Tensor, covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.outcome_network.x_to_y(x, covars)

    @classmethod
    def from_results(cls, dir_name: str) -> "QuantileIVEstimator":
        exposure_network = ExposureQuantileMLP.load_from_checkpoint(
            os.path.join(dir_name, "exposure_network.ckpt")
        )

        outcome_network = OutcomeMLP.load_from_checkpoint(
            os.path.join(dir_name, "outcome_network.ckpt"),
            exposure_network=exposure_network
        )

        outcome_network.eval()  # type: ignore

        if outcome_network.hparams.sqr:  # type: ignore
            with open(os.path.join(dir_name, "meta.json"), "rt") as f:
                meta = json.load(f)

            return QuantileIVEstimatorWithUncertainty(
                exposure_network,
                outcome_network,
                meta
            )

        return cls(exposure_network, outcome_network)


class QuantileIVEstimatorWithUncertainty(
    QuantileIVEstimator,
    MREstimatorWithUncertainty
):
    def __init__(
        self,
        exposure_network: ExposureQuantileMLP,
        outcome_network: OutcomeMLP,
        meta: dict
    ):
        self.exposure_network = exposure_network
        self.outcome_network = outcome_network

        # Conformal prediction adjustment.
        self.conformal_score: NONCONFORMITY_MEASURES_TYPE =\
            meta["conformal_score"]
        assert self.conformal_score in NONCONFORMITY_MEASURES

        self.meta = meta

    def iv_reg_function(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None,
        alpha: float = 0.1
    ) -> torch.Tensor:
        if self.conformal_score == "sqr":
            alpha = self.meta["conformal_alpha_level"]
            pred = []
            with torch.no_grad():
                for tau in [alpha / 2, 0.5, 1 - alpha / 2]:
                    cur_y = self.outcome_network.x_to_y(x, covars, tau)
                    pred.append(cur_y)

            # n x y dimension x 3 for the values in tau.
            pred_tens = torch.stack(pred, dim=2)

            # Conformal prediction adjustment if set.
            pred_tens[:, :, 0] -= self.meta["q_hat"]
            pred_tens[:, :, 2] += self.meta["q_hat"]

            return pred_tens

        raise NotImplementedError()


def main(args: argparse.Namespace) -> None:
    """Command-line interface entry-point."""
    default_validate_args(args)

    # Prepare train and validation datasets.
    # There is theoretically a little bit of leakage here because the histogram
    # or quantiles will be calculated including the validation dataset.
    # This should not have a big impact...
    dataset = IVDatasetWithGenotypes.from_argparse_namespace(args)

    # Automatically add the model hyperparameters.
    kwargs = {k: v for k, v in vars(args).items() if k in DEFAULTS.keys()}

    fit_quantile_iv(
        n_quantiles=args.q,
        dataset=dataset,
        fast=args.fast,
        wandb_project=args.wandb_project,
        **kwargs,
    )


def train_exposure_model(
    n_quantiles: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    input_size: int,
    output_dir: str,
    hidden: List[int],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    add_input_batchnorm: bool,
    max_epochs: int,
    accelerator: Optional[str] = None,
    wandb_project: Optional[str] = None
) -> float:
    info("Training exposure model.")
    model = ExposureQuantileMLP(
        n_quantiles=n_quantiles,
        input_size=input_size,
        hidden=hidden,
        lr=learning_rate,
        weight_decay=weight_decay,
        add_input_layer_batchnorm=add_input_batchnorm,
        add_hidden_layer_batchnorm=True,
    )

    return train_model(
        train_dataset,
        val_dataset,
        model=model,
        monitored_metric="exposure_val_loss",
        output_dir=output_dir,
        checkpoint_filename="exposure_network.ckpt",
        batch_size=batch_size,
        max_epochs=max_epochs,
        accelerator=accelerator,
        wandb_project=wandb_project
    )


def train_outcome_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    exposure_network: ExposureQuantileMLP,
    output_dir: str,
    hidden: List[int],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    add_input_batchnorm: bool,
    max_epochs: int,
    conformal_score: Optional[NONCONFORMITY_MEASURES_TYPE],
    accelerator: Optional[str] = None,
    wandb_project: Optional[str] = None
) -> float:
    info("Training outcome model.")
    n_covars = train_dataset[0][3].numel()
    if (
        conformal_score is None or
        conformal_score == "sqr" or
        conformal_score == "residual-aux-nn"
    ):
        model = OutcomeMLP(
            exposure_network=exposure_network,
            input_size=1 + n_covars,
            lr=learning_rate,
            weight_decay=weight_decay,
            hidden=hidden,
            add_input_layer_batchnorm=add_input_batchnorm,
            sqr=conformal_score == "sqr"
        )

    elif conformal_score == "gaussian-nn":
        # TODO
        raise ValueError("TODO")

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
        wandb_project=wandb_project,
    )


def fit_quantile_iv(
    dataset: IVDataset,
    n_quantiles: int = DEFAULTS["n_quantiles"],  # type: ignore
    stage2_dataset: Optional[IVDataset] = None,  # type: ignore
    output_dir: str = DEFAULTS["output_dir"],  # type: ignore
    validation_proportion: float = DEFAULTS["validation_proportion"],  # type: ignore # noqa: E501
    fast: bool = False,
    conformal_score: Optional[NONCONFORMITY_MEASURES_TYPE] = DEFAULTS["conformal_score"],  # type: ignore # noqa: E501
    conformal_alpha_level: Optional[float] = DEFAULTS["conformal_alpha_level"],  # type: ignore # noqa: E501
    exposure_hidden: List[int] = DEFAULTS["exposure_hidden"],  # type: ignore
    exposure_learning_rate: float = DEFAULTS["exposure_learning_rate"],  # type: ignore # noqa: E501
    exposure_weight_decay: float = DEFAULTS["exposure_weight_decay"],  # type: ignore # noqa: E501
    exposure_batch_size: int = DEFAULTS["exposure_batch_size"],  # type: ignore
    exposure_max_epochs: int = DEFAULTS["exposure_max_epochs"],  # type: ignore
    exposure_add_input_batchnorm: bool = DEFAULTS["exposure_add_input_batchnorm"],  # type: ignore # noqa: E501
    outcome_hidden: List[int] = DEFAULTS["outcome_hidden"],  # type: ignore
    outcome_learning_rate: float = DEFAULTS["outcome_learning_rate"],  # type: ignore # noqa: E501
    outcome_weight_decay: float = DEFAULTS["outcome_weight_decay"],  # type: ignore # noqa: E501
    outcome_batch_size: int = DEFAULTS["outcome_batch_size"],  # type: ignore
    outcome_max_epochs: int = DEFAULTS["outcome_max_epochs"],  # type: ignore
    outcome_add_input_batchnorm: bool = DEFAULTS["outcome_add_input_batchnorm"],  # type: ignore # noqa: E501
    accelerator: str = DEFAULTS["accelerator"],  # type: ignore
    wandb_project: Optional[str] = None,
) -> QuantileIVEstimator:
    # Create output directory if needed.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Metadata dictionary that will be saved alongside the results.
    meta = dict(locals())
    meta["model"] = "quantile_iv"
    meta.update(dataset.exposure_descriptive_statistics())
    del meta["dataset"]  # We don't serialize the dataset.

    covars = dataset.save_covariables(output_dir)

    # Split here into train and val.
    train_dataset, val_dataset = random_split(
        dataset, [1 - validation_proportion, validation_proportion]
    )

    # If there is a separate dataset for stage2, we split it too, otherwise
    # we reuse the stage 1 dataset.
    if stage2_dataset is not None:
        stg2_train_dataset, stg2_val_dataset = random_split(
            stage2_dataset, [1 - validation_proportion, validation_proportion]
        )
    else:
        stg2_train_dataset, stg2_val_dataset = (
            train_dataset, val_dataset
        )

    exposure_val_loss = train_exposure_model(
        n_quantiles=n_quantiles,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_size=dataset.n_exog(),
        output_dir=output_dir,
        hidden=exposure_hidden,
        learning_rate=exposure_learning_rate,
        weight_decay=exposure_weight_decay,
        batch_size=exposure_batch_size,
        add_input_batchnorm=exposure_add_input_batchnorm,
        max_epochs=exposure_max_epochs,
        accelerator=accelerator,
        wandb_project=wandb_project
    )

    meta["exposure_val_loss"] = exposure_val_loss

    exposure_network = ExposureQuantileMLP.load_from_checkpoint(
        os.path.join(output_dir, "exposure_network.ckpt")
    ).eval()  # type: ignore

    exposure_network.freeze()

    if not fast:
        plot_exposure_model(
            exposure_network,
            val_dataset,
            output_filename=os.path.join(
                output_dir, "exposure_model_predictions.png"
            ),
        )

    outcome_val_loss = train_outcome_model(
        train_dataset=stg2_train_dataset,
        val_dataset=stg2_val_dataset,
        exposure_network=exposure_network,
        output_dir=output_dir,
        hidden=outcome_hidden,
        learning_rate=outcome_learning_rate,
        weight_decay=outcome_weight_decay,
        batch_size=outcome_batch_size,
        add_input_batchnorm=outcome_add_input_batchnorm,
        max_epochs=outcome_max_epochs,
        accelerator=accelerator,
        conformal_score=conformal_score,
        wandb_project=wandb_project
    )

    meta["outcome_val_loss"] = outcome_val_loss

    outcome_network = OutcomeMLP.load_from_checkpoint(
        os.path.join(output_dir, "outcome_network.ckpt"),
        exposure_network=exposure_network,
    ).eval()  # type: ignore

    # Training the 2nd stage model copies the exposure net to the GPU.
    # Here, we ensure they're on the same device.
    exposure_network.to(outcome_network.device)

    if conformal_score is not None:
        assert conformal_alpha_level is not None
        fit_conformal(
            outcome_network,
            conformal_score,
            val_dataset,  # type: ignore
            meta,
            alpha=conformal_alpha_level
        )

        estimator: QuantileIVEstimator = QuantileIVEstimatorWithUncertainty(
            exposure_network, outcome_network, meta
        )
    else:
        estimator = QuantileIVEstimator(exposure_network, outcome_network)

    # Save the metadata, estimator statistics and log artifact to WandB if
    # required.
    with open(os.path.join(output_dir, "meta.json"), "wt") as f:
        json.dump(meta, f)

    if not fast:
        save_estimator_statistics(
            estimator,
            covars,
            domain=meta["domain"],
            output_prefix=os.path.join(output_dir, "causal_estimates"),
            alpha=conformal_alpha_level
        )

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


# TODO Change to get_conformal_qhat -> float
def fit_conformal(
    outcome_model: OutcomeMLPBase,
    conformal_score: NONCONFORMITY_MEASURES_TYPE,
    conformal_dataset: IVDataset,
    meta: dict,
    alpha: float = 0.1,
):
    if conformal_score == "sqr":
        # Outcome model was fitted using SQR which we use to get the conformal
        # band.
        conf_scores = nonconformity_sqr(outcome_model, conformal_dataset)
        q_hat = estimate_q_hat(conf_scores, alpha=alpha)
        info(f"Conformal adjustment estimated at q_hat={q_hat}.")

        meta["q_hat"] = q_hat

    elif conformal_score in NONCONFORMITY_MEASURES:
        raise NotImplementedError()

    else:
        raise ValueError(conformal_score)


@torch.no_grad()
def plot_exposure_model(
    exposure_network: ExposureQuantileMLP,
    val_dataset: Dataset,
    output_filename: str
):
    assert hasattr(val_dataset, "__len__")
    dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
    true_x, _, ivs, covariables = next(iter(dataloader))

    input = torch.hstack(
        [tens for tens in (ivs, covariables) if tens.numel() > 0]
    )

    predicted_quantiles = exposure_network(input)

    def identity_line(ax=None, ls='--', *args, **kwargs):
        # see: https://stackoverflow.com/q/22104256/3986320
        ax = ax or plt.gca()
        identity, = ax.plot([], [], ls=ls, *args, **kwargs)

        def callback(axes):
            low_x, high_x = ax.get_xlim()
            low_y, high_y = ax.get_ylim()
            low = min(low_x, low_y)
            high = max(high_x, high_y)
            identity.set_data([low, high], [low, high])

        callback(ax)
        ax.callbacks.connect('xlim_changed', callback)
        ax.callbacks.connect('ylim_changed', callback)
        return ax

    for q in range(predicted_quantiles.size(1)):
        plt.scatter(
            true_x,
            predicted_quantiles[:, q].detach().numpy(),
            label="q={:.2f}".format(exposure_network.quantiles[q].item()),
            s=1,
            alpha=0.2,
        )
    identity_line(lw=1, color="black")
    plt.xlabel("Observed X")
    plt.ylabel("Predicted X (quantiles)")
    plt.legend()

    plt.savefig(output_filename, dpi=400)
    plt.clf()
    plt.close()


def save_estimator_statistics(
    estimator: QuantileIVEstimator,
    covars: Optional[torch.Tensor],
    domain: Tuple[float, float],
    output_prefix: str = "causal_estimates",
    alpha: Optional[float] = None
):
    # Save the causal effect at over the domain.
    xs = torch.linspace(domain[0], domain[1], 500).reshape(-1, 1)

    if estimator.outcome_network.hparams.sqr and alpha:  # type: ignore
        assert isinstance(estimator, QuantileIVEstimatorWithUncertainty)
        ys = estimator.iv_reg_function(xs, covars, alpha=alpha)

        if ys.size(1) != 1:
            raise NotImplementedError(
                "Saving statistics for multidimensional outcome not "
                "implemented yet."
            )

        df = pd.DataFrame(
            torch.hstack((xs, ys[:, 0, :])).numpy(),
            columns=["x", "y_do_x_lower", "y_do_x", "y_do_x_upper"]
        )

    else:
        ys = estimator.iv_reg_function(xs, covars).reshape(-1)
        df = pd.DataFrame({"x": xs.reshape(-1), "y_do_x": ys})

    plt.figure()
    plt.scatter(df["x"], df["y_do_x"], label="Estimated Y | do(X=x)", s=3)

    if "y_do_x_lower" in df.columns:
        # Add the CI on the plot.
        assert alpha is not None
        plt.fill_between(
            df["x"],
            df["y_do_x_lower"],
            df["y_do_x_upper"],
            color="#dddddd",
            zorder=-1,
            label=f"{int((1 - alpha) * 100)}% Prediction interval"
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(f"{output_prefix}.png", dpi=600)
    plt.clf()

    df.to_csv(f"{output_prefix}.csv", index=False)


def configure_argparse(parser) -> None:
    parser.add_argument(
        "--n-quantiles", "-q",
        type=int,
        help="Number of quantiles of the exposure distribution to estimate in "
        "the exposure model.",
        required=True,
    )

    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])

    parser.add_argument(
        "--fast",
        help="Disable plotting and logging of causal effects.",
        action="store_true",
    )

    parser.add_argument(
        "--conformal-score",
        default=DEFAULTS["conformal_score"],
        help="Conformal prediction nonconformity measure.",
        choices=NONCONFORMITY_MEASURES,
        type=str
    )

    parser.add_argument(
        "--conformal-alpha-level",
        default=DEFAULTS["conformal_alpha_level"],
        type=float,
    )

    parser.add_argument(
        "--outcome-type",
        default="continuous",
        choices=["continuous", "binary"],
        help="Variable type for the outcome (binary vs continuous).",
    )

    parser.add_argument(
        "--validation-proportion",
        type=float,
        default=DEFAULTS["validation_proportion"],
    )

    parser.add_argument(
        "--accelerator",
        default=DEFAULTS["accelerator"],
        help="Accelerator (e.g. gpu, cpu, mps) use to train the model. This "
        "will be passed to Pytorch Lightning.",
    )

    parser.add_argument(
        "--wandb-project",
        default=None,
        type=str,
        help="Activates the Weights and Biases logger using the provided "
             "project name. Patterns such as project:run_name are also "
             "allowed."
    )

    MLP.add_mlp_arguments(
        parser,
        "exposure-",
        "Exposure Model Parameters",
        defaults={
            "hidden": DEFAULTS["exposure_hidden"],
            "batch-size": DEFAULTS["exposure_batch_size"],
        },
    )

    MLP.add_mlp_arguments(
        parser,
        "outcome-",
        "Outcome Model Parameters",
        defaults={
            "hidden": DEFAULTS["outcome_hidden"],
            "batch-size": DEFAULTS["outcome_batch_size"],
        },
    )

    IVDatasetWithGenotypes.add_dataset_arguments(parser)


# Standard names for estimators.
estimate = fit_quantile_iv
load = QuantileIVEstimator.from_results
