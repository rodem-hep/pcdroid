from functools import partial
from pathlib import Path
from typing import Mapping

import numpy as np
import pytorch_lightning as pl
import torch as T
import wandb
from nflows import distributions, flows

from src.utils.modules import IterativeNormLayer
from src.utils.plotting import plot_multi_correlations, plot_multi_hists_2
from src.utils.torch_utils import get_sched, to_np


def masked_dequantize(inputs: T.Tensor, mask: list) -> T.Tensor:
    """Add noise to the final dimension of a tensor only where the mask is
    True."""
    inputs = inputs.clone()
    noise = T.randn_like(inputs[..., mask])  # Tested and 1 KDE looks good!
    inputs[..., mask] = inputs[..., mask] + noise
    return inputs


def masked_round(inputs: T.Tensor, mask: list) -> T.Tensor:
    """Round to int the final dimension of a tensor only where the mask is
    True."""
    inputs = inputs.clone()
    inputs[..., mask] = T.round(inputs[..., mask])
    return inputs


class HLVFlow(pl.LightningModule):
    """Neural network to estimate the context inputs for the generator."""

    def __init__(
        self,
        data_dims: tuple,
        gen_ctxt_split: int | None,
        int_dims: list | None,
        invertible_net: partial,
        ctxt_net: partial,
        optimizer: partial,
        sched_config: Mapping,
    ) -> None:
        """
        Parameters
        ----------
        data_dims : tuple
            A tuple with three integers representing the point cloud dimensions,
            the context dimensions, and the number of nodes, respectively.
        gen_ctxt_split:
            Which of the hlv vars are used for generation then context
        int_dims : list
            A list of bools which shows which inputs are intergers
        invertible_net : partial
            The configuration for creating the invertible neural network
        ctxt_net : partial
            For setting up the shared context extractor
        optimizer : partial
            A function to optimize the parameters of the model.
        sched_config : Mapping
            A dictionary with the configuration options for the optimizer scheduler.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # If the flow is contextual we need to split the data dimension
        self.int_dims = int_dims
        if gen_ctxt_split is not None:
            self.inpt_dim = gen_ctxt_split
            self.ctxt_dim = data_dims[1] - gen_ctxt_split
        else:
            self.inpt_dim = data_dims[1]
            self.ctxt_dim = 0

        # The normalisation layer for pre-processing
        self.inpt_normaliser = IterativeNormLayer(self.inpt_dim)
        if self.ctxt_dim:
            self.ctxt_normaliser = IterativeNormLayer(self.ctxt_dim)
            self.ctxt_net = ctxt_net(self.ctxt_dim)

        # The flow itself
        self.flow = flows.Flow(
            invertible_net(
                xz_dim=self.inpt_dim,
                ctxt_dim=self.ctxt_net.outp_dim if self.ctxt_dim else 0,
            ),
            distributions.StandardNormal([self.inpt_dim]),
        )

        # Buffer for holding the ouptuts of the validation epoch
        self.val_step_outs = []

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")

    def _shared_step(self, sample: tuple) -> T.Tensor:
        """Step used by training and validation."""

        # Unpack the input sample
        nodes, mask, hlv, pt = sample

        # Split into the variables for gen and context
        inpt = hlv[..., : self.inpt_dim]
        ctxt = hlv[..., self.inpt_dim :] if self.ctxt_dim else None

        # Add noise to the ctxt to dequantise
        inpt = masked_dequantize(inpt, self.int_dims)

        # Normalise the scale (preprocess)
        inpt = self.inpt_normaliser(inpt)
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)
            ctxt = self.ctxt_net(ctxt)

        # Calculate the negative log liklihood
        nll = -self.flow.log_prob(inpt, ctxt).mean()

        return nll

    def training_step(self, batch: tuple, _batch_idx: int) -> T.Tensor:
        nll = self._shared_step(batch)
        self.log("train/total_loss", nll)

        means = to_np(self.inpt_normaliser.means.squeeze())
        for i in range(len(means)):
            self.log(f"train/means_{i}", means[i])
        vars = to_np(self.inpt_normaliser.vars.squeeze())
        for i in range(len(means)):
            self.log(f"train/vars_{i}", vars[i])

        return nll

    def validation_step(self, batch: tuple, batch_idx: int) -> T.Tensor:
        nll = self._shared_step(batch)
        self.log("valid/total_loss", nll)

        # Run the full generation for the validation set
        nodes, mask, hlv, pt = batch
        inpt = hlv[..., : self.inpt_dim]
        ctxt = hlv[..., self.inpt_dim :] if self.ctxt_dim else None
        n_points = 1 if self.ctxt_dim else len(inpt)
        generated = self.generate(n_points=n_points, ctxt=ctxt)
        self.val_step_outs.append((to_np(inpt), to_np(generated), to_np(ctxt)))

        return nll

    def on_validation_epoch_end(self):
        """Plot histograms of the generated samples compared to the truth."""

        # Combine all outputs
        all_inpt = np.vstack([v[0] for v in self.val_step_outs])
        all_gen = np.vstack([v[1] for v in self.val_step_outs])
        if self.ctxt_dim:
            all_ctxt = np.vstack([v[2] for v in self.val_step_outs])

        # Reshape the generated samples
        all_gen = np.reshape(all_gen, (-1, all_gen.shape[-1]))

        # Get the names of the features used in the network
        feats = self.trainer.datamodule.hparams.data_conf.jetnet_config.jet_features
        inpt_feats = feats[: self.inpt_dim]
        ctxt_feats = feats[self.inpt_dim :]

        # If the jet_type is used in ctxt, then we make seperate plots for each
        used_jet_type = len(ctxt_feats) > 0 and ctxt_feats[-1] == "type"
        if used_jet_type:
            jet_types = ["g", "q", "t", "w", "z"]  # Fixed order based on jetnet
            jet_labels = all_ctxt[:, -1].astype("long")
        else:
            jet_types = ["all"]

        # Cycle through the jet types
        for i, jet_type in enumerate(jet_types):

            # Pull out the events in the validation dataset matching the jet type
            if used_jet_type:
                matching_idx = jet_labels == i
                inpt = all_inpt[matching_idx]
                gen = all_gen[matching_idx]

                # Skip the jet type if it is empty
                if matching_idx.sum() == 0:
                    continue

            # Otherwise just continue with all of the data
            else:
                inpt = all_inpt
                gen = all_gen

            # Plot the marginals
            Path("./plots/").mkdir(parents=False, exist_ok=True)
            img = plot_multi_hists_2(
                data_list=[inpt, gen],
                data_labels=["Original", "Generated"],
                col_labels=inpt_feats,
                do_norm=True,
                return_img=True,
                do_err=True,
                bins=20,
                path=f"./plots/gen_{jet_type}_{self.trainer.current_epoch}",
            )

            # Create the wandb table and add the data
            if wandb.run is not None:
                wandb.run.log({f"gen_{jet_type}": wandb.Image(img)}, commit=False)

            # Plot the correlations of these (very finicky, in try/except)
            try:
                img2 = plot_multi_correlations(
                    data_list=[inpt, gen],
                    data_labels=["Original", "Generated"],
                    col_labels=inpt_feats,
                    n_bins=20,
                    n_kde_points=50,
                    legend_kwargs={
                        "loc": "upper right",
                        "alignment": "right",
                        "fontsize": 15,
                        "bbox_to_anchor": (0.8, 0.90),
                    },
                    hist_kwargs=[{"color": "tab:blue"}, {"color": "tab:orange"}],
                    path=f"./plots/corr_{jet_type}_{self.trainer.current_epoch}",
                    return_img=True,
                )
                if wandb.run is not None:
                    wandb.run.log({f"corr_{jet_type}": wandb.Image(img2)}, commit=False)
            except Exception:
                pass

        # Clear the outputs
        self.val_step_outs.clear()

    def generate(self, n_points: int = 1, ctxt: T.Tensor | None = None) -> T.Tensor:
        """Sample from the flow, undo the scaling and the dequantisation."""
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)
            ctxt = self.ctxt_net(ctxt)
        gen = self.flow.sample(n_points, ctxt)
        gen = self.inpt_normaliser.reverse(gen)
        if ctxt is not None:
            gen = gen.squeeze(1)
        gen = masked_round(gen, self.int_dims)
        return gen

    def predict_step(self, batch: tuple, _batch_idx: int) -> None:
        """Return a sampled prediction of the context inputs."""
        nodes, mask, hlv, pt = batch
        ctxt = hlv[..., self.inpt_dim :] if self.ctxt_dim else None
        n_points = 1 if self.ctxt_dim else len(hlv)
        gen = self.generate(n_points=n_points, ctxt=ctxt)

        # Save the data in the exact form used by our generative models
        outs = {
            "pt": np.zeros(0, dtype=np.float32),
            "eta": np.zeros(0, dtype=np.float32),
            "mass": np.zeros(0, dtype=np.float32),
            "num_particles": np.zeros(0, dtype=np.float32),
            "type": np.zeros(0, dtype=np.float32),
        }
        feats = self.trainer.datamodule.hparams.data_conf.jetnet_config.jet_features
        inpt_feats = feats[: self.inpt_dim]
        ctxt_feats = feats[self.inpt_dim :]
        for k in outs.keys():
            if k == "type":
                outs[k] = ctxt[..., -1:]
            elif k in inpt_feats:
                outs[k] = gen[..., inpt_feats.index(k), None]
            else:
                outs[k] = ctxt[..., ctxt_feats.index(k), None]

        return {"gen_ctxt": gen, **outs}

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        # Use utils to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.sched_config.utils,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.sched_config.lightning},
        }
