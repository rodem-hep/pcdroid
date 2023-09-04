import copy
from functools import partial
from typing import Callable, Mapping, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch as T
import wandb
from jetnet.evaluation import fpnd

from src.models.utils import plot_mpgan_marginals
from src.utils.diffusion import append_dims
from src.utils.modules import CosineEncoding, IterativeNormLayer
from src.utils.numpy_utils import undo_log_squash
from src.utils.torch_utils import (
    ema_param_sync,
    get_loss_fn,
    get_sched,
    to_np,
    torch_undo_log_squash,
)


class TransformerDiffusionGenerator(pl.LightningModule):
    """A generative model which uses the diffusion process on a point cloud."""

    def __init__(
        self,
        *,
        data_dims: tuple,
        cosine_config: Mapping,
        normaliser_config: Mapping,
        architecture: partial,
        optimizer: partial,
        sched_config: Mapping,
        loss_name: partial,
        min_sigma: float = 1e-5,
        max_sigma: float = 80.0,
        ema_sync: float = 0.999,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sampler_function: Callable | None = None,
        sigma_function: Callable | None = None,
    ) -> None:
        """
        Parameters
        ----------
        data_dims : tuple
            A tuple with three integers representing the point cloud dimensions,
            the context dimensions, and the number of nodes, respectively.
        cosine_config : Mapping
            A dictionary with the configuration options for the CosineEncoding object.
        normaliser_config : Mapping
            A dictionary with the configuration options for the IterativeNormLayer object.
        architecture : partial
            A function to initialise the seq-to-seq neural network
        optimizer : partial
            A function to optimize the parameters of the model.
        sched_config : Mapping
            A dictionary with the configuration options for the optimizer scheduler.
        loss_name : str, optional
            The name of the loss function to use. Default is 'mse'.
        min_sigma : float, optional
            The minimum value for the diffusion sigma during generation.
            Default is 1e-5.
        max_sigma : float, optional
            The maximum value for the diffusion sigma. Default is 80.0.
        ema_sync : float, optional
            The exponential moving average synchronization factor. Default is 0.999.
        p_mean : float, optional
            The mean of the log normal distribution used to sample sigmas when training.
            Default is -1.2.
        p_std : float, optional
            The standard deviation of the log normal distribution used to sample the
            sigmas during training. Default is 1.2.
        sampler_function : Callable | None, optional
            A function to sample the points on the point cloud during the
            validation/testing loop. Default is None.
        sigma_function : Callable | None, optional
            A function to compute the diffusion coefficient sigmas for the diffusion
            process. Default is None.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Class attributes
        self.pc_dim = data_dims[0]
        self.ctxt_dim = data_dims[1]
        self.n_nodes = data_dims[2]
        self.loss_fn = get_loss_fn(loss_name)
        self.ema_sync = ema_sync
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.p_mean = p_mean
        self.p_std = p_std

        # The encoder and scheduler needed for diffusion
        self.sigma_encoder = CosineEncoding(
            min_value=0, max_value=max_sigma, **cosine_config
        )

        # The layer which normalises the input point cloud data
        self.normaliser = IterativeNormLayer(self.pc_dim, **normaliser_config)
        if self.ctxt_dim:
            self.ctxt_normaliser = IterativeNormLayer(
                self.ctxt_dim, **normaliser_config
            )

        # The denoising neural network (transformer / perceiver / CAE / Epic)
        self.net = architecture(
            inpt_dim=self.pc_dim,
            outp_dim=self.pc_dim,
            ctxt_dim=self.ctxt_dim + self.sigma_encoder.outp_dim,
        )

        # A copy of the network which will sync with an exponential moving average
        self.ema_net = copy.deepcopy(self.net)
        self.ema_net.requires_grad_(False)

        # Sampler to run in the validation/testing loop
        self.val_step_outs = []
        self.sampler_function = sampler_function
        self.sigma_function = sigma_function

    def get_c_values(self, sigmas: T.Tensor) -> tuple:
        """Calculate the Karras C values needed to modify the inputs, outputs,
        and skip connection for the neural network."""

        # we use cos encoding so we dont need c_noise
        c_in = 1 / (1 + sigmas**2).sqrt()
        c_out = sigmas / (1 + sigmas**2).sqrt()
        c_skip = 1 / (1 + sigmas**2)

        return c_in, c_out, c_skip

    def forward(
        self,
        noisy_data: T.Tensor,
        sigmas: T.Tensor,
        mask: T.BoolTensor,
        ctxt: Optional[T.Tensor] = None,
        use_ema: bool = False,
    ) -> T.Tensor:
        """Return the denoised data given the current sigma values."""

        # Get the c values for the data scaling
        c_in, c_out, c_skip = self.get_c_values(append_dims(sigmas, noisy_data.dim()))

        # Scale the inputs and pass through the network
        outputs = self.get_outputs(c_in * noisy_data, sigmas, mask, ctxt, use_ema)

        # Get the denoised output by passing the scaled input through the network
        return c_skip * noisy_data + c_out * outputs

    def get_outputs(
        self,
        noisy_data: T.Tensor,
        sigmas: T.Tensor,
        mask: T.BoolTensor,
        ctxt: Optional[T.Tensor] = None,
        use_ema: bool = False,
    ) -> T.Tensor:
        """Pass through the model, corresponds to F_theta in the Karras
        paper."""

        # Use the appropriate network for training or validation
        if self.training and not use_ema:
            network = self.net
        else:
            network = self.ema_net

        # Encode the sigmas and combine with existing context info
        context = self.sigma_encoder(sigmas)
        if self.ctxt_dim:
            context = T.cat([context, ctxt], dim=-1)

        # Use the selected network to esitmate the noise present in the data
        return network(noisy_data, mask=mask, ctxt=context)

    def _shared_step(self, sample: tuple) -> Tuple[T.Tensor, T.Tensor]:
        """Shared step used in both training and validaiton."""

        # Unpack the sample tuple
        nodes, mask, ctxt, pt = sample

        # Pass through the normalisers
        nodes = self.normaliser(nodes, mask)
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)

        # Sample sigmas using the Karras method of a log normal distribution
        sigmas = T.zeros(size=(nodes.shape[0], 1), device=self.device)
        sigmas.add_(self.p_mean + self.p_std * T.randn_like(sigmas))
        sigmas.exp_().clamp_(self.min_sigma, self.max_sigma)

        # Get the c values for the data scaling
        c_in, c_out, c_skip = self.get_c_values(append_dims(sigmas, nodes.dim()))

        # Sample from N(0, sigma**2)
        noises = T.randn_like(nodes) * append_dims(sigmas, nodes.dim())

        # Make the noisy samples by mixing with the real data
        noisy_nodes = nodes + noises

        # Pass through the just the base network (manually scale with c values)
        output = self.get_outputs(c_in * noisy_nodes, sigmas, mask, ctxt)

        # Calculate the effective training target
        target = (nodes - c_skip * noisy_nodes) / c_out

        # Return the denoising loss (only non masked samples)
        return self.loss_fn(output, target, mask, mask).mean()

    def training_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        loss = self._shared_step(sample)
        self.log("train/total_loss", loss)
        ema_param_sync(self.net, self.ema_net, self.ema_sync)
        return loss

    def validation_step(self, sample: tuple, batch_idx: int) -> None:
        loss = self._shared_step(sample)
        self.log("valid/total_loss", loss)

        # Run the full generation for 20% of the validation set (takes long)
        if batch_idx % 5 == 0:
            generated = self.full_generation(mask=sample[1], ctxt=sample[2])
            self.val_step_outs.append((to_np(generated), to_np(sample)))

    def on_validation_epoch_end(
        self,
    ) -> None:
        """Plot histograms of fully generated samples from the validation
        epoch."""

        # Only do this step on the main gpu
        if self.trainer.is_global_zero:

            # Combine all outputs
            all_gen_nodes = np.vstack([v[0] for v in self.val_step_outs])
            all_real_nodes = np.vstack([v[1][0] for v in self.val_step_outs])
            all_mask = np.vstack([v[1][1] for v in self.val_step_outs])
            all_high = np.vstack([v[1][2] for v in self.val_step_outs])
            all_pt = np.vstack([v[1][3] for v in self.val_step_outs])

            # Get all of the jet labels WHICH SHOULD BE LAST!
            jet_labels = all_high[:, -1].astype("long")

            # Cycle through each of the labels
            jet_types = ["g", "q", "t", "w", "z"]  # Fixed order based on jetnet
            for i, jet_type in enumerate(jet_types):

                # Pull out the events in the validation dataset matching the jet type
                matching_idx = jet_labels == i
                gen_nodes = all_gen_nodes[matching_idx]
                real_nodes = all_real_nodes[matching_idx]
                mask = all_mask[matching_idx]
                pt = all_pt[matching_idx]

                # Skip the jet type if it is empty (sometimes we only generate t)
                if matching_idx.sum() == 0:
                    continue

                # Change the data from log(pt+1) back to pt fraction for the metrics
                if self.trainer.datamodule.hparams.data_conf.log_squash_pt:
                    gen_nodes[..., -1] = undo_log_squash(gen_nodes[..., -1]) / pt
                    real_nodes[..., -1] = undo_log_squash(real_nodes[..., -1]) / pt

                # Apply clipping to prevent the values from causing issues in metrics
                gen_nodes = np.nan_to_num(gen_nodes)
                gen_nodes[..., 0] = np.clip(gen_nodes[..., 0], -0.5, 0.5)
                gen_nodes[..., 1] = np.clip(gen_nodes[..., 1], -0.5, 0.5)
                gen_nodes[..., 2] = np.clip(gen_nodes[..., 2], 0, 1)
                real_nodes = np.nan_to_num(real_nodes)
                real_nodes[..., 0] = np.clip(real_nodes[..., 0], -0.5, 0.5)
                real_nodes[..., 1] = np.clip(real_nodes[..., 1], -0.5, 0.5)
                real_nodes[..., 2] = np.clip(real_nodes[..., 2], 0, 1)

                # Calculate the FPND metric which is only valid for some jets with 30 csts
                if jet_type in ["g", "t", "q"]:
                    if gen_nodes.shape[-2] > 30:
                        sort_idx = np.argsort(gen_nodes[..., 2], axis=-1)[..., None]
                        top_30 = np.take_along_axis(gen_nodes, sort_idx, axis=1)
                        top_30 = top_30[:, -30:]
                        fpnd_val = fpnd(top_30, jet_type=jet_type)
                    else:
                        fpnd_val = fpnd(gen_nodes, jet_type=jet_type)
                    self.log(f"valid/{jet_type}_fpnd", fpnd_val)

                # Plot the MPGAN-like marginals
                plot_mpgan_marginals(
                    gen_nodes, real_nodes, mask, self.trainer.current_epoch, jet_type
                )

        # Clear the outputs
        self.trainer.strategy.barrier()  # let other cards to wait for the main
        self.val_step_outs.clear()

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")

    @T.no_grad()
    def full_generation(
        self,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        initial_noise: Optional[T.Tensor] = None,
    ) -> T.Tensor:
        """Fully generate a batch of data from noise, given context information
        and a mask."""

        # Either a mask or initial noise must be defined or we dont know how
        # many samples to generate and with what cardinality
        if mask is None and initial_noise is None:
            raise ValueError("Please provide either a mask or noise to generate from")
        if mask is None:
            mask = T.full(initial_noise.shape[:-1], True, device=self.device)
        if initial_noise is None:
            initial_noise = (
                T.randn((*mask.shape, self.pc_dim), device=self.device) * self.max_sigma
            )

        # Normalise the context
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)
            assert len(ctxt) == len(initial_noise)

        # Generate the sigma values
        sigmas = self.sigma_function(self.max_sigma, self.min_sigma).to(self.device)

        # Run the sampler
        outputs = self.sampler_function(
            model=self,
            x=initial_noise,
            sigmas=sigmas,
            extra_args={"ctxt": ctxt, "mask": mask},
        )

        # My samplers return 2 vars, k-diffusion returns 1
        if isinstance(outputs, tuple):
            outputs, _ = outputs

        # Ensure that the output adheres to the mask
        outputs[~mask] = 0

        # Return the normalisation of the generated point cloud
        return self.normaliser.reverse(outputs, mask=mask)

    def on_predict_start(self, *_args):
        print(f"Generating with {self.sampler_function.func.__name__}")

    def predict_step(self, sample: tuple, _batch_idx: int) -> None:
        """Single test step which fully generates a batch of jets using the
        context and mask found in the test set.

        All generated jets must be of the format: (eta, phi, pt), (eta,
        phi, pt_frac)
        """

        # Unpack the sample tuple (dont need constituents)
        _, mask, ctxt, pt = sample

        # Generate the data and move to numpy
        etaphipt = self.full_generation(mask, ctxt)
        etaphipt_frac = etaphipt.clone()

        # Calculate the nodes with pt
        use_log_squash_pt = getattr(
            self.trainer.datamodule.test_set, "log_squash_pt", True
        )
        if use_log_squash_pt:
            etaphipt[..., -1] = torch_undo_log_squash(etaphipt[..., -1])
            etaphipt_frac[..., -1] = etaphipt[..., -1] / pt
        else:
            etaphipt[..., -1] = etaphipt[..., -1] * pt

        # Return in a dict for standard combining later
        return {"etaphipt": to_np(etaphipt), "etaphipt_frac": to_np(etaphipt_frac)}

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.net.parameters())

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
