import copy
from functools import partial
from typing import Callable, Mapping, Tuple

import pytorch_lightning as pl
import torch as T

from src.models.diffusion import TransformerDiffusionGenerator
from src.utils.diffusion import append_dims, multistep_consistency_sampling
from src.utils.torch_utils import get_loss_fn


class ConsistencyDroid(TransformerDiffusionGenerator):
    """A generative model which uses the consistency distilation to learn the
    ODE given by a parent diffusion model."""

    def __init__(
        self,
        *,
        data_dims: tuple,
        teacher_checkpoint: str,
        sampler_function: Callable,
        sigma_function: Callable,
        optimizer: partial,
        sched_config: Mapping,
        loss_name: partial,
        min_sigma: float = 0.002,
        max_sigma: float = 80.0,
        ema_sync: float = 0.999,
        n_gen_steps: int = 3,
    ) -> None:
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(
            {k: v for k, v in locals().items() if k != "self"}, logger=False
        )

        # Class attributes
        self.pc_dim = data_dims[0]
        self.ctxt_dim = data_dims[1]
        self.n_nodes = data_dims[2]
        self.loss_fn = get_loss_fn(loss_name)
        self.ema_sync = ema_sync
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.n_gen_steps = n_gen_steps

        # Load the previous network from the checkpoint file (trim the fat)
        self.teacher_net = TransformerDiffusionGenerator.load_from_checkpoint(
            teacher_checkpoint,
            strict=False,
        )
        del self.teacher_net.net

        # Get get all other pre-post processing from the teacher network
        self.sigma_encoder = self.teacher_net.sigma_encoder
        self.normaliser = self.teacher_net.normaliser
        if self.ctxt_dim:
            self.ctxt_normaliser = self.teacher_net.ctxt_normaliser

        # Make copies for the networks that will be trained
        self.net = copy.deepcopy(self.teacher_net.ema_net)
        self.ema_net = copy.deepcopy(self.teacher_net.ema_net)

        # Sampler to use for generation with the teacher network
        self.sampler_function = sampler_function
        self.sigma_function = sigma_function
        self.fixed_sigmas = self.sigma_function(self.max_sigma, self.min_sigma)
        self.n_steps = len(self.fixed_sigmas)
        self.val_step_outs = []

        # Make sure the gradients are not tracked for the teacher or target
        self.teacher_net.ema_net.requires_grad_(False)
        self.teacher_net.ema_net.eval()
        self.ema_net.requires_grad_(False)
        self.ema_net.eval()
        self.net.requires_grad_(True)

    def get_c_values(self, sigmas: T.Tensor) -> tuple:
        """Calculate the c values needed for the I/O.

        Note the extra min_sigma term needed for the consistency models
        """

        # We use cos encoding so we dont need c_noise
        c_in = 1 / (1 + sigmas**2).sqrt()
        c_out = (sigmas - self.min_sigma) / (1 + sigmas**2).sqrt()
        c_skip = 1 / (1 + (sigmas - self.min_sigma) ** 2)

        return c_in, c_out, c_skip

    def _shared_step(self, sample: tuple) -> Tuple[T.Tensor, T.Tensor]:
        """Shared step used in both training and validaiton."""

        # Unpack the sample tuple
        nodes, mask, ctxt, pt = sample

        # Pass through the normalisers
        nodes = self.normaliser(nodes, mask)
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)

        # Sample the discrete timesteps for which to learn
        n = T.randint(low=0, high=self.n_steps - 1, size=(nodes.shape[0],))

        # Get the sigma values for these times
        sigma_start = self.fixed_sigmas[n].to(self.device)
        sigma_end = self.fixed_sigmas[n + 1].to(self.device)  # Sigmas are decreasing!

        # Sample from N(0, sigma**2)
        noises = T.randn_like(nodes) * append_dims(sigma_start, nodes.dim())

        # Make the noisy samples by mixing with the real data
        noisy_data = nodes + noises

        # Get the denoised estimate from the online network
        denoised_data = self.forward(noisy_data, sigma_start, mask, ctxt)

        # Do one step of the chosen sampler method to get the next part of the ODE
        with T.no_grad():
            self.teacher_net.eval()
            next_data = self.sampler_function(
                self.teacher_net,
                noisy_data,
                sigma_start,
                sigma_end,
                extra_args={"ctxt": ctxt, "mask": mask, "use_ema": True},
            )

            # Get the denoised estimate for the next data using the ema network
            self.ema_net.eval()
            denoised_next = self.forward(
                next_data, sigma_end, mask, ctxt, use_ema=True
            ).detach()

        # Return the consistency loss (only non masked samples)
        return self.loss_fn(denoised_data, denoised_next, mask, mask).mean()

    @T.no_grad()
    def full_generation(
        self,
        mask: T.BoolTensor | None = None,
        ctxt: T.Tensor | None = None,
        initial_noise: T.Tensor | None = None,
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

        # Get which values are going to be selected for the generation
        sigmas = T.linspace(0, 1 - 1 / self.n_gen_steps, self.n_gen_steps)
        sigmas = sigmas * len(self.fixed_sigmas)
        sigmas = sigmas.round().int()
        sigmas = self.fixed_sigmas[sigmas].to(self.device)

        # Do a consistency step generation
        outputs = multistep_consistency_sampling(
            model=self,
            sigmas=sigmas,
            min_sigma=self.min_sigma,
            x=initial_noise,
            extra_args={"ctxt": ctxt, "mask": mask, "use_ema": True},
        )

        # Ensure that the output adheres to the mask
        outputs[~mask] = 0

        # Return the normalisation of the generated point cloud
        return self.normaliser.reverse(outputs, mask=mask)
