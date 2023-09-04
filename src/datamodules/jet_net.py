from copy import deepcopy
from pathlib import Path
from typing import Mapping

import numpy as np
from jetnet.datasets import JetNet
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils.numpy_utils import log_squash, onehot_encode
from src.utils.plotting import plot_multi_hists_2


class JetNetData(Dataset):
    """Wrapper for the JetNet dataset so it works with our models with
    different inputs."""

    def __init__(
        self,
        jetnet_config: Mapping,
        log_squash_pt: bool = True,
        high_as_context: bool = True,
        n_jets: int | None = None,
        one_hot_last: bool | str = True,
        split: str = "train",
    ) -> None:

        # Save the class attributes
        self.log_squash_pt = log_squash_pt
        self.high_as_context = high_as_context
        self.n_jets = n_jets
        self.one_hot_last = one_hot_last
        self.split = split
        self.kwargs_copy = deepcopy(jetnet_config)

        # Use the auto one_hot is the jet type is last
        if self.one_hot_last == "auto":
            self.one_hot_last = jetnet_config["jet_features"][-1] == "type"

        # Use the built in function to return the kwargs from jetnet
        self.csts, self.high = JetNet.getData(**jetnet_config, split=split)

        # Also load the jet as we need it for pre and post processing
        pt_kwargs = deepcopy(self.kwargs_copy)
        pt_kwargs["jet_features"] = ["pt"]
        pt_kwargs["num_particles"] = 1
        _, self.jet_pt = JetNet.getData(**pt_kwargs, split=split)

        # Trim the data based on the requested number of jets (None does nothing)
        self.csts = self.csts[: self.n_jets].astype(np.float32)
        self.high = self.high[: self.n_jets].astype(np.float32)
        self.jet_pt = self.jet_pt[: self.n_jets].astype(np.float32)

        # One hot encode the final input
        if self.one_hot_last:
            jet_type = self.high[:, -1].astype("int")
            jet_onehot = onehot_encode(jet_type, max_idx=4)  # 4 to match jetnet types
            jet_type = jet_type[:, None].astype(self.high.dtype)
            self.high = np.hstack([self.high[:, :-1], jet_onehot, jet_type])

        # Manually calculate the mask by looking for zero padding
        self.mask = ~np.all(self.csts == 0, axis=-1)

        # Change the pt-fraction to log(pt+1)
        if self.log_squash_pt:
            csts = self.csts.copy()
            csts[..., -1] = csts[..., -1] * self.jet_pt
            self.csts[..., -1] = log_squash(csts[..., -1]) * self.mask

    def __getitem__(self, idx) -> tuple:
        csts = self.csts[idx]
        mask = self.mask[idx]
        high = self.high[idx] if self.high_as_context else np.empty(0, dtype="f")
        pt = self.jet_pt[idx]
        return csts, mask, high, pt

    def __len__(self) -> int:
        return len(self.high)

    def plot(self) -> None:
        plot_path = Path("train_dist")
        plot_path.mkdir(parents=True, exist_ok=True)

        plot_multi_hists_2(
            data_list=self.high,
            data_labels="high level features",
            col_labels=(
                self.kwargs_copy["jet_features"][:-1]
                + ["g", "q", "t", "w", "z"]
                + self.kwargs_copy["jet_features"][-1:]
            )
            if self.one_hot_last
            else self.kwargs_copy["jet_features"],
            do_norm=True,
            path=plot_path / "high",
        )

        plot_multi_hists_2(
            data_list=self.csts[self.mask],
            data_labels="constituents",
            col_labels=[
                r"$\Delta \eta$",
                r"$\Delta \phi$",
                r"log $(p_T+1)$" if self.log_squash_pt else r"$\frac{p_T}{Jet_{p_T}}$",
            ],
            do_norm=True,
            path=plot_path / "csts",
        )


class JetNetDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        data_conf: Mapping,
        loader_kwargs: Mapping,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Are we doing one hot encoding ???
        self.one_hot_last = data_conf.one_hot_last
        if self.one_hot_last == "auto":
            self.one_hot_last = data_conf.jetnet_config["jet_features"][-1] == "type"

        # Get the dimensions of the data from the config file
        self.dim = len(data_conf.jetnet_config.particle_features)
        self.n_nodes = data_conf.jetnet_config.num_particles
        if data_conf.high_as_context:
            self.ctxt_dim = (
                len(data_conf.jetnet_config.jet_features) + 5 * self.one_hot_last
            )
        else:
            self.ctxt_dim = 0

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""

        if stage in ["fit", "validate"]:
            self.train_set = JetNetData(**self.hparams.data_conf, split="train")
            self.valid_set = JetNetData(**self.hparams.data_conf, split="val")
            self.train_set.plot()

        if stage in ["test", "predict"]:
            self.test_set = JetNetData(**self.hparams.data_conf, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_kwargs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, **self.hparams.loader_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.hparams.loader_kwargs)
        test_kwargs["drop_last"] = False
        return DataLoader(self.test_set, **test_kwargs, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dims(self) -> tuple:
        return self.dim, self.ctxt_dim, self.n_nodes

    def jet_types(self) -> list:
        return self.hparams.data_conf["jetnet_config"]["jet_type"]
