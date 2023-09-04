from pathlib import Path
from typing import Mapping

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

import h5py
import hydra
import numpy as np
import torch as T
from omegaconf import DictConfig
from tqdm import trange

from src.utils.hydra_utils import reload_original_config
from src.utils.numpy_utils import onehot_encode, undo_log_squash

log = logging.getLogger(__name__)


def dict_to_str(x: Mapping):
    return "_".join(f"{key!s}-{val!r}" for (key, val) in x.items())


JETNET_ID = ["g", "q", "t", "w", "z"]

FLOW_MIN = [159.33, -2.7, 3.35, 5]
FLOW_MAX = [3156.72, 2.7, 573.61, 150]


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="generate.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Setting the deivce based on available hardware")
    device = "cuda" if T.cuda.is_available() else "cpu"

    log.info("Loading the generator checkpoint")
    gen_config = reload_original_config(cfg.generator_path)
    gen = T.load(gen_config.ckpt_path, map_location=device)

    log.info("Loading the flow for unconditional generation")
    flow_config = reload_original_config(cfg.flow_path)
    flow = T.load(flow_config.ckpt_path, map_location=device)
    flow = flow.to(device)

    log.info("Setting up the generation parameters")
    gen.sampler_function = hydra.utils.instantiate(cfg.sampler_function)
    gen.sigma_function = hydra.utils.instantiate(cfg.sigma_function)

    log.info("Creating output folder")
    output_folder = Path(
        cfg.output_folder,
        "_".join(
            [
                gen.sampler_function.func.__name__,
                dict_to_str(gen.sampler_function.keywords),
                gen.sigma_function.func.__name__,
                dict_to_str(gen.sigma_function.keywords),
            ],
        ),
        cfg.jet_type,
    )
    output_folder.mkdir(exist_ok=True, parents=True)

    log.info("Generating flow condition for jet type")
    jet_type = np.full(
        cfg.batch_size,
        JETNET_ID.index(cfg.jet_type),
        dtype=np.int64,
    )
    onehot = onehot_encode(jet_type, max_idx=4)
    flow_ctxt = np.hstack([onehot, jet_type[:, None]])
    flow_ctxt = T.from_numpy(flow_ctxt).float().to(device)
    flow_min = T.tensor([FLOW_MIN], dtype=T.float32, device=device)
    flow_max = T.tensor([FLOW_MAX], dtype=T.float32, device=device)

    log.info("Generating flow outputs")
    flow_outputs = []
    for i in trange(cfg.num_samples // cfg.batch_size + 1):
        flow_output = flow.generate(n_points=1, ctxt=flow_ctxt)
        flow_output = T.clamp(flow_output, flow_min, flow_max)
        flow_outputs.append(flow_output)
    flow_outputs = T.vstack(flow_outputs)

    # Save special values from the flow
    flow_pt = flow_outputs[:, :1]
    max_n = flow_outputs[:, -1].max().int()

    log.info("Generating point cloud data")
    gen_outputs = []
    for i in trange(cfg.num_samples // cfg.batch_size + 1):
        flow_output = flow_outputs[cfg.batch_size * i : cfg.batch_size * (i + 1)]

        # Generate the mask
        num_csts = flow_output[:, -1].int().unsqueeze(-1)
        mask = T.arange(0, max_n, device=device).unsqueeze(0)
        mask = mask < num_csts

        # Bring back the onehot encoding and generates
        gen_ctxt = T.hstack([flow_output, flow_ctxt])
        gen_outputs.append(gen.full_generation(mask, gen_ctxt))
    gen_outputs = T.vstack(gen_outputs)

    # Save the outputs using both pt and pt_frac
    etaphipt_frac = gen_outputs.clone()
    gen_outputs[..., -1] = undo_log_squash(gen_outputs[..., -1])
    etaphipt_frac[..., -1] = gen_outputs[..., -1] / flow_pt

    log.info("Saving data")
    with h5py.File(output_folder / "generated_csts.h5", mode="w") as file:
        file.create_dataset("etaphipt", data=gen_outputs)
        file.create_dataset("etaphipt_frac", data=etaphipt_frac)


if __name__ == "__main__":
    main()
