import h5py
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import numpy as np
from jetnet.datasets import JetNet

from src.utils.plotting import plot_multi_hists_2

jet_type = "t"
jetnet_path = "/srv/beegfs/scratch/groups/rodem/datasets/jetnet/"
generated_path = (
    root / "generated" / "sample_lms_disable-True_get_sigmas_karras_n_steps-100_rho-7"
)

# Load the jetnet dataset
csts, _ = JetNet.getData(
    jet_type="t",
    data_dir=jetnet_path,
    num_particles=150,
    split_fraction=[0.7, 0.15, 0.05],
    split="test",
    particle_features=["etarel", "phirel", "ptrel"],
    jet_features=["num_particles"],
)
mask = np.any(csts != 0, axis=-1)

# Load the generated dataset
full_path = generated_path / jet_type / "generated_csts.h5"
with h5py.File(full_path) as f:
    gen_csts = f["etaphipt_frac"][:]
gen_mask = np.any(gen_csts != 0, axis=-1)

# Create the plots
plot_multi_hists_2(
    data_list=[csts[mask], gen_csts[gen_mask]],
    data_labels=["JetNet", "PCDroid"],
    col_labels=["del_eta", "del_phi", "pt_frac"],
    bins=20,
    logy=True,
    do_norm=True,
    path="here.png",
)
