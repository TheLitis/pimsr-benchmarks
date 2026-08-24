"""Cross-package smoke test for the strict 1D producer/consumer contract."""

import h5py
import numpy as np
from pimsr_forward.dataset import build_dataset, validate_dataset_1d
from pimsr_geogen.generator import GeologyGenerator
from pimsr_geogen.io import save_models
from pimsr_inversion.contracts1d import validate_dataset1d


def test_forward_dataset_is_accepted_by_inversion_contract(tmp_path):
    geology = tmp_path / "geology.h5"
    dataset = tmp_path / "dataset.h5"
    save_models(GeologyGenerator(seed=31).sample_batch(2), geology)

    build_dataset(geology, dataset, seed=32, n_periods=4)
    validate_dataset_1d(dataset)
    with h5py.File(dataset, "r") as file:
        contract = validate_dataset1d(file)
        phase = file["obs_mt_phase"][:]

    assert contract.periods.size == 4
    assert contract.n_observations == 2 * 4 + 16
    assert np.all((phase >= 0.0) & (phase < 180.0))
