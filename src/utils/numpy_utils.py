import numpy as np


def onehot_encode(
    a: np.ndarray,
    max_idx: None | int = None,
    dtype: np.dtype = np.float32,
    count_unique: bool = False,
) -> np.ndarray:
    if count_unique:
        unique, inverse = np.unique(a, return_inverse=True)
        a = inverse
    max_idx = max_idx or a.max()
    ncols = max_idx + 1
    out = np.zeros((a.size, ncols), dtype=dtype)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def log_squash(data: np.ndarray) -> np.ndarray:
    """Apply a log squashing function for distributions with high tails."""
    return np.sign(data) * np.log(np.abs(data) + 1)


def undo_log_squash(data: np.ndarray) -> np.ndarray:
    """Undo the log squash function above."""
    return np.sign(data) * (np.exp(np.abs(data)) - 1)
