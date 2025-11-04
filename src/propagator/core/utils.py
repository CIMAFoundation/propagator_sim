import numpy as np
import numpy.typing as npt


def upcast_to_ndarray(
    value: npt.ArrayLike, shape: tuple[int, ...]
) -> npt.NDArray[np.float32]:
    """Check if the input is a float or a numpy array. If it's a float, create a numpy array of the given shape filled with that float value.
    Args:
        value (float | npt.NDArray[np.float32]): The input value, either a float or a numpy array.
        shape (tuple[int, ...]): The desired shape of the output array if the input is a float.
    Returns:
        npt.NDArray[np.float32]: The resulting numpy array.
    """

    if isinstance(value, np.ndarray):
        return value

    array = np.full(shape, value, dtype=np.float32)
    return array
