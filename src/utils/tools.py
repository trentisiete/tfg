import numpy as np
import re

def _to_jsonable(obj):
    """
    Convert common non-JSON-serializable objects (numpy scalars, sets, etc.)
    into plain Python types. For unknown objects, fall back to repr(obj).
    """
    if obj is None:
        return None

    # numpy scalars
    if isinstance(obj, (np.generic,)):
        return obj.item() # Convert to native Python type

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert an array to a list

    # dict
    if isinstance(obj, dict):
        # Recursively convert:
        # keys to str
        # Values to jsonable - new call
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        # Recursively convert each element of array to jsonable - new call
        return [_to_jsonable(v) for v in obj]

    # set
    if isinstance(obj, set):
        return sorted([_to_jsonable(v) for v in obj])

    # plain python numeric / bool / str
    if isinstance(obj, (int, float, bool, str)):
        # Already JSON serializable
        return obj

    # fallback for kernels / objects
    return repr(obj)

def slugify(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return safe or "target"
