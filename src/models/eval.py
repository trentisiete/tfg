from time import time
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import numpy as np
from sklearn.base import clone
from copy import deepcopy
from multiprocessing import Pool
from .pls import PLSSurrogateRegressor
from .ridge import RidgeSurrogateRegressor



def make_splits(groups: np.ndarray, X: np.ndarray, y:np.ndarray):
    # Create splits based on Leave-One-Diet/Group-Out cross-validation
    lodo = LeaveOneGroupOut()
    lodo.get_n_splits(groups=groups)

    for i, (train_index, test_index) in enumerate(lodo.split(X, y, groups)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield (X_train, X_test, y_train, y_test, i)

def _evaluate_single_model(args):
    """
    Worker function to evaluate a single model across all splits.
    Must be at module level for multiprocessing pickling.
    """
    name, model, X, y, groups = args
    print(f"Evaluating model: {name}")
    base_model = model
    t0 = int(time())
    scores = {}

    for X_train, X_test, y_train, y_test, i in make_splits(groups, X, y):
        m = clone(base_model)
        m.fit(X_train, y_train)
        # Does model have a score method?
        scores[i] = m.score(X_test, y_test)

    print("Finished evaluating model:", name)

    return (name, {"timestamp": time() - t0, "scores": scores})


def evaluate_model(models, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """
    Evaluate multiple models in parallel using Leave-One-Group-Out CV
    """
    tasks = [(name, model, X, y, groups) for name, model in models.items()]

    # Multiprocessing pool
    with Pool() as pool:
        results_list = pool.map(_evaluate_single_model, tasks)
    return dict(results_list)

if __name__ == "__main__":
    groups=np.array([0,0,1,1,2,2])
    X=np.array([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]]).T
    y=np.array([1,2,3,4,5,6])

    models = {
        "PLS": PLSSurrogateRegressor(n_components=2),
        "Ridge": RidgeSurrogateRegressor(alpha=1.0)
    }
    results = evaluate_model(models, X, y, groups)
    print(results)