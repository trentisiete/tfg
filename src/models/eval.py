from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import numpy as np

def make_splits(groups: np.ndarray):
    