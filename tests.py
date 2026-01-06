import numpy as np

if __name__ == "__main__":
    # Toy example
    evaluations = np.array([(15, 0.765), (15, 0.5674), (8, 0.45), (9, 0.67), (13, 0.6), (10,0.5)])

    # Fast checking of best params O(n-1)
    best_params, best_score = min(evaluations, key=lambda x: x[1])
            
    print("Best params:", best_params)
    print("Best score:", best_score)