# Test LODO evaluation and nested tuning on benchmarks
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.benchmark_runner import evaluate_model_with_lodo, nested_lodo_tuning_benchmark
from src.benchmarks import generate_benchmark_dataset
from src.models.gp import GPSurrogateRegressor
import numpy as np

np.random.seed(42)

# Generar benchmark con GRUPOS (simula lotes/dietas)
dataset = generate_benchmark_dataset(
    benchmark='branin',
    n_train=80,
    n_groups=5,  # <-- Clave: asigna grupos sintéticos
    noise='gaussian',
    noise_kwargs={'sigma': 0.1},
    seed=42
)

print(f'Dataset: {dataset}')
print(f'Grupos únicos: {np.unique(dataset.groups_train)}')
print()

# OPCIÓN 1: Evaluación LODO simple (sin tuning)
print('=' * 50)
print('OPCIÓN 1: evaluate_model_with_lodo')
print('=' * 50)
gp = GPSurrogateRegressor(alpha=0.1)
results_lodo = evaluate_model_with_lodo(gp, dataset)

print(f'Folds: {len(results_lodo["folds"])}')
print(f'Summary keys: {list(results_lodo["summary"].keys())}')
print(f'MAE macro: {results_lodo["summary"]["macro"]["mae"]}')
print()

# OPCIÓN 2: Nested LODO Tuning (como datos reales)
print('=' * 50)
print('OPCIÓN 2: nested_lodo_tuning_benchmark')
print('=' * 50)
results_tuning = nested_lodo_tuning_benchmark(
    base_model=GPSurrogateRegressor(),
    param_grid={'alpha': [0.01, 0.1, 1.0]},
    dataset=dataset,
    scoring='mae',
    n_jobs=2
)

print(f'Folds: {len(results_tuning["folds"])}')
print(f'Chosen params: {results_tuning["chosen_params"]}')
print(f'MAE macro: {results_tuning["summary"]["macro"]["mae"]}')
print()
print('✅ Ambas opciones funcionan!')
