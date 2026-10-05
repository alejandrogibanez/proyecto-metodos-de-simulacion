from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def count_collisions(individual: np.ndarray) -> int:
    """Cuenta el número de amenazas diagonales entre las reinas."""
    diag1 = Counter()
    diag2 = Counter()

    for col, row in enumerate(individual):
        diag1[col - row] += 1
        diag2[col + row] += 1

    collisions = 0
    for count in diag1.values():
        if count > 1:
            collisions += count * (count - 1) // 2

    for count in diag2.values():
        if count > 1:
            collisions += count * (count - 1) // 2

    return collisions


def order_crossover_ox(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Cruce por Orden (OX) que preserva permutaciones válidas."""
    n = len(parent1)
    a, b = sorted(rng.choice(n, size=2, replace=False))

    child = np.full(n, -1, dtype=int)
    child[a : b + 1] = parent1[a : b + 1]

    p1_sub_set = set(child[a : b + 1])
    p2_remaining = [x for x in parent2 if x not in p1_sub_set]

    p2_idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = p2_remaining[p2_idx]
            p2_idx += 1

    return child


def swap_mutation(
    individual: np.ndarray,
    p_mut: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Mutación Swap: intercambia dos posiciones con probabilidad p_mut."""
    child = individual.copy()
    if rng.rand() < p_mut:
        n = len(child)
        idx1, idx2 = rng.choice(n, size=2, replace=False)
        child[idx1], child[idx2] = child[idx2], child[idx1]
    return child


def tournament_selection(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Selección por torneo de tamaño k."""
    selected_indices = rng.choice(len(population), size=k, replace=False)
    best_idx = selected_indices[np.argmin(fitnesses[selected_indices])]
    return population[best_idx]


def run_ga_and_plot(
    n: int = 13,
    pop_size: int = 200,
    max_gen: int = 500,
    p_mut: float = 0.3,
    k_tournament: int = 3,
    elitism: int = 2,
    seed: int = 123,
    output_filename: Path | str = "imagenes/convergencia_ag_13reinas.png",
) -> None:
    """Ejecuta el Algoritmo Genético y genera la curva de convergencia."""
    rng = np.random.RandomState(seed)
    population = [rng.permutation(n) for _ in range(pop_size)]
    fitnesses = np.array([count_collisions(ind) for ind in population])

    best_history: list[int] = []
    mean_history: list[float] = []

    for gen in range(max_gen):
        best_fit = int(np.min(fitnesses))
        mean_fit = float(np.mean(fitnesses))

        best_history.append(best_fit)
        mean_history.append(mean_fit)

        if best_fit == 0:
            print(f"¡Solución óptima alcanzada en la generación {gen}!")
            break

        sorted_indices = np.argsort(fitnesses)
        new_population = [population[i].copy() for i in sorted_indices[:elitism]]

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses, k_tournament, rng)
            p2 = tournament_selection(population, fitnesses, k_tournament, rng)
            child = order_crossover_ox(p1, p2, rng)
            child = swap_mutation(child, p_mut, rng)
            new_population.append(child)

        population = new_population
        fitnesses = np.array([count_collisions(ind) for ind in population])

    # Generación y guardado del gráfico de convergencia
    plt.figure(figsize=(8, 4.5))
    generations = range(len(best_history))

    plt.plot(
        generations,
        best_history,
        color="#d62728",
        lw=2.5,
        label="Mínimas Colisiones (Mejor Individuo)",
    )
    plt.plot(
        generations,
        mean_history,
        color="#1f77b4",
        linestyle="--",
        lw=1.8,
        label="Colisiones Medias de la Población",
    )

    plt.title(
        f"Curva de Convergencia del Algoritmo Genético ({n}-Reinas)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("Generación", fontsize=11)
    plt.ylabel("Número de Colisiones (Amenazas)", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Imagen guardada exitosamente como '{output_filename}'.")


if __name__ == "__main__":
    run_ga_and_plot()