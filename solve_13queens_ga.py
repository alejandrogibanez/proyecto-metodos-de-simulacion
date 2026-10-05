from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def count_collisions(individual: np.ndarray) -> int:
    """Cuenta el número de pares de reinas que se amenazan en diagonal.

    Las colisiones en fila y columna son 0 por construcción al usar permutaciones.
    """
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
    """Cruce por Orden (OX - Order Crossover) para garantizar permutaciones válidas."""
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
    """Mutación Swap: intercambia dos posiciones en la permutación con probabilidad p_mut."""
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
    """Selección por Torneo de tamaño k."""
    pop_size = len(population)
    selected_indices = rng.choice(pop_size, size=k, replace=False)
    best_idx = selected_indices[np.argmin(fitnesses[selected_indices])]
    return population[best_idx]


def solve_13_queens(
    n: int = 13,
    pop_size: int = 200,
    max_gen: int = 1000,
    p_mut: float = 0.3,
    k_tournament: int = 3,
    elitism: int = 2,
    seed: int = 123,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Resuelve el problema de las N-Reinas utilizando un Algoritmo Genético.

    Returns:
        positions: Lista de coordenadas (columna, fila) [(0, r0), ..., (N-1, rN-1)].
        row_vector: Lista con los índices de fila [r0, r1, ..., rN-1].
    """
    rng = np.random.RandomState(seed)
    population = [rng.permutation(n) for _ in range(pop_size)]
    fitnesses = np.array([count_collisions(ind) for ind in population])

    for _ in range(max_gen):
        best_idx = np.argmin(fitnesses)
        best_fit = fitnesses[best_idx]

        if best_fit == 0:
            best_individual = population[best_idx]
            row_vector = [int(r) for r in best_individual]
            positions = [(col, row) for col, row in enumerate(row_vector)]
            return positions, row_vector

        sorted_indices = np.argsort(fitnesses)
        new_population = [population[i].copy() for i in sorted_indices[:elitism]]

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, k_tournament, rng)
            parent2 = tournament_selection(population, fitnesses, k_tournament, rng)
            child = order_crossover_ox(parent1, parent2, rng)
            child = swap_mutation(child, p_mut, rng)
            new_population.append(child)

        population = new_population
        fitnesses = np.array([count_collisions(ind) for ind in population])

    best_idx = np.argmin(fitnesses)
    best_individual = population[best_idx]
    row_vector = [int(r) for r in best_individual]
    positions = [(col, row) for col, row in enumerate(row_vector)]
    return positions, row_vector


def plot_chessboard_png(
    positions: list[tuple[int, int]],
    n: int = 13,
    output_filename: Path | str = "tablero_13reinas.png",
) -> None:
    """Genera y guarda una imagen PNG del tablero de ajedrez con las reinas."""
    # Matriz de escaques alternados (0 = blanco, 0.75 = gris)
    board = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            if (r + c) % 2 == 1:
                board[r, c] = 0.75

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(board, cmap="gray_r", vmin=0, vmax=1)

    # Dibujar cada reina en su escaque correspondiente
    for col, row in positions:
        ax.text(
            col,
            row,
            "♛",
            fontsize=24,
            ha="center",
            va="center",
            color="#d62728",
            fontweight="bold",
        )

    # Configuración de ejes principales
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=10)
    ax.set_yticklabels(range(n), fontsize=10)

    # Rejilla menor para delimitar casillas
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#333333", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    plt.title(
        f"Solución Óptima {n}-Reinas (0 Colisiones)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"¡Imagen del tablero guardada con éxito como '{output_filename}'!")


if __name__ == "__main__":
    # 1. Encontrar posición de las reinas
    positions, row_vector = solve_13_queens(
        n=13,
        pop_size=200,
        max_gen=500,
        p_mut=0.3,
        seed=123,
    )

    # 2. Generar imagen gráfica PNG
    plot_chessboard_png(positions, n=13, output_filename="imagenes/tablero_13reinas.png")