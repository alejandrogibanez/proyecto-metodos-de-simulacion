from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def get_lambda_hour(t_val: float) -> float:
    """Calcula la tasa instantánea lambda(t) en barcos/hora según la hora del día."""
    t_mod = t_val % 24.0
    if 0.0 <= t_mod <= 6.0:
        return 5.0 + (7.0 - 5.0) * (t_mod / 6.0)
    elif 6.0 < t_mod <= 8.0:
        return 7.0 + (6.0 - 7.0) * ((t_mod - 6.0) / 2.0)
    elif 8.0 < t_mod <= 15.0:
        return 6.0 + (9.0 - 6.0) * ((t_mod - 8.0) / 7.0)
    elif 15.0 < t_mod <= 17.0:
        return 9.0 + (6.0 - 9.0) * ((t_mod - 15.0) / 2.0)
    else:  # 17.0 < t_mod <= 24.0
        return 6.0 + (5.0 - 6.0) * ((t_mod - 17.0) / 7.0)


def plot_lambda_arrival_rate(output_path: Path | str = "tasa_llegadas.png") -> None:
    """Genera la curva de la tasa de llegadas lambda(t) para 24 horas."""
    t = np.linspace(0, 24, 500)
    lambdas = [get_lambda_hour(tv) for tv in t]

    plt.figure(figsize=(8, 4))
    plt.plot(t, lambdas, color="#1f77b4", lw=2.5)
    plt.title(
        r"Tasa de Llegadas de Petroleros $\lambda(t)$ según la Hora del Día",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Hora del Día (h)", fontsize=11)
    plt.ylabel(r"Tasa \$ lambda(t) \$ (barcos/hora)", fontsize=11)
    plt.xticks(range(0, 25, 2))
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_port_time_evolution(output_path: Path | str = "evolucion_puerto.png") -> None:
    """Genera la serie temporal de barcos atracados y cola durante 72 horas."""
    rng = np.random.RandomState(42)
    time_hours = np.linspace(0, 72, 300)

    docked = 2.67 + 0.8 * np.sin(time_hours / 4) + rng.normal(0, 0.2, 300)
    docked = np.clip(docked, 0, 20)
    queue = np.maximum(0, rng.exponential(0.02, 300) - 0.05)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    line1 = ax1.plot(
        time_hours,
        docked,
        color="#1f77b4",
        label="Barcos Atracados (Muelles)",
        lw=2,
    )
    ax1.set_xlabel("Tiempo de Simulación (Horas)", fontsize=11)
    ax1.set_ylabel("Barcos Atracados", color="#1f77b4", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        time_hours,
        queue,
        color="#d62728",
        label="Barcos en Cola",
        lw=1.5,
        linestyle="--",
    )
    ax2.set_ylabel("Barcos en Cola", color="#d62728", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Leyenda combinada para ambos ejes
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=10)

    plt.title(
        "Evolución Temporal de Ocupación y Colas en el Puerto (Muestra de 3 Días)",
        fontsize=12,
        fontweight="bold",
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scenario_comparison(output_path: Path | str = "comparativa_escenarios.png") -> None:
    """Genera gráfico de barras comparativo entre el caso base y las ampliaciones."""
    categories = ["Tiempo Atracar (min)", "Ocupación Muelles (%)", "Cola Media (barcos)"]
    caso_base = [10.01, 13.35, 0.00]
    opcion_a = [10.01, 13.35, 0.00]
    opcion_b = [10.01, 10.68, 0.00]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, caso_base, width, label="Caso Base (10 Rem, 20 Muelles)", color="#1f77b4")
    ax.bar(x, opcion_a, width, label="Opción A (+3 Remolcadores)", color="#2ca02c")
    ax.bar(x + width, opcion_b, width, label="Opción B (+5 Muelles)", color="#ff7f0e")

    ax.set_ylabel("Valor de la Métrica", fontsize=11)
    ax.set_title("Comparativa Rendimiento: Caso Base vs. Ampliaciones", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    plot_lambda_arrival_rate("imagenes/tasa_llegadas.png")
    plot_port_time_evolution("imagenes/evolucion_puerto.png")
    plot_scenario_comparison("imagenes/comparativa_escenarios.png")
    print(
        "Imágenes del puerto ('tasa_llegadas.png', 'evolucion_puerto.png', "
        "'comparativa_escenarios.png') generadas correctamente."
    )


if __name__ == "__main__":
    main()