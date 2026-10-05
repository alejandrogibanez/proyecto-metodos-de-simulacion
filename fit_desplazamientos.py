from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def main() -> None:
    # 1. Cargar datos
    filename = Path("E5.desplazamientos.txt")
    try:
        data = np.loadtxt(filename)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filename}'.")
        print(f"Asegúrate de colocar '{filename}' en el mismo directorio que este script.")
        return

    print("=" * 55)
    print(" ANÁLISIS ESTADÍSTICO Y AJUSTE DE DISTRIBUCIONES ")
    print("=" * 55)
    print(f"Número de observaciones (N) : {len(data)}")
    print(f"Media empírica              : {np.mean(data):.4f} minutos")
    print(f"Desviación típica empírica  : {np.std(data, ddof=1):.4f} minutos")
    print(
        f"Mínimo: {np.min(data):.4f} | "
        f"Máximo: {np.max(data):.4f} | "
        f"Mediana: {np.median(data):.4f}"
    )
    print("-" * 55, "\n")

    # 2. Ajuste de distribuciones por Máxima Verosimilitud (MLE)
    # A) Normal
    mu_norm, std_norm = stats.norm.fit(data)
    ks_norm = stats.kstest(data, "norm", args=(mu_norm, std_norm))

    # B) Uniforme
    a_unif, b_unif = float(np.min(data)), float(np.max(data))
    ks_unif = stats.kstest(data, "uniform", args=(a_unif, b_unif - a_unif))

    # C) Exponencial
    loc_exp, scale_exp = stats.expon.fit(data)
    lambda_exp = 1.0 / scale_exp
    ks_exp = stats.kstest(data, "expon", args=(loc_exp, scale_exp))

    # 3. Impresión de resultados de los tests K-S
    print("--- 1. CONTRASTES DE KOLMOGOROV-SMIRNOV (K-S) ---")
    print(f"A) NORMAL N(mu={mu_norm:.4f}, sigma={std_norm:.4f}):")
    print(f"   - Estadístico D = {ks_norm.statistic:.4f}")
    print(f"   - p-valor       = {ks_norm.pvalue:.4f}")
    dec_norm = "ACEPTAR H0 (Distribución Normal)" if ks_norm.pvalue > 0.05 else "RECHAZAR H0"
    print(f"   - Decisión (alpha=0.05): {dec_norm}\n")

    print(f"B) UNIFORME U(a={a_unif:.4f}, b={b_unif:.4f}):")
    print(f"   - Estadístico D = {ks_unif.statistic:.4f}")
    print(f"   - p-valor       = {ks_unif.pvalue:.4e}")
    dec_unif = "ACEPTAR H0" if ks_unif.pvalue > 0.05 else "RECHAZAR H0 (No es Uniforme)"
    print(f"   - Decisión (alpha=0.05): {dec_unif}\n")

    print(f"C) EXPONENCIAL Exp(lambda={lambda_exp:.4f}, loc={loc_exp:.4f}):")
    print(f"   - Estadístico D = {ks_exp.statistic:.4f}")
    print(f"   - p-valor       = {ks_exp.pvalue:.4e}")
    dec_exp = "ACEPTAR H0" if ks_exp.pvalue > 0.05 else "RECHAZAR H0 (No es Exponencial)"
    print(f"   - Decisión (alpha=0.05): {dec_exp}\n")

    # 4. Test Chi-Cuadrado para la Distribución Normal
    print("--- 2. CONTRASTE CHI-CUADRADO PARA DISTRIBUCIÓN NORMAL ---")
    num_bins = 15
    observed, bin_edges = np.histogram(data, bins=num_bins)

    # Frecuencias esperadas según la normal ajustada
    cdf_vals = stats.norm.cdf(bin_edges, loc=mu_norm, scale=std_norm)
    expected = len(data) * np.diff(cdf_vals)

    # Filtrar intervalos válidos
    valid = expected > 0
    obs_valid = observed[valid]
    exp_valid = expected[valid]
    chi2_stat = np.sum((obs_valid - exp_valid) ** 2 / exp_valid)

    # Grados de libertad: k - 1 - p (donde p=2 parámetros estimados: mu, std)
    df = len(obs_valid) - 1 - 2
    p_val_chi2 = 1.0 - stats.chi2.cdf(chi2_stat, df)

    print(f"   - Intervalos considerados: {len(obs_valid)}")
    print(f"   - Grados de libertad (df): {df}")
    print(f"   - Estadístico Chi2       = {chi2_stat:.4f}")
    print(f"   - p-valor                = {p_val_chi2:.4f}")
    dec_chi2 = "ACEPTAR H0 (Distribución Normal)" if p_val_chi2 > 0.05 else "RECHAZAR H0"
    print(f"   - Decisión (alpha=0.05)  : {dec_chi2}\n")

    # 5. Generación del gráfico de histograma y curva ajustada
    plt.figure(figsize=(9, 5))
    plt.hist(
        data,
        bins=25,
        density=True,
        alpha=0.6,
        color="#1f77b4",
        edgecolor="black",
        label="Datos Empíricos",
    )

    x = np.linspace(np.min(data) - 2, np.max(data) + 2, 300)
    pdf_norm = stats.norm.pdf(x, loc=mu_norm, scale=std_norm)
    plt.plot(
        x,
        pdf_norm,
        "r-",
        lw=2.5,
        label=rf"Ajuste Normal $\mathcal{{N}}(\mu={mu_norm:.2f}, \sigma={std_norm:.2f})$",
    )

    plt.title("Ajuste de la Distribución de Tiempos de Desplazamiento", fontsize=12, fontweight="bold")
    plt.xlabel("Tiempo de Desplazamiento (minutos)", fontsize=11)
    plt.ylabel("Densidad de Probabilidad", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    output_img = "imagenes/ajuste_desplazamientos.png"
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico del ajuste guardado exitosamente como '{output_img}'.")


if __name__ == "__main__":
    main()