# Métodos de Simulación — Enunciado 5

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LaTeX](https://img.shields.io/badge/LaTeX-Memoria_Técnica-1f4e79.svg)](pdf/memoria_simulacion.tex)

Solución integral al **Enunciado 5** de la asignatura **Métodos de Simulación** del *Máster Universitario en Inteligencia Artificial (MUIA)*. 

El repositorio cubre tres bloques analíticos y computacionales:
1. **Ajuste y Bondad de Ajuste:** Caracterización estadística de tiempos empíricos mediante estimación por máxima verosimilitud (MLE) y contrastes Kolmogorov-Smirnov y $\chi^2$.
2. **Simulación de Eventos Discretos:** Modelado estocástico de la operativa de un puerto comercial bajo un proceso de llegadas Poisson No Homogéneo (NHPP) con algoritmo de *thinning* (Lewis-Shedler).
3. **Optimización Combinatoria:** Resolución del problema de las 13 Reinas mediante un Algoritmo Genético permutacional libre de conflictos diagonales.

---

## 📁 Estructura del Repositorio

```text
.
├── E5.desplazamientos.txt      # Muestra empírica de tiempos de desplazamiento (N=511)
├── fit_desplazamientos.py      # MLE y contrastes de hipótesis (Normal, Uniforme, Exp)
├── simulate_port.py            # Simulación DES del puerto comercial y análisis de escenarios
├── generate_port_plots.py      # Generador de figuras de la dinámica y métricas portuarias
├── solve_13_queens.py          # AG permutacional (13 Reinas) y renderizado del tablero
├── generate_ga_plots.py        # Curvas de convergencia y evolución del fitness del AG
│
├── imagenes/                   # Salidas gráficas empleadas en la memoria técnica
│   ├── ajuste_desplazamientos.png
│   ├── tasa_llegadas.png
│   ├── evolucion_puerto.png
│   ├── comparativa_escenarios.png
│   ├── convergencia_ag_13reinas.png
│   └── tablero_13reinas.png
│
├── pdf/                        # Fuente y compilación de la memoria en LaTeX
│   └── memoria_simulacion.tex
│
└── README.md