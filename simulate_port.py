import heapq
import numpy as np
import scipy.stats as stats


def get_lambda_hour(t_hour: float) -> float:
    """Calcula la tasa instantánea de llegadas lambda(t) en barcos/hora

    para t en horas del día [0, 24).
    """
    t = t_hour % 24.0
    if 0.0 <= t <= 6.0:
        return 5.0 + (7.0 - 5.0) * (t / 6.0)
    elif 6.0 < t <= 8.0:
        return 7.0 + (6.0 - 7.0) * ((t - 6.0) / 2.0)
    elif 8.0 < t <= 15.0:
        return 6.0 + (9.0 - 6.0) * ((t - 8.0) / 7.0)
    elif 15.0 < t <= 17.0:
        return 9.0 + (6.0 - 9.0) * ((t - 15.0) / 2.0)
    else:  # 17.0 < t <= 24.0
        return 6.0 + (5.0 - 6.0) * ((t - 17.0) / 7.0)


def generate_nhpp_arrivals(sim_duration_min: float, seed: int | None = None) -> list[float]:
    """Genera los tiempos de llegada de los barcos (en minutos) usando el

    algoritmo de adelgazamiento (Thinning de Lewis-Shedler) para un
    Proceso de Poisson No Homogéneo (NHPP).
    """
    rng = np.random.RandomState(seed)
    arrivals = []
    t = 0.0
    lambda_max = 9.0 / 60.0  # Tasa máxima (9 barcos/h) convertida a barcos/minuto

    while t < sim_duration_min:
        dt = rng.exponential(1.0 / lambda_max)
        t += dt
        if t >= sim_duration_min:
            break

        # Aceptar llegada con probabilidad lambda(t) / lambda_max
        lambda_t = get_lambda_hour(t / 60.0) / 60.0
        if rng.rand() <= (lambda_t / lambda_max):
            arrivals.append(t)

    return arrivals


class PortSimulation:
    """Simulador de Sucesos Discretos de la Operativa Portuaria."""

    def __init__(
        self,
        num_tugs: int = 10,
        num_berths: int = 20,
        sim_duration_min: float = 30 * 24 * 60,
        seed: int = 42,
    ):
        self.num_tugs = num_tugs
        self.num_berths = num_berths
        self.sim_duration = sim_duration_min
        self.rng = np.random.RandomState(seed)

    def sample_tow_time(self) -> float:
        # N(mu=9.9951, std=3.0743) de ajuste empírico (minutos)
        val = self.rng.normal(9.9951, 3.0743)
        return max(0.5, val)

    def sample_empty_travel_time(self) -> float:
        # Desplazamiento en vacío del remolcador N(mu=2, std=1) (minutos)
        val = self.rng.normal(2.0, 1.0)
        return max(0.1, val)

    def sample_unload_time(self) -> float:
        # Tiempo de descarga Chi-cuadrado df=5 (media = 5 minutos)
        return float(self.rng.chisquare(df=5))

    def run(self) -> dict[str, float]:
        arrivals = generate_nhpp_arrivals(
            self.sim_duration,
            seed=self.rng.randint(1_000_000),
        )

        # Cola de eventos ordenada por tiempo: (tiempo, ev_counter, ev_type, payload)
        event_queue = []
        for idx, arr in enumerate(arrivals):
            heapq.heappush(event_queue, (arr, idx, "ARRIVAL", idx))

        free_berths = self.num_berths
        tugs_at_entrance = self.num_tugs
        tugs_at_berth = 0

        queue_incoming = []  # Barcos esperando para entrar
        queue_outgoing = []  # Barcos esperando para salir

        ship_arrival_time = {}
        ship_docked_time = {}

        # Métricas integradas en el tiempo
        area_docked = 0.0
        area_queue = 0.0
        max_queue = 0
        last_t = 0.0
        ev_counter = len(arrivals)

        def dispatch(curr_t: float):
            nonlocal tugs_at_entrance, tugs_at_berth, free_berths, ev_counter

            while (tugs_at_entrance + tugs_at_berth) > 0:
                # Prioridad 1: Barcos esperando para atracar (si hay muelle disponible)
                if queue_incoming and free_berths > 0:
                    ship_id, arr_t = queue_incoming.pop(0)
                    free_berths -= 1

                    if tugs_at_entrance > 0:
                        tugs_at_entrance -= 1
                        t_empty = 0.0
                    else:
                        tugs_at_berth -= 1
                        t_empty = self.sample_empty_travel_time()

                    t_tow = self.sample_tow_time()
                    ev_counter += 1

                    if t_empty > 0:
                        heapq.heappush(
                            event_queue,
                            (curr_t + t_empty, ev_counter, "TUG_ARRIVED_ENTRANCE", (ship_id, t_tow)),
                        )
                    else:
                        heapq.heappush(
                            event_queue,
                            (curr_t + t_tow, ev_counter, "DOCK", (ship_id, arr_t)),
                        )

                # Prioridad 2: Barcos esperando para desatracar y salir
                elif queue_outgoing:
                    ship_id = queue_outgoing.pop(0)

                    if tugs_at_berth > 0:
                        tugs_at_berth -= 1
                        t_empty = 0.0
                    else:
                        tugs_at_entrance -= 1
                        t_empty = self.sample_empty_travel_time()

                    t_tow = self.sample_tow_time()
                    ev_counter += 1

                    if t_empty > 0:
                        heapq.heappush(
                            event_queue,
                            (curr_t + t_empty, ev_counter, "TUG_ARRIVED_BERTH", (ship_id, t_tow)),
                        )
                    else:
                        heapq.heappush(
                            event_queue,
                            (curr_t + t_tow, ev_counter, "LEFT", ship_id),
                        )
                else:
                    break

        while event_queue:
            t, _, ev_type, data = heapq.heappop(event_queue)
            if t > self.sim_duration:
                break

            # Actualización de acumuladores
            dt = t - last_t
            area_docked += (self.num_berths - free_berths) * dt
            area_queue += len(queue_incoming) * dt
            if len(queue_incoming) > max_queue:
                max_queue = len(queue_incoming)
            last_t = t

            if ev_type == "ARRIVAL":
                ship_id = data
                ship_arrival_time[ship_id] = t
                queue_incoming.append((ship_id, t))
                dispatch(t)

            elif ev_type == "TUG_ARRIVED_ENTRANCE":
                ship_id, t_tow = data
                ev_counter += 1
                heapq.heappush(
                    event_queue,
                    (t + t_tow, ev_counter, "DOCK", (ship_id, ship_arrival_time[ship_id])),
                )

            elif ev_type == "DOCK":
                ship_id, _ = data
                ship_docked_time[ship_id] = t
                tugs_at_berth += 1
                ev_counter += 1
                t_unload = self.sample_unload_time()
                heapq.heappush(event_queue, (t + t_unload, ev_counter, "UNLOAD_END", ship_id))
                dispatch(t)

            elif ev_type == "UNLOAD_END":
                ship_id = data
                queue_outgoing.append(ship_id)
                dispatch(t)

            elif ev_type == "TUG_ARRIVED_BERTH":
                ship_id, t_tow = data
                ev_counter += 1
                heapq.heappush(event_queue, (t + t_tow, ev_counter, "LEFT", ship_id))

            elif ev_type == "LEFT":
                ship_id = data
                free_berths += 1
                tugs_at_entrance += 1
                dispatch(t)

        time_to_dock = [ship_docked_time[s] - ship_arrival_time[s] for s in ship_docked_time]

        return {
            "mean_time_to_berth": float(np.mean(time_to_dock)) if time_to_dock else 0.0,
            "max_time_to_berth": float(np.max(time_to_dock)) if time_to_dock else 0.0,
            "mean_docked_ships": area_docked / self.sim_duration,
            "mean_queue_waiting": area_queue / self.sim_duration,
            "max_queue_waiting": max_queue,
        }


def run_experiment(name: str, num_tugs: int, num_berths: int, num_replications: int = 30) -> None:
    print(f"\n{'=' * 55}")
    print(f" ESCENARIO: {name} ({num_tugs} Remolcadores, {num_berths} Muelles)")
    print(f"{'=' * 55}")

    mean_dock_times, max_dock_times = [], []
    mean_docked_ships, mean_queues, max_queues = [], [], []

    for rep in range(num_replications):
        sim = PortSimulation(num_tugs=num_tugs, num_berths=num_berths, seed=1000 + rep)
        res = sim.run()

        mean_dock_times.append(res["mean_time_to_berth"])
        max_dock_times.append(res["max_time_to_berth"])
        mean_docked_ships.append(res["mean_docked_ships"])
        mean_queues.append(res["mean_queue_waiting"])
        max_queues.append(res["max_queue_waiting"])

    print(f" - Tiempo medio en atracar : {np.mean(mean_dock_times):.2f} +/- {stats.sem(mean_dock_times):.2f} min")
    print(f" - Tiempo máximo en atracar: {np.mean(max_dock_times):.2f} min (Máx absoluto: {np.max(max_dock_times):.2f} min)")
    print(f" - Nº medio barcos atracados: {np.mean(mean_docked_ships):.2f} +/- {stats.sem(mean_docked_ships):.2f}")
    print(f" - Nº medio barcos en cola : {np.mean(mean_queues):.2f} +/- {stats.sem(mean_queues):.2f}")
    print(f" - Nº máximo barcos en cola : {np.mean(max_queues):.1f} (Máx absoluto: {np.max(max_queues):.1f})")


if __name__ == "__main__":
    run_experiment("Caso Base", num_tugs=10, num_berths=20)
    run_experiment("Opción A (+3 Remolcadores)", num_tugs=13, num_berths=20)
    run_experiment("Opción B (+5 Muelles)", num_tugs=10, num_berths=25)