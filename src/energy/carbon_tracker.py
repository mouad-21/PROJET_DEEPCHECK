"""Suivi energetique (Green IT) - version robuste Mac/CPU."""
import time
import csv
import functools
from datetime import datetime
from contextlib import ContextDecorator

from config import settings

try:
    from codecarbon import EmissionsTracker
    _HAS_CC = True
except Exception:
    _HAS_CC = False

CPU_POWER_W = 20.0
GRID_GCO2_PER_KWH = 50.0


def _append_csv(row: dict):
    path = settings.ENERGY_LOG_DIR / "emissions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


class EnergyTracker(ContextDecorator):
    def __init__(self, task_name: str = "task"):
        self.task_name = task_name
        self._tracker = None
        self._t0 = None
        self.result = {}

    def __enter__(self):
        self._t0 = time.perf_counter()
        if _HAS_CC:
            try:
                self._tracker = EmissionsTracker(
                    project_name=f"{settings.ENERGY_PROJECT_NAME}_{self.task_name}",
                    output_dir=str(settings.ENERGY_LOG_DIR),
                    output_file="emissions.csv",
                    country_iso_code=settings.ENERGY_COUNTRY_ISO,
                    log_level="error",
                    save_to_file=True,
                    allow_multiple_runs=True,
                )
                self._tracker.start()
            except Exception:
                self._tracker = None
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._t0
        emissions = None
        if self._tracker is not None:
            try:
                emissions = self._tracker.stop()
            except Exception:
                emissions = None

        energy_kwh = (CPU_POWER_W * elapsed) / 3_600_000
        if emissions is not None and emissions > 0:
            mode = "codecarbon"
            co2_kg = round(emissions, 8)
        else:
            mode = "estime (duree x puissance CPU)"
            co2_kg = round(energy_kwh * GRID_GCO2_PER_KWH / 1000, 8)

        self.result = {
            "task": self.task_name,
            "duree_s": round(elapsed, 3),
            "energy_kwh": round(energy_kwh, 8),
            "co2_kg": co2_kg,
            "mode": mode,
        }
        _append_csv({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task": self.result["task"],
            "duration": self.result["duree_s"],
            "energy_consumed": self.result["energy_kwh"],
            "emissions": self.result["co2_kg"],
            "mode": self.result["mode"],
        })
        return False


def track_energy(task_name: str = "task"):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with EnergyTracker(task_name) as t:
                out = fn(*args, **kwargs)
            wrapper.last_energy = t.result
            return out
        return wrapper
    return deco


if __name__ == "__main__":
    with EnergyTracker("demo") as t:
        sum(i * i for i in range(2_000_000))
    print(t.result)
