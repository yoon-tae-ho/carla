"""Small observers used by controller experiments."""

from __future__ import annotations

import csv
import os
from typing import Any, Iterable, List

from ..controllers.base import VehicleState
from .carla_adapter import import_carla, read_vehicle_state, vector_norm


class StateHistory:
    """Collect and write `VehicleState` rows."""

    def __init__(self):
        self.rows: List[VehicleState] = []

    def append(self, state: VehicleState) -> None:
        self.rows.append(state)

    def observe(self, world: Any, vehicle: Any, step: int, previous_state=None, control=None):
        state = read_vehicle_state(
            world,
            vehicle,
            step=step,
            previous_state=previous_state,
            control=control)
        self.append(state)
        return state

    def as_dicts(self):
        return [row.as_dict() for row in self.rows]

    def write_csv(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fields = tuple(VehicleState.__dataclass_fields__.keys())
        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.as_dict())


class CollisionObserver:
    """Attach a CARLA collision sensor and keep a simple event list."""

    def __init__(self, world: Any, vehicle: Any, label: str = "", carla_module: Any = None):
        self.world = world
        self.vehicle = vehicle
        self.label = label
        self.carla = carla_module or import_carla()
        self.sensor = None
        self.events = []

    def start(self):
        blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(
            blueprint,
            self.carla.Transform(),
            attach_to=self.vehicle)

        def on_collision(event):
            impulse = event.normal_impulse
            self.events.append({
                "label": self.label,
                "frame": event.frame,
                "other_actor": event.other_actor.type_id,
                "impulse": vector_norm(impulse),
            })

        self.sensor.listen(on_collision)
        return self

    def stop(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.stop()


def write_dict_rows(path: str, rows: Iterable[dict], fields: Iterable[str]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fields = tuple(fields)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
