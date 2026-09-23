"""Smart Campus Digital Twin: self-contained Streamlit web application."""
from collections import deque
from dataclasses import dataclass, field
import heapq
import math
import random
import time

import plotly.graph_objects as go
import streamlit as st


def clamp(value, low, high):
    return max(low, min(high, value))


@dataclass
class Place:
    code: str
    name: str
    kind: str
    node: tuple
    capacity: int
    inside: int = 0
    closed_until: int = 0


@dataclass
class Parking:
    code: str
    node: tuple
    capacity: int
    occupied: int = 0
    blocked: int = 0
    blocked_until: int = 0
    trend: float = 0.0

    @property
    def free(self):
        return max(0, self.capacity - self.occupied)


@dataclass
class Student:
    number: int
    node: tuple
    destination: str = ""
    path: list = field(default_factory=list)
    index: int = 0
    dwell: int = 0
    inside: str = ""


@dataclass
class Bike:
    number: int
    node: tuple
    lot: str = ""
    destination: str = ""
    path: list = field(default_factory=list)
    index: int = 0
    dwell: int = 0


class RoadNetwork:
    """Orthogonal road grid with weighted Dijkstra and temporary closures."""

    def __init__(self):
        self.graph = {(x, y): {} for x in range(9) for y in range(7)}
        self.closed = {}  # edge -> reopening tick
        for x, y in self.graph:
            for dx, dy in ((1, 0), (0, 1)):
                other = (x + dx, y + dy)
                if other in self.graph:
                    self.graph[(x, y)][other] = 1
                    self.graph[other][(x, y)] = 1

        # Permanent campus barriers make routes meaningfully different from lines.
        for a, b in [((3, 1), (4, 1)), ((3, 2), (4, 2)),
                     ((3, 3), (4, 3)), ((6, 3), (6, 4)),
                     ((6, 4), (6, 5)), ((1, 4), (2, 4))]:
            self.graph[a].pop(b, None)
            self.graph[b].pop(a, None)

    @staticmethod
    def edge(a, b):
        return tuple(sorted((a, b)))

    def open_edge(self, a, b, tick):
        return self.closed.get(self.edge(a, b), 0) <= tick

    def route(self, start, goal, tick):
        if start not in self.graph or goal not in self.graph:
            return [], math.inf
        costs = {start: 0}
        previous = {}
        queue = [(0, start)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != costs[node]:
                continue
            if node == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                path.reverse()
                return path, distance
            for neighbor, weight in self.graph[node].items():
                if not self.open_edge(node, neighbor, tick):
                    continue
                next_cost = distance + weight
                if next_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = next_cost
                    previous[neighbor] = node
                    heapq.heappush(queue, (next_cost, neighbor))
        return [], math.inf

    def active_edges(self, tick):
        for a, neighbors in self.graph.items():
            for b in neighbors:
                if a < b:
                    yield a, b, not self.open_edge(a, b, tick)


class SensorHub:
    """Validated observations: bad/missing/late/offline samples fall back."""

    def __init__(self, rng):
        self.rng = rng
        self.last_good = {}
        self.offline_until = {}
        self.readings = {}
        self.fault_count = 0

    def sample(self, key, truth, low, high, tick):
        previous = self.last_good.get(key)
        roll = self.rng.random()
        if self.offline_until.get(key, 0) > tick or roll < 0.015:
            raw, status = None, "OFFLINE"
        elif roll < 0.055:
            raw, status = None, "MISSING"
        elif roll < 0.095:
            raw, status = high + 999, "OUTLIER"
        elif roll < 0.135 and previous is not None:
            raw, status = previous[0], "DELAYED"
        else:
            raw, status = truth, "OK"
        valid = isinstance(raw, (int, float)) and not isinstance(raw, bool)
        valid = valid and math.isfinite(raw) and low <= raw <= high
        if valid and status == "OK":
            self.last_good[key] = (raw, tick)
        if status != "OK" or not valid:
            self.fault_count += 1
            # A missing reading has an explicit uncertain status, not fake zero.
            value = previous[0] if previous else None
            age = tick - previous[1] if previous else math.inf
            status = status if status != "OK" else "OUTLIER"
        else:
            value, age = raw, 0
        self.readings[key] = (value, status, age)

    def get(self, key):
        return self.readings.get(key, (None, "MISSING", math.inf))


class CampusWorld:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.roads = RoadNetwork()
        self.tick = 0
        self.minutes = 8 * 60
        self.sensors = SensorHub(self.rng)
        definitions = [
            ("A1", "Engineering", "teaching", (2, 5), 90),
            ("A2", "Mathematics", "teaching", (4, 5), 75),
            ("A3", "Science", "teaching", (7, 5), 90),
            ("A4", "Humanities", "teaching", (1, 3), 70),
            ("A5", "Design", "teaching", (3, 3), 70),
            ("A6", "Business", "teaching", (5, 3), 85),
            ("A7", "Computing", "teaching", (7, 3), 85),
            ("A8", "Sports Hall", "teaching", (4, 1), 65),
            ("C1", "North Dining", "canteen", (2, 6), 90),
            ("C2", "Central Dining", "canteen", (5, 4), 100),
            ("C3", "South Dining", "canteen", (6, 1), 85),
            ("L1", "Main Library", "study", (1, 5), 80),
            ("L2", "Learning Hub", "study", (7, 1), 65),
            ("G1", "West Gate", "gate", (0, 3), 300),
            ("G2", "East Gate", "gate", (8, 3), 300),
        ]
        self.places = {code: Place(code, name, kind, node, capacity)
                       for code, name, kind, node, capacity in definitions}
        parking_nodes = [(0, 5), (3, 6), (5, 6), (8, 6), (0, 1),
                         (2, 2), (4, 4), (8, 4), (5, 0), (8, 0)]
        capacities = [17, 20, 15, 18, 16, 19, 22, 17, 21, 18]
        self.parking = {f"P{i + 1}": Parking(f"P{i + 1}", node, cap)
                        for i, (node, cap) in enumerate(zip(parking_nodes, capacities))}
        self.students = []
        self.bikes = []
        self.logs = deque(maxlen=6)
        self.history = {name: [] for name in
                        ("time", "parking", "population", "canteen", "study")}
        self.start = "G1"
        self.target = "A1"
        self.result = {}
        for number in range(260):
            place = self._pick_destination()
            student = Student(number, place.node, inside=place.code,
                              dwell=self.rng.randint(1, 6))
            self.students.append(student)
        lots = list(self.parking.values())
        for number in range(115):
            available = [lot for lot in lots if lot.occupied < lot.capacity]
            if not available:
                break
            lot = self.rng.choice(available)
            lot.occupied += 1
            self.bikes.append(Bike(number, lot.node, lot=lot.code,
                                   dwell=self.rng.randint(1, 7)))
        self._count_people()
        self._update_sensors()
        self.recommend()
        self._record()
        self.log("Digital twin online: 260 students, 115 bikes")

    def log(self, message):
        self.logs.appendleft(f"{self.clock()}  {message}")

    def clock(self):
        minute = self.minutes % (24 * 60)
        return f"{minute // 60:02d}:{minute % 60:02d}"

    def _pick_destination(self):
        hour = (self.minutes % 1440) / 60
        if 11.4 <= hour < 13.4 or 17.4 <= hour < 19:
            weights = {"canteen": 7, "teaching": 2, "study": 1, "gate": 0.3}
        elif 8 <= hour < 11.4 or 13.4 <= hour < 17.4:
            weights = {"canteen": 0.5, "teaching": 5,
                       "study": 2, "gate": 0.2}
        else:
            weights = {"canteen": 1, "teaching": 0.6,
                       "study": 4, "gate": 1}
        choices = [p for p in self.places.values()
                   if p.closed_until <= self.tick and p.kind != "gate"
                   and p.inside < p.capacity]
        if not choices:
            choices = [p for p in self.places.values()
                       if p.closed_until <= self.tick and p.kind != "gate"]
        if not choices:
            return self.places["G1"]
        return self.rng.choices(choices,
                                weights=[weights[p.kind] for p in choices])[0]

    def _count_people(self):
        for place in self.places.values():
            place.inside = 0
        for student in self.students:
            if student.inside in self.places:
                self.places[student.inside].inside += 1

    def _start_student_trip(self, student):
        destination = self._pick_destination()
        if destination.code == student.inside:
            student.dwell = self.rng.randint(1, 3)
            return
        path, _ = self.roads.route(student.node, destination.node, self.tick)
        if path:
            student.inside = ""
            student.destination = destination.code
            student.path = path
            student.index = 0
        else:
            student.dwell = 1

    def _advance_students(self):
        for student in self.students:
            if student.inside:
                student.dwell -= 1
                if student.dwell <= 0:
                    self._start_student_trip(student)
                continue
            if not student.path:
                self._start_student_trip(student)
                continue
            destination = self.places.get(student.destination)
            if destination is None or destination.closed_until > self.tick:
                student.path = []
                student.destination = ""
                continue
            next_index = student.index + 1
            if (next_index >= len(student.path) or
                    not self.roads.open_edge(student.node,
                                             student.path[next_index], self.tick)):
                student.path, _ = self.roads.route(
                    student.node, destination.node, self.tick)
                student.index = 0
                next_index = 1
            if not student.path:
                continue
            if next_index < len(student.path):
                student.index = next_index
                student.node = student.path[next_index]
            if student.node == destination.node:
                student.inside = destination.code
                student.path = []
                student.dwell = self.rng.randint(2, 6)
        self._count_people()

    def _advance_bikes(self):
        old = {code: lot.occupied for code, lot in self.parking.items()}
        for bike in self.bikes:
            if bike.lot:
                bike.dwell -= 1
                if bike.dwell > 0:
                    continue
                choices = [lot for lot in self.parking.values()
                           if lot.code != bike.lot and lot.free > 0]
                if not choices:
                    bike.dwell = 2
                    continue
                destination = self.rng.choice(choices)
                path, _ = self.roads.route(bike.node, destination.node, self.tick)
                if not path:
                    bike.dwell = 2
                    continue
                self.parking[bike.lot].occupied -= 1
                bike.lot = ""
                bike.destination = destination.code
                bike.path = path
                bike.index = 0
            if not bike.lot:
                destination = self.parking.get(bike.destination)
                if destination is None:
                    continue
                next_index = bike.index + 1
                if (next_index >= len(bike.path) or
                        not self.roads.open_edge(bike.node,
                                                 bike.path[next_index], self.tick)):
                    bike.path, _ = self.roads.route(
                        bike.node, destination.node, self.tick)
                    bike.index = 0
                    next_index = 1
                if bike.path and next_index < len(bike.path):
                    bike.index = next_index
                    bike.node = bike.path[next_index]
                if bike.node == destination.node:
                    if destination.free:
                        destination.occupied += 1
                        bike.lot = destination.code
                        bike.destination = ""
                        bike.path = []
                        bike.dwell = self.rng.randint(2, 8)
                    else:
                        alternatives = [lot for lot in self.parking.values()
                                        if lot.free > 0]
                        if alternatives:
                            bike.destination = self.rng.choice(alternatives).code
        for code, lot in self.parking.items():
            lot.trend = 0.6 * lot.trend + 0.4 * (lot.occupied - old[code])

    def _time_population(self):
        hour = (self.minutes % 1440) / 60
        if 8 <= hour < 10 and len(self.students) < 320:
            count = 2
        elif 17 <= hour < 22 and len(self.students) > 220:
            count = -2
        else:
            count = 0
        if count > 0:
            gate = self.rng.choice(["G1", "G2"])
            for _ in range(count):
                number = max((s.number for s in self.students), default=-1) + 1
                student = Student(number, self.places[gate].node)
                self.students.append(student)
                self._start_student_trip(student)
        elif count < 0:
            candidates = [s for s in self.students if s.inside == ""
                          and s.node[0] in (0, 8)]
            for student in candidates[:min(-count, len(candidates))]:
                self.students.remove(student)

    def _update_sensors(self):
        hub = self.sensors
        for lot in self.parking.values():
            hub.sample("park:" + lot.code, lot.occupied,
                       0, lot.capacity, self.tick)
        for place in self.places.values():
            hub.sample("people:" + place.code, place.inside,
                       0, max(place.capacity * 4, 400), self.tick)
            if place.kind == "study":
                hub.sample("seat:" + place.code,
                           max(0, place.capacity - place.inside),
                           0, place.capacity, self.tick)
        temperature = 23 + 4 * math.sin(self.minutes / 1440 * math.tau)
        hub.sample("environment", round(temperature, 1), -20, 55, self.tick)

    def _record(self):
        h = self.history
        h["time"].append(self.minutes)
        h["population"].append(len(self.students))
        h["parking"].append(sum(p.occupied for p in self.parking.values()) /
                            max(1, sum(p.capacity for p in self.parking.values())) * 100)
        for kind, key in (("canteen", "canteen"), ("study", "study")):
            places = [p for p in self.places.values() if p.kind == kind]
            h[key].append(sum(p.inside for p in places) /
                          max(1, sum(p.capacity for p in places)) * 100)
        # Bound memory over arbitrarily long running sessions.
        if len(h["time"]) > 144:
            for series in h.values():
                del series[0]

    def tick_once(self):
        self.tick += 1
        self.minutes += 10
        for lot in self.parking.values():
            if lot.blocked and lot.blocked_until <= self.tick:
                lot.occupied -= lot.blocked
                lot.blocked = 0
                self.log(lot.code + " temporary occupation cleared")
        self._time_population()
        self._advance_students()
        self._advance_bikes()
        self._update_sensors()
        if self.rng.random() < 0.035:
            self.generate_event()
        self.recommend()
        self._record()

    def generate_event(self, kind=None):
        kind = kind or self.rng.choice(("parking", "road", "canteen",
                                        "sensor", "building", "surge"))
        if kind == "parking":
            lot = self.rng.choice(list(self.parking.values()))
            reserve = lot.free
            lot.blocked += reserve
            lot.occupied += reserve
            lot.blocked_until = max(lot.blocked_until, self.tick + 4)
            self.log(f"EVENT: {lot.code} suddenly FULL")
        elif kind == "road":
            candidates = [(a, b) for a, b, closed in
                          self.roads.active_edges(self.tick) if not closed]
            if candidates:
                # Prefer a road on the displayed route, so the event usually
                # demonstrates an actual reroute instead of a distant closure.
                route = self.result.get("direct", [])
                on_route = [self.roads.edge(a, b)
                            for a, b in zip(route, route[1:])
                            if self.roads.edge(a, b) in candidates]
                a, b = self.rng.choice(on_route if on_route and
                                       self.rng.random() < 0.75 else candidates)
                self.roads.closed[self.roads.edge(a, b)] = self.tick + 5
                self.log(f"EVENT: road {a}-{b} closed for 50 min")
        elif kind == "canteen":
            places = [p for p in self.places.values() if p.kind == "canteen"]
            place = self.rng.choice(places)
            # An actual crowd of students arrives, rather than changing a label.
            arrivals = min(25, max(0, 450 - len(self.students)))
            for _ in range(arrivals):
                number = max((s.number for s in self.students), default=-1) + 1
                self.students.append(Student(number, place.node,
                                             inside=place.code, dwell=3))
            self._count_people()
            self.log(f"EVENT: {arrivals} people crowd {place.code}")
        elif kind == "sensor":
            key = "park:" + self.rng.choice(list(self.parking))
            self.sensors.offline_until[key] = self.tick + 4
            self.log(f"EVENT: {key} sensor offline")
        elif kind == "building":
            places = [p for p in self.places.values() if p.kind != "gate"]
            place = self.rng.choice(places)
            place.closed_until = self.tick + 5
            for student in self.students:
                if student.inside == place.code:
                    student.inside = ""
                    student.dwell = 0
                    student.path = []
            self._count_people()
            self.log(f"EVENT: {place.code} closed for 50 min")
        else:
            gate = self.rng.choice(("G1", "G2"))
            arrivals = min(30, max(0, 450 - len(self.students)))
            for _ in range(arrivals):
                number = max((s.number for s in self.students), default=-1) + 1
                student = Student(number, self.places[gate].node)
                self.students.append(student)
                self._start_student_trip(student)
            self.log(f"EVENT: {arrivals} people enter at {gate}")
        self._update_sensors()
        self.recommend()
        self._record()

    def set_trip(self, start, target):
        start = start.strip().upper()
        target = target.strip().upper()
        if start not in self.places and start not in self.parking:
            self.log("Invalid start code; choose a map label")
            return False
        if target not in self.places or self.places[target].kind == "gate":
            self.log("Invalid destination code; choose a building")
            return False
        self.start, self.target = start, target
        self.recommend()
        self.log(f"Trip selected: {start} to {target}")
        return True

    def _node(self, code):
        entity = self.places.get(code) or self.parking.get(code)
        return entity.node if entity else None

    def _amenity(self, kind, target_node):
        options = []
        for place in self.places.values():
            if place.kind != kind or place.closed_until > self.tick:
                continue
            path, distance = self.roads.route(target_node, place.node, self.tick)
            if not path:
                continue
            sensor_key = ("seat:" if kind == "study" else "people:") + place.code
            value, status, age = self.sensors.get(sensor_key)
            # Never promise a seat known to be physically occupied.
            free = max(0, place.capacity - place.inside)
            if free == 0:
                continue
            observed = value if value is not None else (
                free if kind == "study" else place.inside)
            observed_free = observed if kind == "study" else place.capacity - observed
            conservative_free = clamp(min(free, observed_free), 0, place.capacity)
            freshness = 1 if status == "OK" else (0.75 if age <= 2 else 0.45)
            score = 100 * (0.40 / (1 + distance / 5)
                           + 0.40 * conservative_free / place.capacity
                           + 0.20 * freshness)
            options.append((score, place, distance, status))
        return max(options, key=lambda row: row[0], default=None)

    def recommend(self):
        start_node = self._node(self.start)
        target = self.places.get(self.target)
        if start_node is None or target is None:
            self.result = {"direct": [], "parking": None,
                           "canteen": None, "study": None, "via": []}
            return self.result
        closed = target.closed_until > self.tick
        direct, direct_len = ([], math.inf) if closed else self.roads.route(
            start_node, target.node, self.tick)
        lots = []
        if not closed:
            for lot in self.parking.values():
                if lot.free <= 0:
                    continue
                first, ride = self.roads.route(start_node, lot.node, self.tick)
                second, walk = self.roads.route(lot.node, target.node, self.tick)
                if not first or not second:
                    continue
                value, status, age = self.sensors.get("park:" + lot.code)
                observed_free = lot.capacity - value if value is not None else lot.free
                free = clamp(min(lot.free, observed_free), 0, lot.capacity)
                eta = max(1, ride // 3)
                forecast = lot.occupied + max(0, lot.trend) * eta
                risk = clamp(forecast / lot.capacity, 0, 1)
                if status != "OK":
                    risk = clamp(risk + (0.13 if age <= 2 else 0.3), 0, 1)
                score = 100 * (0.35 / (1 + walk / 5)
                               + 0.25 * free / lot.capacity
                               + 0.15 * (1 - lot.occupied / lot.capacity)
                               + 0.25 * (1 - risk))
                lots.append((score, lot, first, second, risk, status, walk))
        winner = max(lots, key=lambda row: row[0], default=None)
        self.result = {
            "direct": direct, "direct_len": direct_len,
            "parking": winner, "via":
                winner[2] + winner[3][1:] if winner else [],
            "canteen": self._amenity("canteen", target.node),
            "study": self._amenity("study", target.node),
            "closed": closed,
        }
        return self.result


def xy(node):
    return (node[0] * 10 + 10, node[1] * 10 + 10)


def map_figure(world):
    """An interactive, touch-friendly map with labels visible without hover."""
    fig = go.Figure()
    for closed, color, dash, width in ((False, "#93a9a6", "solid", 2),
                                       (True, "#ef5350", "dash", 4)):
        xs, ys = [], []
        for a, b, is_closed in world.roads.active_edges(world.tick):
            if is_closed == closed:
                ax, ay = xy(a)
                bx, by = xy(b)
                xs.extend((ax, bx, None))
                ys.extend((ay, by, None))
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(color=color, width=width, dash=dash),
                                     name="Closed road" if closed else "Road",
                                     hoverinfo="skip"))

    if world.students:
        coords = [xy(s.node) for s in world.students]
        fig.add_trace(go.Scatter(
            x=[point[0] + (student.number % 7 - 3) * 0.45
               for point, student in zip(coords, world.students)],
            y=[point[1] + (student.number % 5 - 2) * 0.45
               for point, student in zip(coords, world.students)],
            mode="markers", marker=dict(size=5, color="#42a5f5", opacity=0.48),
            name="Students", hoverinfo="skip"))
    if world.bikes:
        coords = [xy(b.node) for b in world.bikes]
        fig.add_trace(go.Scatter(
            x=[point[0] + (bike.number % 5 - 2) * 0.45
               for point, bike in zip(coords, world.bikes)],
            y=[point[1] + (bike.number % 7 - 3) * 0.45
               for point, bike in zip(coords, world.bikes)],
            mode="markers", marker=dict(size=7, color="#fb923c", symbol="triangle-up",
                                        opacity=0.65),
            name="E-bikes", hoverinfo="skip"))

    for key, color, dash, label in (("direct", "#2563eb", "solid", "Shortest route"),
                                    ("via", "#22c55e", "dash", "Via recommended parking")):
        path = world.result.get(key, [])
        if path:
            coords = [xy(node) for node in path]
            fig.add_trace(go.Scatter(x=[p[0] for p in coords],
                                     y=[p[1] for p in coords], mode="lines",
                                     line=dict(color=color, width=5, dash=dash),
                                     name=label, hoverinfo="skip"))

    start_node = world._node(world.start)
    if start_node:
        x, y = xy(start_node)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers",
                                 marker=dict(size=30, color="#22d3ee"),
                                 name="FROM", hoverinfo="skip"))
    target = world.places.get(world.target)
    if target:
        x, y = xy(target.node)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers",
                                 marker=dict(size=31, color="#fde047"),
                                 name="TO", hoverinfo="skip"))
    best = world.result.get("parking")
    if best:
        x, y = xy(best[1].node)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers",
                                 marker=dict(size=32, color="#86efac"),
                                 name="Recommended parking", hoverinfo="skip"))

    styles = {"teaching": ("#315ab6", "square", "Teaching"),
              "canteen": ("#e97835", "diamond", "Canteen"),
              "study": ("#9559c3", "pentagon", "Study"),
              "gate": ("#364152", "x", "Gate")}
    for kind, (color, symbol, label) in styles.items():
        places = [p for p in world.places.values() if p.kind == kind]
        if not places:
            continue
        coords = [xy(p.node) for p in places]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in coords], y=[p[1] for p in coords],
            mode="markers+text", text=[p.code for p in places],
            textposition="top right", textfont=dict(color="#14273b", size=12),
            marker=dict(size=15, color=["#9ca3af" if p.closed_until > world.tick
                                        else color for p in places], symbol=symbol,
                        line=dict(color="white", width=1)),
            name=label, customdata=[[p.name, p.inside, p.capacity] for p in places],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}/%{customdata[2]} people<extra></extra>"))

    lots = list(world.parking.values())
    if lots:
        coords = [xy(p.node) for p in lots]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in coords], y=[p[1] for p in coords],
            mode="markers+text",
            text=[f"{p.code}<br>{p.occupied}/{p.capacity}" for p in lots],
            textposition="bottom right", textfont=dict(color="#173a2b", size=10),
            marker=dict(size=13, symbol="cross", color=["#ef4444" if p.free == 0
                                                        else "#16a34a" for p in lots]),
            name="Parking", hoverinfo="skip"))
    fig.update_layout(
        height=560, margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor="#edf6f3", plot_bgcolor="#edf6f3",
        legend=dict(orientation="h", y=-0.08, x=0, font=dict(size=10)),
        xaxis=dict(range=[3, 97], visible=False, fixedrange=True),
        yaxis=dict(range=[2, 80], visible=False, scaleanchor="x", scaleratio=1,
                   fixedrange=True),
        hovermode="closest", dragmode=False, showlegend=True,
    )
    return fig


def trend_figure(world, key, color):
    values = world.history.get(key, [])
    times = world.history.get("time", [])
    fig = go.Figure()
    if values and times:
        fig.add_trace(go.Scatter(x=times, y=values, mode="lines+markers",
                                 line=dict(width=3, color=color),
                                 marker=dict(size=4), showlegend=False))
    fig.update_layout(height=190, margin=dict(l=12, r=8, t=10, b=25),
                      xaxis=dict(title="Campus minutes", tickfont=dict(size=10)),
                      yaxis=dict(tickfont=dict(size=10)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def act(action):
    """Callbacks run before Streamlit recreates widgets on a fragment rerun."""
    session = st.session_state
    world = session.world
    if action == "reset":
        session.world = CampusWorld()
        session.origin = "G1"
        session.destination = "A1"
        session.event_type = "Random"
        session.running = True
        session.speed = 1
    elif action == "route":
        world.set_trip(session.origin, session.destination)
    elif action == "pause":
        session.running = False
        world.log("Simulation paused")
    elif action == "resume":
        session.running = True
        world.log("Simulation resumed")
    elif action.startswith("speed:"):
        session.speed = int(action.split(":")[1])
        world.log(f"Speed set to x{session.speed}")
    elif action == "event":
        types = {"Parking full": "parking", "Road closed": "road",
                 "Canteen crowd": "canteen", "Sensor offline": "sensor",
                 "Building closed": "building", "People surge": "surge"}
        world.generate_event(types.get(session.event_type))
    session.last_advance = time.monotonic()


def draw_dashboard(world, running, speed):
    used = sum(p.occupied for p in world.parking.values())
    capacity = sum(p.capacity for p in world.parking.values())
    temp, status, _ = world.sensors.get("environment")
    st.markdown(f"#### {'🟢 Live' if running else '⏸️ Paused'} · Day "
                f"{1 + world.minutes // 1440} · {world.clock()} · ×{speed}")
    cards = st.columns(4)
    cards[0].metric("Students", len(world.students))
    cards[1].metric("E-bikes", len(world.bikes))
    cards[2].metric("Parking occupied", f"{used}/{capacity}")
    cards[3].metric("Temperature", f"{temp}°C" if temp is not None else "—",
                    help=f"Environmental sensor: {status}")


def draw_recommendations(world):
    result = world.result
    direct = result.get("direct") or []
    st.subheader("Route & recommendations")
    if direct:
        st.success(f"{world.start} → {world.target}: shortest road route "
                   f"{result['direct_len']} segments · "
                   + " → ".join(f"({x},{y})" for x, y in direct))
    else:
        st.warning("Destination closed or no reachable road route.")
    parking = result.get("parking")
    canteen = result.get("canteen")
    study = result.get("study")
    one, two, three = st.columns(3)
    if parking:
        score, lot, _, _, risk, status, walk = parking
        one.metric("Recommended parking", lot.code, f"Score {score:.0f}/100")
        one.caption(f"{lot.free} spaces free · {walk} road segments to target · "
                    f"arrival risk {risk:.0%} · sensor {status}")
    else:
        one.warning("No reachable free parking")
    if canteen:
        two.metric("Best canteen", canteen[1].code,
                   f"Score {canteen[0]:.0f}/100")
        two.caption(f"{canteen[1].name} · "
                    f"{canteen[1].inside}/{canteen[1].capacity} people")
    else:
        two.warning("No canteen available")
    if study:
        three.metric("Best study area", study[1].code,
                     f"Score {study[0]:.0f}/100")
        three.caption(f"{study[1].name} · "
                      f"{max(0, study[1].capacity - study[1].inside)} seats free")
    else:
        three.warning("No available study seats")


def draw_status(world):
    st.subheader("Live campus status")
    left, right = st.columns(2)
    with left:
        st.markdown("**Parking sensors**")
        lots = list(world.parking.values())
        for lot in lots:
            _, state, age = world.sensors.get("park:" + lot.code)
            mark = " ⚠️" if state != "OK" else ""
            st.progress(min(1.0, lot.occupied / max(1, lot.capacity)),
                        text=f"{lot.code} · {lot.occupied}/{lot.capacity} occupied · "
                             f"{lot.free} free{mark}")
        st.caption("⚠️ Sensor reading degraded; last valid data is retained and "
                   "arrival risk is raised.")
    with right:
        st.markdown("**Buildings & people counters**")
        rows = []
        for p in world.places.values():
            if p.kind == "gate":
                continue
            rows.append({"Facility": f"{p.code} · {p.name}",
                         "Type": p.kind.title(),
                         "People": p.inside,
                         "Capacity": p.capacity,
                         "Crowding": f"{p.inside / max(1, p.capacity):.0%}",
                         "Status": "CLOSED" if p.closed_until > world.tick else "OPEN"})
        st.dataframe(rows, hide_index=True, width="stretch", height=390)
        st.markdown("**Study seats**")
        for p in world.places.values():
            if p.kind == "study":
                value, state, _ = world.sensors.get("seat:" + p.code)
                st.caption(f"{p.code}: {max(0, p.capacity - p.inside)} / "
                           f"{p.capacity} free · seat sensor {state}")


def draw_analytics(world):
    st.subheader("Analytics · last 24 simulated hours")
    specs = [("parking", "Parking occupancy %", "#22c55e"),
             ("population", "Campus population", "#38bdf8"),
             ("canteen", "Canteen congestion %", "#fb923c"),
             ("study", "Study space usage %", "#a78bfa")]
    for index in (0, 2):
        columns = st.columns(2)
        for col, (key, label, color) in zip(columns, specs[index:index + 2]):
            with col:
                st.caption(label)
                st.plotly_chart(trend_figure(world, key, color),
                                width="stretch",
                                config={"displayModeBar": False},
                                key=f"chart_{key}")


@st.fragment(run_every=1)
def live_app():
    session = st.session_state
    world = session.world

    # A widget interaction does not advance time unless a full second elapsed.
    now = time.monotonic()
    if session.running and now - session.last_advance >= 0.85:
        try:
            for _ in range(session.speed):
                world.tick_once()
        except Exception as exc:
            world.log(f"Simulation recovered: {type(exc).__name__}")
        session.last_advance = now

    draw_dashboard(world, session.running, session.speed)

    locations = list(world.places) + list(world.parking)
    destinations = [p.code for p in world.places.values() if p.kind != "gate"]
    from_col, to_col = st.columns(2)
    from_col.selectbox("FROM · current location", locations, key="origin",
                       format_func=lambda code: f"{code} · " +
                       (world.places[code].name if code in world.places
                        else "E-bike parking"))
    to_col.selectbox("TO · destination building", destinations,
                     key="destination",
                     format_func=lambda code: f"{code} · {world.places[code].name}")
    st.button("🧭 Route + Recommend", on_click=act, args=("route",),
              type="primary", width="stretch")

    row = st.columns([1, 1, 1, 1, 1])
    row[0].button("⏸ Pause", on_click=act, args=("pause",),
                  width="stretch", disabled=not session.running)
    row[1].button("▶ Resume", on_click=act, args=("resume",),
                  width="stretch", disabled=session.running)
    for index, amount in enumerate((1, 2, 5), 2):
        row[index].button(f"×{amount}", on_click=act,
                          args=(f"speed:{amount}",), width="stretch",
                          type="primary" if session.speed == amount else "secondary")
    event_col, event_button, reset_col = st.columns([2, 1, 1])
    event_col.selectbox("Campus event", ["Random", "Parking full", "Road closed",
                                       "Canteen crowd", "Sensor offline",
                                       "Building closed", "People surge"],
                        key="event_type", label_visibility="collapsed")
    event_button.button("⚡ Generate Event", on_click=act, args=("event",),
                        width="stretch")
    reset_col.button("↻ Reset", on_click=act, args=("reset",),
                     width="stretch")

    st.plotly_chart(map_figure(world), width="stretch",
                    config={"displayModeBar": False, "scrollZoom": False},
                    key="campus_map")
    st.caption("Blue line: shortest road route · Green dashed line: route via parking · "
               "Red dashed line: closed road · Labels are always visible on touchscreens.")
    draw_recommendations(world)
    draw_status(world)
    draw_analytics(world)
    st.subheader("Event log")
    for message in world.logs:
        st.write("• " + message)
    st.caption(f"IoT samples rejected, delayed, or missing: {world.sensors.fault_count}. "
               "Each browser session has an independent simulation.")


def main():
    st.set_page_config(page_title="Smart Campus Digital Twin", page_icon="🏫",
                       layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""<style>
    .stButton > button { min-height: 3rem; border-radius: 0.8rem; font-weight: 700; }
    [data-testid="stMetric"] { padding: .8rem; border-radius: .8rem;
        background: rgba(110, 160, 175, .10); }
    @media (max-width: 900px) {
        .block-container { padding-left: .7rem; padding-right: .7rem; }
        .stButton > button { min-height: 3.3rem; }
    }
    </style>""", unsafe_allow_html=True)
    st.title("🏫 Smart Campus Digital Twin")
    st.caption("Live university simulation · road-aware routing · IoT-assisted decisions")
    if "world" not in st.session_state:
        st.session_state.world = CampusWorld()
        st.session_state.origin = "G1"
        st.session_state.destination = "A1"
        st.session_state.event_type = "Random"
        st.session_state.running = True
        st.session_state.speed = 1
        st.session_state.last_advance = time.monotonic()
    live_app()


if __name__ == "__main__":
    main()
