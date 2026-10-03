"""Trip state machine: stable vehicle identity across tracker ID changes, speed, OD counting.

Trip status:  ACTIVE -> COMPLETED   (reached a different zone and stayed >= dwell_s, or left frame inside it)
              ACTIVE -> GHOST       (track vanished outside any exit zone; waits ghost_ttl_s to be re-identified)
              GHOST  -> ACTIVE      (a new/old track ID appears where the constant-velocity prediction says)
              GHOST  -> LOST        (never re-appeared)
"""
import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

HEAVY = {"car", "truck", "bus"}  # detector flips between these a lot -> treat as compatible


@dataclass
class Trip:
    id: int
    t0: float
    last_t: float
    origin: str | None = None
    origin_source: str = ""          # "zone" (touched at birth) | "guess" (entry-side heuristic)
    dest: str | None = None
    status: str = "ACTIVE"
    end_reason: str = ""
    zone: str | None = None
    cand: str | None = None
    cand_t0: float = 0.0
    track_ids: list = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)
    hist: deque = field(default_factory=deque)
    speed_kmh: float | None = None
    speeds: list = field(default_factory=list)
    vel: tuple = (0.0, 0.0)
    xy: tuple = (0.0, 0.0)
    n_obs: int = 0
    merges: int = 0
    t_complete: float | None = None

    @property
    def cls(self):
        return self.votes.most_common(1)[0][0] if self.votes else "?"


class TripManager:
    def __init__(self, zone_centroids: dict, metric=True, stitch=True, dwell_s=0.3, lost_after_s=0.7,
                 ghost_ttl_s=4.0, gate=6.0, gate_speed_factor=0.2, speed_window_s=0.8,
                 speed_min_span_s=0.4, speed_ema=0.35, max_kmh=160.0, guess_slack=12.0, min_move=0.3):
        self.centroids = zone_centroids
        names = list(zone_centroids)
        self.od = {o: {d: 0 for d in names} for o in names}
        self.metric, self.stitch = metric, stitch
        self.dwell_s, self.lost_after_s, self.ghost_ttl = dwell_s, lost_after_s, ghost_ttl_s
        self.gate, self.gate_k = gate, gate_speed_factor
        self.sw, self.smin, self.sema, self.max_kmh = speed_window_s, speed_min_span_s, speed_ema, max_kmh
        self.guess_slack, self.min_move = guess_slack, min_move
        self.trips, self.alias, self.finished, self.events = {}, {}, [], []
        self._next, self.n_completed = 1, 0

    # ---------------------------------------------------------------- public
    def update(self, t, obs):
        """obs: list of (track_id, cls, zone, xy_world, valid). Returns {track_id: Trip}."""
        seen, out, pending = set(), {}, []
        for o in obs:                                   # pass 1: tracks that already belong to a trip
            trip = self.trips.get(self.alias.get(o[0]))
            if trip is None:
                pending.append(o)
                continue
            self._observe(trip, t, o[1], o[2], o[3], o[4])
            seen.add(trip.id); out[o[0]] = trip
        for tid, cls, zone, xy, valid in pending:       # pass 2: new tracks may re-identify an unseen trip
            trip = self._birth(tid, cls, zone, xy, t, seen)
            self._observe(trip, t, cls, zone, xy, valid)
            seen.add(trip.id); out[tid] = trip
        for trip in list(self.trips.values()):
            if trip.id in seen:
                continue
            gap = t - trip.last_t
            if trip.status == "ACTIVE" and gap >= self.lost_after_s:
                self._on_lost(trip)
            elif trip.status == "GHOST" and gap >= self.ghost_ttl:
                self._retire(trip, "LOST", "ghost_expired")
            elif trip.status == "COMPLETED" and gap >= 1.0:
                self._retire(trip, "COMPLETED", trip.end_reason)
        return out

    def finalize(self):
        for trip in list(self.trips.values()):
            st = {"ACTIVE": "INCOMPLETE", "GHOST": "LOST"}.get(trip.status, trip.status)
            self._retire(trip, st, "video_end" if st != "COMPLETED" else trip.end_reason)
        return self.finished

    # --------------------------------------------------------------- internals
    def _birth(self, tid, cls, zone, xy, t, seen):
        trip = None
        if self.stitch:
            best, bd = None, 1e9
            for g in self.trips.values():
                if g.id in seen or g.status not in ("GHOST", "ACTIVE") or g.last_t >= t \
                        or not self._compatible(g.cls, cls):
                    continue
                gap = t - g.last_t
                px, py = g.xy[0] + g.vel[0] * gap, g.xy[1] + g.vel[1] * gap
                d = math.hypot(px - xy[0], py - xy[1])
                limit = self.gate + self.gate_k * math.hypot(*g.vel) * gap
                if d <= limit and d < bd:
                    best, bd = g, d
            if best is not None:
                trip = best
                trip.status, trip.merges = "ACTIVE", trip.merges + 1
                trip.cand = None
                self.events.append({"type": "merge", "trip": trip.id, "new_track": tid,
                                    "old_tracks": list(trip.track_ids), "gap_s": round(t - trip.last_t, 2),
                                    "dist": round(bd, 2)})
        if trip is None:
            trip = Trip(id=self._next, t0=t, last_t=t)
            self._next += 1
            self.trips[trip.id] = trip
            if zone is not None:
                trip.origin, trip.origin_source = zone, "zone"
        trip.track_ids.append(tid)
        self.alias[tid] = trip.id
        return trip

    @staticmethod
    def _compatible(a, b):
        return a == b or (a in HEAVY and b in HEAVY)

    def _observe(self, trip, t, cls, zone, xy, valid):
        trip.last_t, trip.xy, trip.zone = t, xy, zone
        trip.n_obs += 1
        trip.votes[cls] += 1
        if trip.status == "GHOST":
            trip.status = "ACTIVE"
        h = trip.hist
        h.append((t, xy[0], xy[1]))
        while h and t - h[0][0] > self.sw:
            h.popleft()
        if len(h) >= 2 and h[-1][0] - h[0][0] >= self.smin:
            dt = h[-1][0] - h[0][0]
            vx, vy = (h[-1][1] - h[0][1]) / dt, (h[-1][2] - h[0][2]) / dt
            trip.vel = (vx, vy)
            if self.metric:
                v = math.hypot(vx, vy) * 3.6
                if v <= self.max_kmh:
                    trip.speed_kmh = v if trip.speed_kmh is None else self.sema * v + (1 - self.sema) * trip.speed_kmh
            if valid and trip.speed_kmh is not None:
                trip.speeds.append(trip.speed_kmh)
        if trip.status != "ACTIVE":
            return
        # entry-side heuristic: born mid-frame -> origin is the zone "behind" the motion direction
        if trip.origin is None and math.hypot(*trip.vel) > self.min_move:
            trip.origin, trip.origin_source = self._guess_origin(trip), "guess"
        if trip.origin and zone and zone != trip.origin:
            if trip.cand != zone:
                trip.cand, trip.cand_t0 = zone, t
            elif t - trip.cand_t0 >= self.dwell_s:      # hysteresis: must stay in the zone
                self._complete(trip, zone, t, "dwell")
        else:
            trip.cand = None

    def _guess_origin(self, trip):
        """Project backwards along the velocity; pick the zone closest to that ray (allowing slack if we're already in it)."""
        vx, vy = trip.vel
        n = math.hypot(vx, vy)
        ux, uy = -vx / n, -vy / n
        best, bd = None, 1e18
        for name, c in self.centroids.items():
            dx, dy = c[0] - trip.xy[0], c[1] - trip.xy[1]
            if dx * ux + dy * uy < -self.guess_slack:          # zone is ahead of us, not behind
                continue
            perp = abs(dx * uy - dy * ux)
            if perp < bd:
                best, bd = name, perp
        return best
    def _complete(self, trip, dest, t, reason):
        trip.status, trip.dest, trip.t_complete, trip.end_reason = "COMPLETED", dest, t, reason
        self.od[trip.origin][dest] += 1
        self.n_completed += 1

    def _on_lost(self, trip):
        if trip.origin and trip.zone and trip.zone != trip.origin:   # left the frame while inside an exit zone
            self._complete(trip, trip.zone, trip.last_t, "exit_on_loss")
        elif self.stitch:
            trip.status = "GHOST"
        else:
            self._retire(trip, "LOST", "track_lost")

    def _retire(self, trip, status, reason):
        sp = np.array(trip.speeds) if trip.speeds else None
        self.finished.append({
            "trip_id": trip.id, "track_ids": "|".join(map(str, trip.track_ids)), "cls": trip.cls,
            "origin": trip.origin, "origin_source": trip.origin_source, "dest": trip.dest,
            "status": status, "end_reason": reason, "t_start": round(trip.t0, 2), "t_end": round(trip.last_t, 2),
            "mean_kmh": None if sp is None else round(float(sp.mean()), 1),
            "p95_kmh": None if sp is None else round(float(np.percentile(sp, 95)), 1),
            "n_merges": trip.merges, "t_complete": (None if trip.t_complete is None else round(trip.t_complete, 2))})
        for tid in trip.track_ids:
            if self.alias.get(tid) == trip.id:
                del self.alias[tid]
        self.trips.pop(trip.id, None)
