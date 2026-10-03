"""Traffic condition (Smooth / Moderate / Heavy) from speed relative to free-flow, with hysteresis."""
from collections import deque


class TrafficClassifier:
    LEVELS = ["Smooth", "Moderate", "Heavy"]

    def __init__(self, window_s=30.0, eval_every_s=5.0, free_flow_kmh=None, smooth=0.7, heavy=0.4,
                 occ_heavy=None, hold=3, min_samples=20):
        self.win, self.every, self.ff = window_s, eval_every_s, free_flow_kmh
        self.smooth, self.heavy, self.occ_heavy, self.hold, self.min_samples = smooth, heavy, occ_heavy, hold, min_samples
        self.buf = deque()             # (t, speed_sum, n_speed, n_vehicles, n_completed)
        self.all_speeds = deque(maxlen=20000)
        self.label, self.cand, self.cand_n, self.next_eval = "Smooth", None, 0, window_s / 2
        self.timeline = []

    def add(self, t, speeds, n_vehicles, n_completed=0):
        self.buf.append((t, sum(speeds), len(speeds), n_vehicles, n_completed))
        self.all_speeds.extend(speeds[:3])
        while self.buf and t - self.buf[0][0] > self.win:
            self.buf.popleft()
        if t >= self.next_eval:
            self.next_eval = t + self.every
            return self._evaluate(t)
        return None

    def _free_flow(self):
        if self.ff:
            return self.ff
        if len(self.all_speeds) < 50:
            return 50.0
        s = sorted(self.all_speeds)
        return max(30.0, s[int(0.9 * len(s))])

    def _evaluate(self, t):
        n = sum(b[2] for b in self.buf)
        if n < self.min_samples:
            return None
        mean = sum(b[1] for b in self.buf) / n
        occ = sum(b[3] for b in self.buf) / len(self.buf)
        flow = sum(b[4] for b in self.buf) / max(self.buf[-1][0] - self.buf[0][0], 1e-6) * 60
        ratio = mean / self._free_flow()
        raw = "Smooth" if ratio >= self.smooth else ("Moderate" if ratio >= self.heavy else "Heavy")
        if self.occ_heavy and occ >= self.occ_heavy and raw == "Moderate":
            raw = "Heavy"
        if raw == self.label:
            self.cand, self.cand_n = None, 0
        elif raw == self.cand:
            self.cand_n += 1
            if self.cand_n >= self.hold:
                self.label, self.cand, self.cand_n = raw, None, 0
        else:
            self.cand, self.cand_n = raw, 1
        row = {"t": round(t, 1), "label": self.label, "raw": raw, "mean_kmh": round(mean, 1),
               "ratio": round(ratio, 2), "occupancy": round(occ, 1), "flow_veh_min": round(flow, 1)}
        self.timeline.append(row)
        return row
