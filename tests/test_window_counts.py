import pandas as pd
import pytest

from window_counts import window_counts

TRIPS = pd.DataFrame({
    "status": ["COMPLETED", "COMPLETED", "COMPLETED", "LOST", "COMPLETED"],
    "origin": ["FAR", "NEAR", "FAR", "FAR", "FAR"],
    "dest": ["NEAR", "FAR", "NEAR", None, "NEAR"],
    "t_complete": [21.0, 25.5, 30.0, None, 29.9],
})


def test_counts_by_completion_time_and_direction():
    assert window_counts(TRIPS, 20, 30) == {("FAR", "NEAR"): 2, ("NEAR", "FAR"): 1}   # t=30.0 is outside [20, 30)


def test_empty_window():
    assert window_counts(TRIPS, 0, 5) == {}


def test_old_runs_without_t_complete_are_rejected():
    with pytest.raises(ValueError):
        window_counts(TRIPS.drop(columns="t_complete"), 0, 10)
