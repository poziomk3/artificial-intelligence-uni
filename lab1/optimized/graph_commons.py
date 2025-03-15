from dataclasses import dataclass
from typing import NamedTuple
import heapq
from datetime import datetime, timedelta
import pandas as pd


@dataclass
class RouteData:
    start_stop: str
    end_stop: str
    line: str
    start_stop_lat: float
    start_stop_lon: float
    end_stop_lat: float
    end_stop_lon: float
    departure_time: pd.Timestamp
    arrival_time: pd.Timestamp


def fix_invalid_time(time_str: str) -> pd.Timestamp:
    """
    Corrects invalid time formats where the hour exceeds 24.

    :param time_str: A string representing a time (HH:MM:SS)
    :return: A corrected pd.Timestamp with a valid datetime format.
    """
    try:
        hh, mm, ss = map(int, time_str.split(":"))
        if hh >= 24:
            adjusted_time = datetime(1900, 1, 1, hh - 24, mm, ss) + timedelta(days=1)
        else:
            adjusted_time = datetime(1900, 1, 1, hh, mm, ss)

        return pd.Timestamp(adjusted_time)

    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}")


def calculate_time_delta(departure_time: pd.Timestamp, arrival_time: pd.Timestamp) -> pd.Timedelta:
    time_delta = arrival_time - departure_time
    if time_delta.total_seconds() < 0:
        time_delta += pd.Timedelta(days=1)
    return time_delta


class RoutePiece(NamedTuple):
    """A named tuple representing a single segment of a transit route."""
    line: str
    departure_time: pd.Timestamp
    arrival_time: pd.Timestamp
    time_delta: pd.Timedelta


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return not self.elements

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]
