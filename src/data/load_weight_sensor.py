"""
load_weight_sensor.py
get the raw seat and foot weight data, and get
the clean total weight
"""

import warnings
import sqlite3
import os
import pandas as pd
import numpy as np
from typing import Optional, List
from src.data import load_water_distance
import config


warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_seat_weight_raw(
    user_id: int,
    database_path: Optional[str] = config.DATABASE_PATH
) -> pd.Series:
    """
    Get the raw seat weight data from the database.

    :param user_id: an integer of user id
    :param database_path: a string of location of database
    :return: 1D pd Series mapping from timestamps to mmeasurements
    """
    sql_s = f"SELECT timestamp_ms, value FROM data WHERE data_capture_id={user_id} AND sensor_id=2"
    conn = sqlite3.connect(database_path)
    cursor = conn.execute(sql_s)
    time_measurements = []
    weight_measurements = []
    for row in cursor:
        time_measurements.append(row[0])
        weight_measurements.append(row[1])
    data_t = pd.Series(weight_measurements, index=time_measurements)

    return data_t


def get_foot_weight_raw(
    user_id: int,
    database_path: Optional[str] = config.DATABASE_PATH
) -> pd.Series:
    """
    Get the raw foot weight data from the database.

    :param user_id: an integer of user id
    :param database_path: a string of location of database
    :return: 1D pd Series mapping from timestamps to mmeasurements
    """
    sql_s = f"SELECT timestamp_ms, value FROM data WHERE data_capture_id={user_id} AND sensor_id=3"
    conn = sqlite3.connect(database_path)
    cursor = conn.execute(sql_s)
    time_measurements = []
    weight_measurements = []
    for row in cursor:
        time_measurements.append(row[0])
        weight_measurements.append(row[1])
    data_t = pd.Series(weight_measurements, index=time_measurements)

    return data_t


def get_seat_and_foot_weight_raw(
    user_id: int,
    database_path: Optional[str] = config.DATABASE_PATH
) -> List[pd.Series]:
    """
    Get the clean seat weight and foot weight data from the database.

    :param user_id: an integer of user id
    :param database_path: a string of location of database
    :return: a list that contains two 1D pd Series, seat and foot weight
    after being cleaned, mapping from timestamps to measurements
    """
    seat_raw = get_seat_weight_raw(user_id, database_path)
    foot_raw = get_foot_weight_raw(user_id, database_path)
    min_t = min(min(seat_raw.index), min(foot_raw.index))
    max_t = max(max(seat_raw.index), max(foot_raw.index))

    step_t = 500
    min_floor_t = int(np.floor(min_t/step_t)*step_t)
    max_ceil_t = int(np.ceil(max_t/step_t)*step_t)
    step1_d = {}
    step2_d = {}
    for i in range(min_floor_t, max_ceil_t+step_t, step_t):
        step1_d[i] = []
        step2_d[i] = []

    for i in range(len(seat_raw)):
        interval_t = int(np.floor(seat_raw.index[i]/step_t)*step_t)
        step1_d[interval_t].append(seat_raw.values[i])
    for i in range(len(foot_raw)):
        interval_t = int(np.floor(foot_raw.index[i]/step_t)*step_t)
        step2_d[interval_t].append(foot_raw.values[i])

    clean1_d = {}
    for i in step1_d.keys():
        clean1_d[i] = np.mean(step1_d[i])
    clean1_sz = pd.Series(clean1_d)

    clean2_d = {}
    for i in step2_d.keys():
        clean2_d[i] = np.mean(step2_d[i])
    clean2_sz = pd.Series(clean2_d)

    return clean1_sz, clean2_sz


def get_seat_and_foot_weight_clean(
    user_id: int,
    database_path: Optional[str] = config.DATABASE_PATH
) -> List[pd.Series]:
    """
    Get the clean seat and foot weight from the database.

    :param user_id: an integer of user id
    :param database_path: a string of location of database
    :return: two 1D pd Series mapping from
    timestamps (adjusted relative to the initial timestamp) to
    measurements of the clean seat weight and foot weight data
    (scale in kg)
    """
    water_distance_raw = load_water_distance.get_water_distance_raw(
        user_id, database_path)
    t0 = water_distance_raw.index[0]
    seat_raw, foot_raw = get_seat_and_foot_weight_raw(user_id, database_path)
    seat_clean = pd.Series(
        seat_raw.values / 1000,
        index=(seat_raw.index - t0) / 1000
    )

    foot_clean = pd.Series(
        foot_raw.values / 1000,
        index=(foot_raw.index - t0) / 1000
    )

    return seat_clean, foot_clean


def get_total_weight_clean(
    user_id: int,
    database_path: Optional[str] = config.DATABASE_PATH
) -> pd.Series:
    """
    Get the clean total weight data from the database.

    :param user_id: an integer of user id
    :param database_path: a string of location of database
    :return: 1D pd Series mapping from
    timestamps (adjusted relative to the initial timestamp) to
    measurements of the clean total weight data (scale in kg)
    """
    seat_clean, foot_clean = get_seat_and_foot_weight_clean(
        user_id, database_path)
    total_weight = seat_clean + foot_clean
    return total_weight
