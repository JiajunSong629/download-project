"""
load_water_distance.py
Get the water distance raw and clean from the database.
"""

import os
import pandas as pd
import sqlite3
from typing import Optional
import config


def get_water_distance_raw(
    user_id: int,
    database_path: Optional[str] = config.DATABASE_PATH
) -> pd.Series:
    """
    Get the raw water distance data for a user id from the database,
    and return a Series mapping from the time stamp to distance measurement

    :param user_id: an integer of the user id
    :param database_path: a string of the location of the database
    :return: a pandas Series mapping from the time stamp to distance
    measurement
    """
    sql_s = f"SELECT timestamp_ms, value FROM data WHERE data_capture_id={user_id} AND sensor_id=1"
    conn = sqlite3.connect(database_path)
    cursor = conn.execute(sql_s)
    time_measurements = []
    distance_measurements = []
    for entry in cursor:
        time_measurements.append(entry[0])
        distance_measurements.append(entry[1])
    data_r = pd.Series(distance_measurements, index=time_measurements)

    return data_r


def get_water_distance_clean(
    user_id: int,
    database_path: Optional[str] = config.DATABASE_PATH
) -> pd.Series:
    """
    Clean the water distance data for a user id from the database. Adjust the
    timestamp based on the first timestamp (so that starts from 0 and the
    timestamp is on second scale) and rescale the measurement.

    :param user_id: an integer of the user id
    :param database_path: a string of the location of the database
    :return: a pandas Series mapping from the time stamp to distance
    measurement   
    """
    data_r = get_water_distance_raw(user_id, database_path)
    t0 = data_r.index[0]
    data_c = pd.Series(
        100 * data_r.values,
        index=(data_r.index - t0) / 1000
    )

    return data_c
