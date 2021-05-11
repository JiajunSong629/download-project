"""
load_radar.py
Get raw radar data and clean radar sum from the database.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.data import load_water_distance
from src.utils import apply_median_filter
from scipy.integrate import simps, trapz
import config


def get_radar_raw(
    user_id: int,
    dataframe_path: Optional[str] = config.DATAFRAME_PATH
) -> pd.DataFrame:
    """
    Get the raw radar data from the dataframe.

    :param user_id: an integer of the user id
    :param dataframe: a string of the location of the dataframe where radar
    data is stored.
    :return: 2D pd DataFrame of the radar data mapping from the time stamp
    to the multidimensional radar data
    """
    data_fn = f"{dataframe_path}/data_capture_{user_id}/radar_data.txt"
    data_f = open(data_fn, 'rt')
    line_s = data_f.read()
    data_l = eval(line_s)
    t0_sz = pd.Series(data_l[0]['data'])
    data_d = {}
    for j in range(len(data_l)):
        t = data_l[j]['timestamp_ms']
        j_sz = pd.Series(data_l[j]['data'][0])
        data_d[t] = j_sz
    data_df = pd.DataFrame(data_d)

    return data_df


def get_radar_sum_clean(
    user_id: int,
    database_path: Optional[str] = config.DATABASE_PATH,
    dataframe_path: Optional[str] = config.DATAFRAME_PATH
) -> pd.Series:
    """
    Get the cleaned radar sum data for a user id. Radar sum is the sum
    of squares of the radar data at one timestamp.

    :param user_id: an integer of user id
    :param database_path: a string of the location of database
    :param dataframe_path: a string of the location of dataframe

    :return: 1D pd Series mapping from the timestamp to the sum of radar data
    """
    water_distance_raw = load_water_distance.get_water_distance_raw(user_id)
    t0 = water_distance_raw.index[0]
    radar_df = get_radar_raw(user_id)
    area_d = {}
    floor_i = 50
    ceil_i = 200
    for i in radar_df.columns:
        sq_sz = (radar_df[i])**2
        area_d[i] = sum(sq_sz.iloc[floor_i:ceil_i])
    area_sz = pd.Series(area_d)
    area_sz.index = (area_sz.index-t0)/1000

    return area_sz / 1e9


def get_area_under_radarsum(
    user_id: int,
    database_path: Optional[str] = config.DATABASE_PATH,
    dataframe_path: Optional[str] = config.DATAFRAME_PATH,
    smooth: Optional[bool] = True,
    **kwargs
) -> float:
    """
    Get the area under the radar_sum curve.

    :param user_id: an integer of user id
    :param database_path: a string of the location of database
    :param dataframe_path: a string of the location of dataframe
    :param smooth: a boolean for whether to smooth the radar sum series
    :return: a float number of the area under the radar sum curve.
    """
    radar_sum = get_radar_sum_clean(user_id, database_path, dataframe_path)
    if smooth:
        radar_sum = apply_median_filter(radar_sum, **kwargs)
    x = np.array(radar_sum.index)
    f = radar_sum.values

    return trapz(f, x)
