import os
from csv import reader
from collections import defaultdict, namedtuple
from typing import DefaultDict, Optional, List
import numpy as np
import pandas as pd
import config

Annotated_Event = namedtuple('Annotated_Event', ['start', 'stop', 'event'])


def get_annotation(
    annotation_path: Optional[str] = config.ANNOTATION_PATH
) -> DefaultDict[int, List[Annotated_Event]]:
    """
    Read the annotation excel file.

    :param annotation_path: string of the location of the annotation file
    :return: dictionary with key --> value pair of
        user id --> a list of annotations with user id in the form of
        [start timestamp, end timestamp, event category]
    """
    with open(annotation_path, 'r') as read_obj:
        entries = reader(read_obj)
        annotations = defaultdict(list)
        for entry in entries:
            try:
                user_id = int(entry[0])  # avoid empty entry of annotation
            except:
                continue

            start_time_raw = entry[1].split(":")
            if len(start_time_raw) == 1:
                break
            start_time_s = float(
                start_time_raw[0]) * 60 + float(start_time_raw[1])

            stop_time_raw = entry[2].split(":")
            stop_time_s = float(
                stop_time_raw[0]) * 60 + float(stop_time_raw[1])

            if start_time_s == stop_time_s:
                #start_time_s = start_time_s - 1
                #stop_time_s = stop_time_s + 1
                continue

            event = entry[3]
            annotated_event = Annotated_Event(
                start=start_time_s,
                stop=stop_time_s,
                event=event
            )
            annotations[user_id].append(annotated_event)

    return annotations


def get_complete_ids(
    category: str,
    annotation_path: Optional[str] = config.ANNOTATION_PATH
) -> np.ndarray:
    """
    Read the annotation excel and return the unique user ids corresponds to
    event category.

    :param annotation_path: string of location of the annotation file
    :param category: string of the event category, whether Urination or
        Defecation
    :return: 1D np.array of unique user ids that are annotated in the
        assigned event category
    """
    assert category in ('Urination', 'Defecation')
    annotation_df = pd.read_csv(annotation_path)
    user_ids = annotation_df[annotation_df.Event == category].iloc[:, 0]
    return user_ids.drop_duplicates().values
