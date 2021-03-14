"""
make_dataset.py
Dataset Class:
    - RandomForest: cover basic sources of sensors
    - RandomForestExtened： add AudioEmbedding
    - Seq2Seq:
    - ManualAlg:
"""

import numpy as np
import pandas as pd
import torch
import config
from typing import Union

from src.data import load_annotation, load_audio
from src.data import load_radar, load_water_distance, load_weight_sensor


DataGetter = {
    'TotalWeight': load_weight_sensor.get_total_weight_clean,
    'WaterDistance': load_water_distance.get_water_distance_clean,
    'RadarSum': load_radar.get_radar_sum_clean,
    'AudioDelay4': load_audio.get_audio_delay4_clean
}

FeatureGetter = {
    'Max': lambda source: np.max(source.values),
    'Min': lambda source: np.min(source.values),
    'Mean': lambda source: np.mean(source.values),
    'Median': lambda source: np.median(source.values),
    'Variance': lambda source: np.var(source.values, ddof=1),
    'LogVariance': lambda source: np.log(np.var(source.values, ddof=1)),
    'LinearTrend': lambda source: np.polyfit(source.index, source.values, 1)[0]
}

# ========================
# Random Forest Dataset
# ========================


class RandomForestDataset:
    """
    A class to hold RandomForest dataset. Specifically, it is used
    to create the dataset for the random-forest approach. Sources of
    sensors cover the total weight scale, water distance, radar sum,
    and the audio delay version 4th, which corrsponds to the keys in
    `DataGetter` as well.

    :param user_ids: a list of integers for a list of user ids to
    generate data from.
    :param source_names: a list of strings for a list of source names
    to generate data from. Note that they should be chosen from the keys
    of `DataGetter`.
    :param category: a string of the event category, Urination or Defecation
    :param window_seconds: a float number of the window size in seconds. Window
    size determines the length of samples to collect features from.
    :param hop_seconds: a float number of the hop size in seconds. Hop size
    determines the distance between consecutive windows.
    :param feature_names: a list of strings for a list of feature names to
    generate data from. They should be chosen from the keys of `FeatureGetter`.
    """

    def __init__(self, dataset_config):
        self.user_ids = dataset_config['USER_IDS']
        self.source_names = dataset_config['SOURCE_NAMES']
        self.feature_names = dataset_config['FEATURE_NAMES']
        self.category = dataset_config['CATEGORY']
        self.window_seconds = dataset_config['WINDOW_SECONDS']
        self.hop_seconds = dataset_config['HOP_SECONDS']
        self.annotations = load_annotation.get_annotation(
            config.ANNOTATION_PATH)

    @staticmethod
    def get_framed_label(
        framed_timestamps: np.ndarray,
        annotated_timestamps: np.ndarray
    ) -> list:
        """
        Get the framed label based on the framed timestamps and annotated
        timestamps. Specifically, the annotation comes with the start and
        end time of an event. For each frame in the framed timestamps, i.e.,
        for each [ti, ti+window], we determine the label based on whether
        this framed window has overlap with the annotation start-end range.

        :param annotated_timestamps: 2D np.array which is a list of start-end
        pairs representing the annotation timestamps
        :param framed_timestamps: 2D np.array which is a list of [ti, ti+window]
        representing the framed timestamps
        :return: 1D list which has the same length as framed_timestamps,
        the label for each frame in the framed timestamps
        """
        res = [0] * len(framed_timestamps)
        for idx, framed_time in enumerate(framed_timestamps):
            for annotated_timestamp in annotated_timestamps:
                st, ed = annotated_timestamp
                framed_st, framed_ed = framed_time
                if framed_st >= st and framed_ed <= ed:
                    res[idx] = 1
                    break
        return res

    def get_framed_timestamps(self, user_id: int) -> np.ndarray:
        """
        For a user id, get the framed timestamps under the assigned
        window size and hop size. Typically it would look like
        np.array([ [t0, t0+window], [t1, t1+window], ... ])
        and t{j} - t{j-1} = hop
        """
        self.sources = {
            source_name: DataGetter[source_name](user_id)
            for source_name in self.source_names
        }
        st = max(source.index[0] for name, source in self.sources.items())
        ed = min(source.index[-1] for name, source in self.sources.items())
        framed_timestamps = []
        t = st
        while t <= ed - self.window_seconds:
            framed_timestamps.append([t, t+self.window_seconds])
            t += self.hop_seconds
        self.framed_timestamps = np.array(framed_timestamps)
        return self.framed_timestamps

    def get_label_from_one_user(self, user_id: int) -> np.ndarray:
        """
        Get the label for user_id
        Return a 1D np.array with shape (NUM_FRAMES, )
        """
        framed_timestamps = self.get_framed_timestamps(user_id)
        annotated_timestamps = [
            [int(annotated[0]), int(annotated[1])]
            for annotated in self.annotations.get(user_id, [])
            if annotated[2] == self.category
        ]
        return self.get_framed_label(
            framed_timestamps=framed_timestamps,
            annotated_timestamps=annotated_timestamps
        )

    def get_feature_from_one_user(self, user_id: int) -> np.ndarray:
        """
        Get the feature for user_id
        Return a 2D np.array with shape (NUM_FRAMES, NUM_FEATURES)
        """
        feature = []
        for framed_timestamp in self.framed_timestamps:
            feature_current = []
            for source_name in self.source_names:
                source = self.sources[source_name]
                source_current = source[
                    (source.index >= framed_timestamp[0]) &
                    (source.index <= framed_timestamp[1])
                ]
                feature_current += [
                    FeatureGetter[feature_name](source_current)
                    for feature_name in self.feature_names
                ]
            feature.append(feature_current)
        return np.array(feature)

    def get_features_and_labels_from_users(self):
        """
        Get the features and labels from all users in the self.user_ids
        Return a 2D np.array features and 1D np.array labels
        features shape = (SUM_NUM_FRAMES, NUM_FEATURES)
        labels shape = (SUM_NUM_FRAMES, )
        """
        labels = []
        first = True
        for user_id in self.user_ids:
            print(f"updating {user_id}")
            labels += self.get_label_from_one_user(user_id)
            if first:
                features = self.get_feature_from_one_user(user_id)
                first = False
            else:
                features = np.r_[
                    features, self.get_feature_from_one_user(user_id)]
        colnames = [
            source_name + "_" + feature_name
            for source_name in self.source_names
            for feature_name in self.feature_names
        ]
        labels = np.array(labels)
        features = pd.DataFrame(features, columns=colnames)
        return features, labels


# ===========================================
# Random Forest Dataset With AudioEmbedding
# ===========================================

class RandomForestExtended(RandomForestDataset):
    """
    A child of RandomForestDataset with new source, the audio
    embedding added to the original dataset. Specifically,
    the audio embedding will be added by default.
    """

    def __init__(self, dataset_config):
        super().__init__(dataset_config)

    def get_framed_timestamps(self, user_id: int) -> np.ndarray:
        """
        Get the framed timestamps based on the source timestamps and
        the window size and hop size config. It is similar to the
        method in the parent class. However, this extended method
        includes the audio embedding.

        :param user_id: int
        :return: a 2D np.array composed of a list of framed windows.
        """
        self.sources = {
            source_name: DataGetter[source_name](user_id)
            for source_name in self.source_names
        }

        st = max(source.index[0] for name, source in self.sources.items())
        ed = min(source.index[-1] for name, source in self.sources.items())

        audio = load_audio.Audio(user_id)
        self.audio_embedding = audio.get_vggish_embedding()
        audio_timestamps = np.array(self.audio_embedding.index)
        st = max(st, audio_timestamps[0])
        ed = min(ed, audio_timestamps[-1])

        st_window_example = int(np.ceil(st/0.96))
        ed_window_example = int(np.floor(ed/0.96))
        window_num_audio_examples = int(self.window_seconds/0.96)
        hop_num_audio_examples = int(self.hop_seconds/0.96)

        self.framed_timestamps = [
            [i*0.96, (i + window_num_audio_examples)*0.96]
            for i in range(
                st_window_example,
                ed_window_example - window_num_audio_examples,
                hop_num_audio_examples
            )
        ]

        return self.framed_timestamps

    def get_feature_from_one_user(self, user_id: int) -> np.ndarray:
        """
        Get the feature for user user_id.

        :param: user_id: an integer for user_id
        :return: a 2D np.array of feature for user_id. The shape
        is (NUM_FRAMES, NUM_FEATURES).
        """
        feature = []
        try:
            for framed_timestamp in self.framed_timestamps:
                feature_current = []
                for source_name in self.source_names:
                    source = self.sources[source_name]
                    source_current = source[
                        (source.index >= framed_timestamp[0]) &
                        (source.index <= framed_timestamp[1])
                    ]

                    feature_current += [
                        FeatureGetter[feature_name](source_current)
                        for feature_name in self.feature_names
                    ]

                audio_embedding_current = self.audio_embedding[
                    (self.audio_embedding.index >= framed_timestamp[0]) &
                    (self.audio_embedding.index <= framed_timestamp[1])
                ]

                feature_current += list(np.array(audio_embedding_current).flatten())
                feature.append(feature_current)
        except:
            print(f"updating {user_id} failed")
        return np.array(feature)

    def get_features_and_labels_from_users(self) -> Union[pd.DataFrame, np.ndarray]:
        """
        Get features and labels for all the users in self.user_ids.

        :return: a union of 2D pd.DataFrame for features and 1D np.array
        for labels of all the users in user_ids.
        features shape = (SUM_NUM_FRAMES, NUM_FEATURES)
        labels shape = (SUM_NUM_FRAMES, )
        """
        labels = []
        first = True
        for user_id in self.user_ids:
            print(f"updating {user_id}")
            labels += self.get_label_from_one_user(user_id)
            if first:
                features = self.get_feature_from_one_user(user_id)
                first = False
            else:
                try:
                    features = np.r_[
                        features, self.get_feature_from_one_user(user_id)]
                except:
                    print(f"appending {user_id} failed")

        colnames = [
            source_name + "_" + feature_name
            for source_name in self.source_names
            for feature_name in self.feature_names
        ] + ["AudioEmbedding_" + str(i) for i in range(
            features.shape[1] - len(self.source_names)*len(self.feature_names)
        )]

        labels = np.array(labels)
        features = pd.DataFrame(features, columns=colnames)

        return features, labels
