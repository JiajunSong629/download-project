"""
post_analysis.py

Classes and methods for the post analysis after model
training.

    - WeightChangeCalculator: interface for calculating the weight
    change
    - EvalPlot: for a user id, plot the total weight with user-annotation
    and predicted event intervals of both urinate and defecate cases. On top
    of the graph also shows the precision and recall rate.
- 
"""

from typing import Callable, Optional, List
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import librosa.display
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score, precision_score


import config
from src.utils import get_double_iqr
from src.utils import apply_median_filter, apply_double_median_filter
from src.data import load_weight_sensor
from src.utils import get_framed_label, train_test_split, from_boolean_array_to_intervals
from src.data import load_annotation
from src.data import load_radar, load_water_distance, load_weight_sensor, load_audio
from src import make_dataset


class WeightChangeCalculator:
    def __init__(self, user_id):
        self.user_id = user_id
        self.total_weight_clean = load_weight_sensor.get_total_weight_clean(
            user_id)

    def get_total_weight_smoothed(
        self,
        smooth_method: Optional[Callable] = apply_median_filter,
        **kwargs
    ) -> pd.Series:
        """
        Smooth the total weight data with the assigned method.
        """
        return smooth_method(self.total_weight_clean, **kwargs)

    def get_weight_change(
        self,
        start_stop_list: List[List[float]],
        smooth_method: Optional[Callable] = apply_double_median_filter,
        diff_method: Optional[Callable] = get_double_iqr,
        **kwargs
    ) -> float:
        """
        Get the weight change during a list of [start, stop]
        """
        total_weight_smooth = self.get_total_weight_smoothed(
            smooth_method, **kwargs)

        res = 0
        for start_stop in start_stop_list:
            start, stop = start_stop
            total_weight_within = total_weight_smooth[
                (total_weight_smooth.index >= start) &
                (total_weight_smooth.index <= stop)
            ]

            res += diff_method(total_weight_within.values)

        return res


class EvalPlot:
    def __init__(self, eval_config):
        self.user_ids = eval_config['USER_IDS']
        self.u_train_config = eval_config['U_TRAIN_CONFIG']
        self.d_train_config = eval_config['D_TRAIN_CONFIG']
        self.u_threshold = eval_config['U_THRESHOLD']
        self.d_threshold = eval_config['D_THRESHOLD']

        with open(eval_config['U_TRAINED_MODEL_PATH'], "rb") as f:
            self.u_trained_model = pickle.load(f)
        with open(eval_config['D_TRAINED_MODEL_PATH'], "rb") as f:
            self.d_trained_model = pickle.load(f)
        self.annotations = load_annotation.get_annotation()

    def get_x_and_y(self, user_id, category):
        if category == "Urination":
            config = self.u_train_config.copy()
        elif category == "Defecation":
            config = self.d_train_config.copy()
        config['USER_IDS'] = [user_id]
        dataset_i = make_dataset.RandomForestExtended(config)
        x_i, y_i = dataset_i.get_features_and_labels_from_users()
        return x_i, y_i

    def get_predicted_region(self, user_id, category):
        if category == "Urination":
            model = self.u_trained_model
            threshold = self.u_threshold
        elif category == "Defecation":
            model = self.d_trained_model
            threshold = self.d_threshold
        self.x_i, self.y_i = self.get_x_and_y(user_id, category)
        self.boolean_array_i = (model.predict_proba(
            self.x_i)[:, 1] > threshold).astype(int)
        predicted_region = from_boolean_array_to_intervals(
            self.boolean_array_i)
        return predicted_region

    def get_annotated_region(self, user_id, category):
        annotated_region = []
        for region in self.annotations[user_id]:
            if region[-1] == category:
                annotated_region.append(region[:2])
        return annotated_region

    def get_eval_statistics(self):
        recall_ = recall_score(y_true=self.y_i, y_pred=self.boolean_array_i)
        precision_ = precision_score(
            y_true=self.y_i, y_pred=self.boolean_array_i)
        return recall_, precision_

    def make_subplot_1(self, ax, user_id, caption):
        total_weight_i = load_weight_sensor.get_total_weight_clean(user_id)
        ax.plot(total_weight_i)
        ax.set_ylim(total_weight_i.median()-1, total_weight_i.median()+1)
        ax.title.set_text(
            f'annotated: {user_id} Urine recall: {caption[0]: .2f}, precision: {caption[1]: .2f}')
        for region in self.urinate_annotated_region:
            ax.axvspan(region[0], region[1], color="gold", alpha=0.8)
        for region in self.defecate_annotated_region:
            ax.axvspan(region[0], region[1], color="red", alpha=0.5)

    def make_subplot_2(self, ax, user_id, caption):
        total_weight_i = load_weight_sensor.get_total_weight_clean(user_id)
        ax.plot(total_weight_i)
        ax.title.set_text(
            f'predicted: {user_id}; Stool recall:{caption[0]:.2f}, precision:{caption[1]:.2f}')
        ax.set_ylim(total_weight_i.median()-1, total_weight_i.median()+1)
        for region in self.urinate_predicted_region:
            ax.axvspan(region[0], region[1], color="gold", alpha=0.8)
        for region in self.defecate_predicted_region:
            ax.axvspan(region[0], region[1], color="red", alpha=0.5)

    def plots(self, filename):
        nrow = len(self.user_ids)
        ncol = 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(10*ncol, 2*nrow))
        for idx, user_id in enumerate(self.user_ids):
            self.urinate_annotated_region = self.get_annotated_region(
                user_id, "Urination")
            self.urinate_predicted_region = self.get_predicted_region(
                user_id, "Urination")
            self.urinate_recall, self.urinate_precision = self.get_eval_statistics()

            self.defecate_annotated_region = self.get_annotated_region(
                user_id, "Defecation")
            self.defecate_predicted_region = self.get_predicted_region(
                user_id, "Defecation")
            self.defecate_recall, self.defecate_precision = self.get_eval_statistics()

            # axes[idx][0]: total_weight with annotations
            self.make_subplot_1(axes[idx][0], user_id, [
                                self.urinate_recall, self.urinate_precision])
            # axes[idx][1]: total_weight with prediction
            self.make_subplot_2(axes[idx][1], user_id, [
                                self.defecate_recall, self.defecate_precision])

        fig.tight_layout()
        plt.savefig(f'../reports/{filename}')
