"""
run_randomforst.py
"""

import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
import pickle
from datetime import date


from src.utils import get_framed_label, train_test_split
from src.data import load_annotation
from src.data import load_radar, load_water_distance, load_weight_sensor, load_audio
from src import make_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_roc_curve


# ===================
# Urination
# ==================

CATEGORY = "Urination"

config = {
    'USER_IDS': [],  # placeholder for user ids
    'FEATURE_NAMES': ['Max', 'Min', 'Mean', 'Median', 'LogVariance', 'LinearTrend'],
    'SOURCE_NAMES': ['TotalWeight', 'RadarSum', 'AudioDelay4'],
    'WINDOW_SECONDS': 2,
    'HOP_SECONDS': 1,
    'CATEGORY': CATEGORY
}

complete_ids = load_annotation.get_complete_ids(
    category=CATEGORY
)

selected_ids = complete_ids[:60]
TRAIN_IDS, TEST_IDS = train_test_split(selected_ids, seed=1234)

print(f"Category: {CATEGORY}")
print(f"Training {len(TRAIN_IDS)} use_ids: {TRAIN_IDS[:5]}...")
print(f"Testing  {len(TEST_IDS)} use_ids: {TEST_IDS[:5]}...")


train_config = config.copy()
test_config = config.copy()

train_config['USER_IDS'] = TRAIN_IDS
test_config['USER_IDS'] = TEST_IDS

dataset = {}
dataset['train'] = make_dataset.RandomForestExtended(train_config)
dataset['test'] = make_dataset.RandomForestExtended(test_config)


# it may take around 0.5 hr to run
train_x, train_y = dataset['train'].get_features_and_labels_from_users()
test_x, test_y = dataset['test'].get_features_and_labels_from_users()


print(f"train_x.shape = {train_x.shape}, test_x.shape = {test_x.shape}")
print(f"#positive/#total train_y = {sum(train_y)}/{len(train_y)}")
print(f"#positive/#total test_y = {sum(test_y)}/{len(test_y)}")

rf = RandomForestClassifier(n_estimators=30)
rf.fit(train_x, train_y)

current_time = date.today().strftime("%Y-%m-%d")
model_name = f"../models/{CATEGORY}-rf-extended-embedding-{current_time}.pkl"

with open(model_name, "wb") as f:
    pickle.dump(rf, f)


# ===================
# Defecation
# ==================

CATEGORY = "Defecation"

config = {
    'USER_IDS': [],
    'FEATURE_NAMES': ['Max', 'Min', 'Mean', 'Median', 'LogVariance', 'LinearTrend'],
    'SOURCE_NAMES': ['TotalWeight', 'RadarSum', 'AudioDelay4'],
    'WINDOW_SECONDS': 2,
    'HOP_SECONDS': 1,
    'CATEGORY': CATEGORY
}

complete_ids = load_annotation.get_complete_ids(
    category=CATEGORY
)

selected_ids = [idx for idx in complete_ids if idx <= 1950 and idx >= 1800]
TRAIN_IDS, TEST_IDS = train_test_split(selected_ids)

print(f"Category: {config['CATEGORY']}")
print(f"Training {len(TRAIN_IDS)} use_ids: {TRAIN_IDS[:5]}...")
print(f"Testing  {len(TEST_IDS)} use_ids: {TEST_IDS[:5]}...")


train_config = config.copy()
test_config = config.copy()

train_config['USER_IDS'] = TRAIN_IDS
test_config['USER_IDS'] = TEST_IDS

dataset = {}
dataset['train'] = make_dataset.RandomForestExtended(train_config)
dataset['test'] = make_dataset.RandomForestExtended(test_config)

train_x, train_y = dataset['train'].get_features_and_labels_from_users()
test_x, test_y = dataset['test'].get_features_and_labels_from_users()

print(f'train_x.shape: {train_x.shape} test_x.shape: {test_x.shape}')
print(f'No. Positive in training {train_y.sum()}/{train_y.shape}')
print(f'No. Positive in testing  {test_y.sum()}/{test_y.shape}')

rf = RandomForestClassifier(
    n_estimators=10,
    class_weight="balanced"
)
rf.fit(train_x, train_y)

current_time = date.today().strftime("%Y-%m-%d")
model_name = f"../models/{CATEGORY}-rf-extended-embedding-{current_time}.pkl"

with open(model_name, "wb") as f:
    pickle.dump(rf, f)
