"""
config.py
environment variables
"""

PROJECT_PATH = "C:/Users/Jiajun/Desktop/download-project"

# sensor data
DATAFRAME_PATH = f"{PROJECT_PATH}/data/raw/data_frames"
ANNOTATION_PATH = f"{PROJECT_PATH}/data/processed/Annotation.csv"
DATABASE_PATH = f"{PROJECT_PATH}/data/raw/toilet.db"
AUDIO_SAMPLING_RATE = 48000

# pretrained vggish for audio embeddings
PRETRAINED_VGGISH_PATH = f"{PROJECT_PATH}/pytorch_vggish.pth"

# dataset
DATASET_CONFIG = {
    'Urination': {
        'USER_IDS': [],  # placeholder for user ids
        'FEATURE_NAMES': ['Max', 'Min', 'Mean', 'Median', 'LogVariance', 'LinearTrend'],
        'SOURCE_NAMES': ['TotalWeight', 'RadarSum', 'AudioDelay4'],
        'WINDOW_SECONDS': 2,
        'HOP_SECONDS': 1,
        'CATEGORY': 'Urination'
    },

    'Defecation': {
        'USER_IDS': [],  # placeholder for user ids
        'FEATURE_NAMES': ['Max', 'Min', 'Mean', 'Median', 'LogVariance', 'LinearTrend'],
        'SOURCE_NAMES': ['TotalWeight', 'RadarSum', 'AudioDelay4'],
        'WINDOW_SECONDS': 2,
        'HOP_SECONDS': 1,
        'CATEGORY': 'Defecation'
    }
}

# evaluation
EVAL_CONFIG = {
    'USER_IDS': [],
    'U_TRAINED_MODEL_PATH': f"{PROJECT_PATH}/models/urination-rf-extended-embedding-2021-05-11.pkl",
    'D_TRAINED_MODEL_PATH': f"{PROJECT_PATH}/models/defecation-rf-extended-embedding-2021-05-11.pkl",
    'U_TRAIN_CONFIG': DATASET_CONFIG['Urination'],
    'D_TRAIN_CONFIG': DATASET_CONFIG['Defecation'],
    'U_THRESHOLD': 0.3,
    'D_THRESHOLD': 0.3,
}

# report
SAVE_PLOT_ALL_SENSORS_PATH = f"{PROJECT_PATH}/report/figures/PlotUses"
