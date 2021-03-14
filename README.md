download-project
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Download Project


To-do

- [x] .env environment variable
- [x] fix the file structure if any falls apart
    - [x] data
    - [x] visualization
    - [x] notebooks
    - [x] models
- [] fix code
    - [x] change pa to pd
    - [x] fix the dotenv in each script
    - [] deal with the pytorch vggish case
    - [x] type annotation for each script

    - [x] data
        - [x] load_annotation
        - [x] load_radar
        - [x] load_water_distance
        - [x] load_weight_sensor
        - [x] load_audio
        - [x] load_audio Audio Class
    - [] visualization
        - [] GetSensors
        - [] Thermal
        - [] GetEvent
    - [] make_dataset
        - [x] RandomForest
        - [x] RandomForestExtended
        - [] Seq2Seq (in src/archive/pick_up_later.py)
        - [] Manual Algorithm (in src/archive/pick_up_later.py)
    - [] post_analysis

- [] rename notebooks
- [] modify any possible failure might cause in the notebook
- [] logging functionality
- [] namedtuple to enhance readability