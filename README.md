download-project
==============================

Project Organization
------------

    ├── LICENSE
    |
    ├── models             <- Trained models
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── requirements.txt   <- Dependencies
    │
    ├── src                <- Source code
    │   │
    │   ├── data/           <- Scripts to load the sensor data, including audio,
    |   |                    radar, water distance, and load cell.
    │   |
    |   |── make_dataset.py <- Generate the dataset.
    |   |
    |   |—— run_randomforest.py <- Run the random forest model on the
    |   |                    generated dataset, and save the model.
    |   |
    |   |—— post_analysis.py <- Calculate the performance metrics based
    |   |                    on the trained models.
    |
    |______________________
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. </small></p>
