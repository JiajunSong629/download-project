# download-project

## Project Organization

    ├── LICENSE
    |
    ├── models                   <- Trained models
    │      
    ├── notebooks                <- Jupyter notebooks
    │      
    ├── requirements.txt         <- Dependencies
    │      
    ├── src                      <- Source code
    │   │      
    │   ├── data/                <- Scripts to load the sensor data, including audio,
    |   |                           radar, water distance, and load cell.
    │   |
    |   |── make_dataset.py      <- Generate the dataset.
    |   |
    |   |—— run_randomforest.py  <- Run the random forest model on the
    |   |                           generated dataset, and save the model.
    |   |
    |   |—— post_analysis.py     <- Calculate the performance metrics and plot
    |                               the evaluation figures based on the trained
    |                               model.


## Get started

```{bash}
# create virtualenv and add dependencies
python -m venv <virtual_env>
source <virtual_env>/bin/activate
pip3 install -r requirements.txt
```
