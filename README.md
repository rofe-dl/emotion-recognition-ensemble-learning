# Emotion Recognition from Speech & Text using Ensemble Learning

Code for our thesis (CSE400) where we tried multimodal emotion recognition from speech and text features using an ensemble of different classifiers on the IEMOCAP dataset. For that, we tried different hetergoneneous ensemble learning techniques to compare and find the best ensemble. We covered the following techniques:

* Hard Voting
* Soft Voting
* Stacking
* Blending

We have 6 models trained on speech data, and 6 models trained on text data. The findings of all 12 models are combined using the ensemble methods. The results are found in `main.ipynb`.

## Setup Instructions

1. Clone the repository

1. `cd` to the codebase and create a virtual environment and activate it.
    ```
    cd emotion-recognition-ensemble-learning
    ```
    * For Linux
    ```
    python3 -m venv env
    source env/bin/activate
    ```
    * For Windows
    ```
    python -m venv env
    env\Scripts\activate.bat
    ```
1. Install necessary libraries
    ```
    pip install -r requirements.txt
    pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl # (OPTIONAL) depends on your python version (37, 38 or 39)
    ```
1. Download the IEMOCAP dataset by submitting a request from [here](https://sail.usc.edu/iemocap/iemocap_release.htm). Will take 1-3 days for them to email you.
1. Make a folder named `data` in the project directory and put the dataset there. Rename the folder to `IEMOCAP_dataset`
1. Process the dataset and extract features into the data folder
    ```
    python3 -m process_dataset.speech_features
    python3 -m process_dataset.text_features
    ```
1. Run main.ipynb

## Individual Models

To test out the performance of each model individually, just run their respective file as a module. For example
```
python3 -m speech_models.speech_logistic_regression
```

---

IEMOCAP metadata was obtained from [here](https://www.kaggle.com/datasets/samuelsamsudinng/iemocap-emotion-speech-database).