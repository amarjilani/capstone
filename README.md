# Capstone Project - Predicting Longitudinal Cognition in Alzheimer's Disease using Deep Learning 
#### Author: Amar Jilani (University of California, Berkeley)
#### In collaboration with: Daren Ma (Raj Lab, UCSF), Ashish Raj (Raj Lab, UCSF)

### Abstract 

This project introduces a novel approach to Alzheimer's disease prognosis by developing a Long Short-Term Memory (LSTM) model to longitudinally predict protein progression and baseline cognitive scores using neuroimaging data. Addressing the limitations of current diagnostic methods, which are largely retrospective and invasive, this project leverages PET-derived data for its non-invasive nature and detailed biomarker insights. Diverging from traditional classification-focused machine learning research, this project emphasizes regression tasks to forecast clinical measures critical to Alzheimer's progression and treatment. By targeting the prediction of cognitive decline and biomarker evolution, the model aims to facilitate earlier diagnosis, enable personalized treatment strategies, and contribute to the development of new therapeutic interventions. This work responds to the challenges of existing machine learning tools in handling the complex dimensionality mismatch of neuroimaging and biomarker data, emphasizing the need for a custom, hybrid deep learning approach to improve predictive accuracy in Alzheimer's disease progression. This project yielded a model that demonstrates promising results in accurately forecasting the progression of tau protein concentrations across the brain and cognitive metrics.

### Description 
This repository contains the files needed to run and train the LSTM model for regional tau and cognitive score predictions.

#### Directories 
- `data/` - This directory contains the data files required to train the model and run the training notebooks. 
  - `demo.csv` - This contains demographic data for patients which can be merged using the patient IDs. 
  - `endm_data.mat` - This is the primary data source used to train the LSTM model. It contains the timeseries regional tau values for 196 patients. This data is generated from the eNDM model. 
  - `Lap.mat` - This file contains eigenvectors and eigenvalues of the Laplacian, which is used for dimensionality reduction. 
  - `pre_split.csv` - This file contains the preprocessed endm_data.mat data before being split into smaller sequences.
  - `Tau_with_Demographics.csv` - A larger dataset containing baseline tau and demographic data for 819 patients. Used for training the benchmark model. 
  - `transformed_data.csv` - Complete processed dataset used for training the LSTM model.
  - `transformed_test_data.csv` - Complete processed dataset used for evaluting the LSTM model. 
- `models/` - This directory stores the parameters for the models as they are being trained. This directory currently only includes the best performing lstm model from my own training. This corresponds to the model that had the lowest validation losses. 
- `training/` - This directory contains the training notebooks. 
  - `training_bench_full.ipynb` - Notebook consisting of the data processing and training process for the benchmark model. This version is trained on the larger dataset (`Tau_with_Demographics.csv`).
  - `training_bench_small.ipynb` - Notebook consisting of the data processing and training procesz for the benchmark model. This version is trained on the smallest dataset (same patient ids as the LSTM model). 
  - `training_final.ipynb` - Notebook consisting of data processing and training process for the final LSTM model (LSTMwithStatic). 

#### Files
- `model.py` - This file contains all models that were implemented over the course of this project. Most notably it contains `LSTMwithStatic`, which is the final working iteration of the LSTM models. It also contains `SimpleMLP`, which is the benchmark model. 
- `README.MD` 
- `LICENSE`
- `Analysis.pdf` - Detailed report on the model implementation and performance. 
