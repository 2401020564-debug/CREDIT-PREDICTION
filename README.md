# CREDIT-PREDICTION
AIML training project: Give Me Some Credit default prediction model.
# Give Me Some Credit – Default Prediction

## Project Overview
This repository contains the machine learning pipeline for predicting financial distress within two years. This project was developed as part of our AIML training curriculum.

## Team
* Group - 5
* Name - The Debuggers
## Team Members
* Anuj Prasad (2401020538)
* Raj Aryan (2401020564)
* Rishu Kumar (2401020566)
* Sanjeet Samal (2401020514)
* Nitin Pandey (2401020558)
* A V Sai Subham Naidu (2401020541)

## Dataset
The dataset is from the Kaggle competition: [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit). It includes historical data on borrowers to predict the probability of default (`SeriousDlqin2yrs`).

## Workflow
1. **Exploratory Data Analysis (EDA):** Understanding data distributions and handling missing values.
2. **Data Preprocessing:** Imputation of missing values (`MonthlyIncome` and `NumberOfDependents`).
3. **Model Building:** Training an XGBoost Classifier for high-performance tabular data prediction.
4. **Evaluation:** Assessing the model using the ROC-AUC metric.
