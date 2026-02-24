# Give Me Some Credit – Default Prediction

## Project Overview
This repository contains a complete machine learning pipeline for predicting financial distress within two years. This project was developed as part of our 4th-semester B.Tech AIML training curriculum. The goal is to build an algorithm that can predict the probability that somebody will experience financial distress in the next two years.

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
The data originates from the Kaggle competition: [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit). It includes historical data on borrowers (e.g., monthly income, number of dependents, age).

## Project Workflow
1. **Data Loading & Cleaning:** Ensuring target variables are intact and removing empty records.
2. **Exploratory Data Analysis (EDA):** Visualizing class imbalances and identifying missing features.
3. **Data Preprocessing:** Imputing missing values using the median strategy.
4. **Model Building:** Training an XGBoost Classifier customized for highly imbalanced tabular data.
5. **Comprehensive Evaluation:** Validating the model using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and a Confusion Matrix.
