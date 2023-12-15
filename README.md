# Evaluating The Trade-Off Between Performance And Carbon Emission In Wind Power Forecasting

This repository contains code for my MSc CS research project. The code focuses on a wind power dataset from the KDD CUP 2022, and it includes analysis of the 3rd place and 11th place of the competition.

## To work with the repository

1. Make two folders: models/KDDCUP2022_BERT and models/KDDCUP2022_GNN in the root where contents of the 3rd place and 11th place Githubs are placed.
2. Put the official competition dataset in a new folder data/ in the root

Note that the code from the BERT and GNN models are highly modified, for example by including during training Carbontracker and by forecasting only with a subset of the architecture of the GNN.

## Contents ##
- **preprocess.py** creates a new dataset with the preprocessing steps of the baseline ARIMA
- **data_analysis.py** contains several methods that produce figures for data analysis
- **evaluation.py** has several functionalities:
   - Fit an ARIMA baseline model for each 134 wind turbines on the train data while simutanously track the carbon
   - Get the RMSE score of the baseline, BERT or GNN model. The latter two may require modifying the contents of the original Githubs, so it does not work out of the box. 
