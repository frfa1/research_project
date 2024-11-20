# Evaluating The Trade-Off Between Performance And Carbon Emission In Wind Power Forecasting

This repository contains code for my MSc CS research project. The code focuses on a wind power dataset from the KDD CUP 2022, and it includes analysis of the 3rd place and 11th place of the competition. It is thus highly reliant of those repositories, which must be placed in the root folder.

The goal of the project was to evaluate how much carbon the state-of-the-art deep learning models emit during training, and how this relates to the forecasting performance improvements that they bring. The three models compared are ARIMA (autoregressive integrated moving average, the baseline), a transformer architecture and a graph neural network. For more details, [read the report](https://github.com/frfa1/research_project/blob/main/report.pdf).

## To work with the repository

1. Make two folders: models/KDDCUP2022_BERT and models/KDDCUP2022_GNN in the root where contents of the 3rd place and 11th place Githubs are placed.
2. Put the official competition dataset in a new folder data/ in the root and in models/KDDCUP2022_BERT/data/raw

Note that the code from the BERT and GNN models are highly modified. For example, Carbontracker is including during training of each model, and the GNN trains and forecasts only one of the two GNN architectures that they propose (the MGTNN).

## Contents ##
- **preprocess.py** creates a new dataset with the preprocessing steps of the baseline ARIMA
- **data_analysis.py** contains several methods that produce figures for data analysis
- **evaluation.py** has several functionalities:
   - Fit an ARIMA baseline model for each 134 wind turbines on the train data while simutanously track the carbon
   - Get the RMSE score of the baseline, BERT or GNN model. The latter two may require modifying the contents of the original Githubs, so it does not work out of the box. 
