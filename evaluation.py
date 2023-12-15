import os, sys, argparse
#import tensorflow as tf
import numpy as np

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from preprocess import read_data

sys.path.insert(0, '../models/') 

from carbontracker.tracker import CarbonTracker

def build_train_data():
    folder = "../data/"
    file = "wtbdata_245days.csv"
    file2 = "sdwpf_baidukddcup2022_turb_location.CSV"

    #preprocessed_train_data = "wtbdata_245days.csv" # Temporary
    preprocessed_train_data = "preprocessed_train_data.csv"
    #preprocessed_train_data = "filled_data_short.csv"

    df = pd.read_csv(folder + preprocessed_train_data)

    train = df[df["Day"] <= 214]
    val = df[df["Day"] > 214]

    return train, val

def build_test_data():
    """
    Returns dicts of all test data
    Format: {id : [x, y]}
    """

    test_data = {}
    test_folder = "../data/final_phase_test/"
    test_folder_y = "../data/final_phase_test/outfile/"

    for filename in os.listdir(test_folder + "infile"):
        x_path = os.path.join(test_folder + "infile", filename)
        # checking if it is a file
        if os.path.isfile(x_path):
            id = int(filename[:4])
            #y_path = x_path[:-6] + "out.csv"
            y_path = test_folder_y + filename[:4] + "out.csv"

            test_data[id] = [x_path, y_path]
    return test_data

def evaluation(y_pred, path_to_test_y, test_y="file"):
    """
    Single timestamp predictions
    Input shape: (134, 288, 1)
    Output: Final score of model
    Evaluate prediction result for each turbine, and then sum the prediction scores as the final score.
    """

    print("GOING THROUGH")
    print(path_to_test_y)

    if test_y == "file":
        # One instance of y with all turbines
        df = pd.read_csv(path_to_test_y)
    else:
        pass

    turbine_ids = df["TurbID"].unique()

    # Score for one data instance of 142
    S_t = 0
    for turb_id in turbine_ids: # For each turbine

        df_y = df[df["TurbID"] == turb_id]
        # Abnormal conditions
        combined_conditions = (
            (df_y["Patv"] < 0) |
            (df_y["Wspd"] < 1) & (df_y["Patv"] > 10) |
            (df_y["Wspd"] < 2) & (df_y["Patv"] > 100) |
            (df_y["Wspd"] < 3) & (df_y["Patv"] > 200) |
            (df_y["Wspd"] > 2.5) & (df_y["Patv"] == 0) |
            (df_y["Wspd"] == 0) & (df_y["Wdir"] == 0) & (df_y["Etmp"] == 0) |
            (df_y["Etmp"] < -21) |
            (df_y["Itmp"] < -21) |
            (df_y["Etmp"] > 60) |
            (df_y["Itmp"] > 70) |
            (df_y["Wdir"] > 180) | (df_y["Wdir"] < -180) |
            (df_y["Ndir"] > 720) | (df_y["Ndir"] < -720) |
            (df_y["Pab1"] > 89) | (df_y["Pab2"] > 89) | (df_y["Pab3"] > 89)
        )
        actual_y = np.array(df_y["Patv"])
        mask_test = np.array(combined_conditions)
        turb_pred = y_pred[turb_id-1,:,0]
        turb_pred[mask_test] = actual_y[mask_test]

        ## RMSE
        turb_mse = np.nanmean((actual_y - turb_pred)**2)
        turb_rmse = np.sqrt(turb_mse)
        
        S_t += turb_rmse
    return S_t

def bert_predict(path_to_test_x, settings):
    settings['path_to_test_x'] = path_to_test_x
    settings['data_path'] = '../models/KDDCUP2022_BERT/data/raw'
    y = forecast(settings)
    return y

def gnn_predict(path_to_test_x, settings):
    settings['path_to_test_x'] = path_to_test_x
    settings['data_path'] = '../models/KDDCUP2022_BERT/data/raw' # Re-using path from BERT
    y = forecast(settings)
    return y

def arima_predict(path_to_test_x, settings):
    test_x = pd.read_csv(path_to_test_x)

    turbine_ids = test_x["TurbID"].unique() # Unique turbine IDs

    forecast_steps = settings["forecast_steps"]
    models = settings["models"]

    all_forecasts = []

    for turb_id in turbine_ids:
        #if turb_id == 3:
        #    break

        current_x = np.array(test_x[test_x["TurbID"] == turb_id]["Patv"].fillna(method='ffill').fillna(method='bfill'))
        model = models[turb_id].apply(current_x)
        turbine_forecast = model.forecast(forecast_steps)

        all_forecasts.append(turbine_forecast)

    y = np.array(all_forecasts)

    return y.reshape((len(turbine_ids), forecast_steps, 1))


def evaluate_model(model="bert"):

    """
        model: Str of model name or dict of bottom ARIMA models
    """

    test_data = build_test_data() # Get test data
    
    if model == "bert":
        # KDD Cup 3rd place evaluation - BERT Model
        settings = prep_env()
        model_wrapper = bert_predict

    elif model == "gnn":
        # KDD Cup 11th place evaluation - GNN
        settings = prep_env()
        model_wrapper = gnn_predict

    else:
        # ARIMA
        settings = {"forecast_steps": 288, "models": model}
        model_wrapper = arima_predict

    score_sum = 0
    for data in test_data:
        test_x_path = test_data[data][0]
        test_y_path = test_data[data][1]
        #y_pred = get_bert(test_x_path)
        y_pred = model_wrapper(test_x_path, settings) # Generic model_wrapper made with either specific model
        score_sum += evaluation(y_pred, test_y_path)
    total_score = score_sum / len(test_data)

    return total_score

# Check for stationarity
def check_stationarity(time_series):
    print(time_series)

    result = adfuller(time_series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] > 0.05:
        print('The time series is not stationary. Perform differencing.')
        return False
    else:
        print('The time series is stationary.')
        return True

def baseline(cross_validation=True):
    """
        Hierarchical bottom-up ARIMA model, a model for each wind turbine
    """

    forecast_steps = 288 # 48 hours: 48*6

    train_data, val_data = build_train_data()

    all_data = pd.concat([train_data, val_data])

    turbine_ids = train_data["TurbID"].unique() # Unique turbine IDs
    bottom_models = {}

    #orders = [(i,j,k) for i in range(1,5) for j in range(3) for k in range(1,5)] # Cross validate orders
    #order = (1, 0, 150)  # (p, d, q) p: PACF, d: Stationary, q: ACF
    order = (1, 0, 10)

    ## CARBONTRACKER START
    tracker = CarbonTracker(epochs=1, epochs_before_pred=0, monitor_epochs=1, verbose=3)
    tracker.epoch_start() 

    print("STARTS TRAINING ARIMA")
    
    for turb_id in turbine_ids:

        print("MODEL #", turb_id)
        #if turb_id == 3:
        #    break

        #current_train = train_data[train_data["TurbID"] == turb_id]
        #current_val = val_data[val_data["TurbID"] == turb_id]
        current_data = np.array(all_data[all_data["TurbID"] == turb_id]["Patv"].fillna(method='ffill').fillna(method='bfill'))

        model = ARIMA(current_data, order=order)
        fit_model = model.fit()
        bottom_models[turb_id] = fit_model

        #forecast = fit_model.forecast(steps=forecast_steps)
        #print(forecast)
        #print(fit_model.aic)

    ### CARBONTRACKER END
    tracker.epoch_end()
    tracker.stop()

    print(bottom_models)
    #print(train_data)
    #print(val_data)

    return bottom_models

    #model = ARIMA(series, order=(1,1,1))
    #model_fit = model.fit()

if __name__ == "__main__":

    # Reading argument: model name. Either "bert", "gnn" or "arima" for evaluation. Arima includes training.
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--model", help="name of model")
    args = argParser.parse_args()

    # Evaluate depending on model in user input
    if args.model == "bert":
        # KDD Cup 3rd place evaluation - BERT Model
        from KDDCUP2022_BERT.weights.prepare import prep_env
        from KDDCUP2022_BERT.weights.predict import forecast
        model = "bert"
    elif args.model == "gnn":
        # KDD Cup 11th place evaluation - GNN
        sys.path.insert(0, '../models/KDDCUP2022_GNN/submit') 
        #from KDDCUP2022_GNN.submit.weights import prep_env, forecast
        from KDDCUP2022_GNN.submit.prepare import prep_env
        from KDDCUP2022_GNN.submit.predict import forecast
        model = "gnn"
    elif args.model == "arima":
        model = baseline() # Trains arima, returns dictionary with each bottom turbine model
    
    score = evaluate_model(model)

    print(args.model + "score:")
    print(score)