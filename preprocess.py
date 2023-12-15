import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

def fill_df(df, all_columns):

    def fill_helper(timestamp, x_coord, y_coord, column):

        #start_time = time.time()
        _, indices = nbrs.kneighbors([[x_coord, y_coord]])
        #print("--- 1) %s seconds ---" % (time.time() - start_time))

        #start_time = time.time()
        nearest_turbines = list(turbine_locations.iloc[indices[0]]["TurbID"])
        #print("--- 2) %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        subset_df = df.loc[(nearest_turbines, timestamp), column] # Optimized indexing
        #print("--- 3) %s seconds ---" % (time.time() - start_time))

        #start_time = time.time()
        column_average = np.nanmean(subset_df)
        #print("--- 4) %s seconds ---" % (time.time() - start_time))

        return column_average


    turbine_locations = df.drop_duplicates(subset=["x", "y"], keep='first')[["TurbID", "x", "y"]]
    nbrs = NearestNeighbors(n_neighbors=10).fit(np.array(turbine_locations[["x", "y"]]))

    df.set_index(["TurbID", "Tmstamp"], inplace=True) # For faster indexing

    for column in all_columns:
        print("GOING THROUGH:", column)
        df[column] = df.apply(
            lambda x:
                fill_helper(
                    x.name[1], x["x"], x["y"], column
                ) if pd.isnull(x[column]) else x[column],
            axis=1
            )


def clean_data(df):

    # Columns to change
    all_columns = list(df.columns)
    for element in ["TurbID", "Day", "Tmstamp", "x", "y"]:
        try:
            all_columns.remove(element)
        except:
            pass

    # Abnormal conditions
    combined_conditions = (
        (df["Patv"] < 0) |
        (df["Wspd"] < 1) & (df["Patv"] > 10) |
        (df["Wspd"] < 2) & (df["Patv"] > 100) |
        (df["Wspd"] < 3) & (df["Patv"] > 200) |
        (df["Wspd"] > 2.5) & (df["Patv"] == 0) |
        (df["Wspd"] == 0) & (df["Wdir"] == 0) & (df["Etmp"] == 0) |
        (df["Etmp"] < -21) |
        (df["Itmp"] < -21) |
        (df["Etmp"] > 60) |
        (df["Itmp"] > 70) |
        (df["Wdir"] > 180) | (df["Wdir"] < -180) |
        (df["Ndir"] > 720) | (df["Ndir"] < -720) |
        (df["Pab1"] > 89) | (df["Pab2"] > 89) | (df["Pab3"] > 89)
    )

    print("# OF NAN VALUES IN PATV (1)")
    print(df['Patv'].isna().sum())

    # Make selected columns with rows that match conditions NaN
    df.loc[combined_conditions, all_columns] = np.nan

    print("# OF NAN VALUES IN PATV (2)")
    print(df['Patv'].isna().sum())

    # Change NaN values to average of k-nearest neighbours at same Tmstamp
    filled_df = fill_df(df, all_columns)

    print("# OF NAN VALUES IN PATV (3)")
    print(df['Patv'].isna().sum())

    return df

def read_data(filepath, filepath2, clean=True):
    df = pd.read_csv(filepath)
    df2 = pd.read_csv(filepath2)
    new_df = df.merge(df2, how="left", on="TurbID")

    new_df = new_df #[:500000] # Testing
    
    if clean==True:
        new_df = clean_data(new_df)
    
    return new_df

if __name__ == "__main__":
    folder = "../data/"
    file = "wtbdata_245days.csv"
    file2 = "sdwpf_baidukddcup2022_turb_location.CSV"
    cleaned_df = read_data(folder + file, folder + file2)

    cleaned_df.to_csv(folder + "preprocessed_train_data.csv")



