import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from preprocess import read_data

def whole_single_turbine(df, turb_id=1):
    df = df[df["TurbID"] == 1]
    grouped_df = df.groupby(['Day'])['Patv', 'Wspd'].mean().reset_index().rename(columns={"Patv": "Mean patv"})

    # Plotting with two y-axes
    fig, ax1 = plt.subplots()

    color = 'darkorange'
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Daily Patv', color=color)
    ax1.plot(grouped_df["Day"], grouped_df["Mean patv"], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Creating a second y-axis
    ax2 = ax1.twinx()
    color = 'darkgreen'
    ax2.set_ylabel('Wind Speed', color=color)
    ax2.plot(grouped_df["Day"], grouped_df["Wspd"], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.savefig('src/figs/whole_period.png')
    plt.show()

def pacf_plot(df, turb_id=1):
    df = df[df["TurbID"] == 1]
    data_a = np.array(df["Patv"])
    print(data_a)

    data_a = data_a[~np.isnan(data_a)]
    print(data_a)

    plt.rc("figure", figsize=(30,15))
    plt.figure(figsize=(30,15))
    plot_pacf(data_a, lags=3*60/10, markersize=20, linewidth=2) # lags = amount of timesteps back

    plt.xlabel('Lag', fontsize = 45)
    plt.ylabel('PACF', fontsize = 45)
    plt.title('Partial Autocorrelation', fontsize = 50)
    plt.xticks(fontsize = 40) 
    plt.yticks(fontsize = 40) 

    plt.tight_layout()

    plt.savefig('src/figs/pacf_plot.png')
    plt.show()

def acf_plot(df, turb_id=1):
    df = df[df["TurbID"] == 1]
    data_a = np.array(df["Patv"])
    print(data_a)

    plt.rc("figure", figsize=(30,15))
    plt.figure(figsize=(30,15))
    plot_acf(data_a, lags=24*60/10*4, missing="drop", markersize=3, linewidth=0.00001) # lags = amount of timesteps back

    plt.xlabel('Lag', fontsize = 45)
    plt.ylabel('ACF', fontsize = 45)
    plt.title('Autocorrelation', fontsize = 50)
    plt.xticks(fontsize = 40) 
    plt.yticks(fontsize = 40) 

    plt.tight_layout()

    plt.savefig('src/figs/acf_plot.png')
    plt.show()

def single_turbine(df, turb_ids=[1,4]):
    df = df[df["TurbID"].isin(turb_ids)]

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
    df["Condition"] = "Normal"
    df.loc[combined_conditions, "Condition"] = "Abnormal"

    #plt.scatter(df["Wspd"], df["Patv"], s=0.5, marker='o')
    #plt.figure(figsize=(30,60))
    g = sns.relplot(data=df, x='Wspd', y='Patv', s=25, hue="Condition", height=6, aspect=1.5, col="TurbID", col_wrap=1, linewidth = 0) #.set(xlabel='Wind Speed', ylabel='Patv')
    # Set titles for each subplot
    g.set_titles(col_template="Turbine ID: {col_name}", size=20)
    # Set sizes for x and y tick labels
    g.set(xlabel="Wind Speed", ylabel="Patv")
    # Set tick sizes by accessing Axes objects directly
    for ax in g.axes.flat:
        ax.tick_params(axis='both', which='both', labelsize=16)
        ax.xaxis.label.set_size(20)  # Set x-axis label size
        ax.yaxis.label.set_size(20)  # Set y-axis label size

    #g._legend.set_title('Size')  # Set title size
    g._legend.set_bbox_to_anchor([1, 0.6])  # Adjust legend position
    g._legend.get_title().set_fontsize(20)  # Set legend title size
    for item in g._legend.get_texts():
        item.set_fontsize(16)  # Set legend item size

    # Save the figure as a PNG file
    plt.savefig('src/figs/one_turbine_scatter' + '.png')

    # Show the plot
    plt.show()

def spatial_distribution(df):

    # Group by turbine with mean patv to color by
    grouped_df = df.groupby(['TurbID', 'x', 'y'])['Patv'].mean().reset_index().rename(columns={"Patv": "Mean patv"})

    # Scatter plot with labels
    #plt.scatter(df['x'], df['y'])
    sns.relplot(data=grouped_df, x='x', y='y', hue='Mean patv', palette=sns.color_palette("flare", as_cmap=True)) #, aspect=1.61)

    # Add axis labels
    plt.xlabel('X')
    plt.ylabel('Y')

    #plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig('src/figs/spatial_distribution.png')

    # Show the plot
    plt.show()

def mean_bar_plots(df):

    np.random.seed(42)

    grouped_df = df.groupby('TurbID')['Patv'].mean().reset_index().sort_values("Patv")
    top_10 = pd.concat([grouped_df[:10], grouped_df[-10:]])

    # Plot each column as a bar chart
    top_10.plot.bar(x="TurbID", y="Patv")

    plt.ylabel('Mean Patv')
    plt.xlabel('Turbine ID')
    plt.legend().remove()
    plt.axhline(y = 400, color = 'r', linestyle = 'dashed')
    plt.axhline(y = 300, color = 'g', linestyle = 'dashed')  

    # Save the figure as a PNG file
    plt.savefig('src/figs/mean_patv.png')

    # Show the plots
    plt.show()

def bar_plot(df, turb_id=1):
    # Create a sample DataFrame with 12 columns
    columns = [
        "Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"
    ]

    df = df[df["TurbID"] == turb_id]

    np.random.seed(42)

    # Plot each column as a bar chart
    #df[columns].plot(kind='bar', subplots=True, layout=(1, 10), figsize=(12, 8), sharex=False)
    """df[columns].hist(bins=100, layout=(10, 1), figsize=(12, 16), sharex=False)
    plt.ylabel('Numbers')
    plt.tight_layout()"""

    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(12, 16), sharex=False, sharey=True)

    for i, col in enumerate(columns):
        data_min = df[col].min()
        data_max = df[col].max()
        axes[i].hist(df[col], bins=100)
        #axes[i].set_title(col, size=20)  # Optionally, set subplot titles
        # Customize y-axis limits if needed
        axes[i].set_ylim([data_min, data_max])
        axes[i].set_ylabel("Count", size=20) 
        axes[i].set_xlabel(col, size=20) 
        axes[i].tick_params(axis='both', labelsize=16)

    plt.tight_layout()

    # Save the figure as a PNG file
    name = "bar_turb_" + str(turb_id) + ".png"
    plt.savefig('src/figs' + name)

    # Show the plots
    plt.show()

if __name__ == "__main__":
    folder = "../data/"
    file = "wtbdata_245days.csv"
    file2 = "sdwpf_baidukddcup2022_turb_location.CSV"
    df = read_data(folder + file, folder + file2, clean=False)
    df2 = pd.read_csv(folder + file2)

    #bar_plot(df)
    #spatial_distribution(df)
    #mean_bar_plots(df)
    #single_turbine(df)
    #acf_plot(df)
    pacf_plot(df)
    #whole_single_turbine(df)
    