import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from MachineLearningModels.DataPreProcess import data_preprocess, denormlize_data
import pandas as pd


def SJF(df, burst_time_col='', processID_col=''):
    """
    Shortest Job First Algorithm
    input: dataframe, burst_time_col, processID_col
    output: same data frame with new columns: waiting_time, turn_around_time as well as average waiting time and average turn around time
    """
    # sort the data frame based on burst time
    df = df.sort_values(by=burst_time_col)
    #assuming the data arrived at time 0
    df['waiting_time'] = 0
    df['turn_around_time'] = 0
    # calculate waiting time and turn around time for each process
    for idx in df.index:
        if idx == df.index[0]:
            df.loc[idx, 'waiting_time'] = 0  # First process has no waiting time
        else:
            prev_idx = df.index[df.index.get_loc(idx) - 1]  # Get the previous index
            df.loc[idx, 'waiting_time'] = df.loc[prev_idx, burst_time_col] + df.loc[prev_idx, 'waiting_time']

        df.loc[idx, 'turn_around_time'] = df.loc[idx, burst_time_col] + df.loc[idx, 'waiting_time']
    # calculate the average waiting time and average turn around time
    avg_waiting_time = df['waiting_time'].mean()
    avg_turn_around_time = df['turn_around_time'].mean()
    return df, avg_waiting_time, avg_turn_around_time


if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), '..', 'MachineLearningModels', 'processes_datasets.csv')
    df1 = pd.read_csv(file_path)
    df1, mapping1 = data_preprocess(df1,target_col=['JobID','RunTime '], samples=100, no_norm_col=['JobID'])
    df2 = pd.read_csv(file_path)
    df2, mapping2 = data_preprocess(df2, target_col=['JobID', 'RunTime '], samples=1000, no_norm_col=['JobID'])
    df1 = denormlize_data(df1, mapping1, no_norm_col=['JobID'])
    df2 = denormlize_data(df2, mapping2, no_norm_col=['JobID'])
    print(df2)
    df2, avg_waiting_time, avg_turn_around_time = SJF(df2, 'RunTime ', 'JobID')
    print(df2)
    print(f"Average Waiting Time: {avg_waiting_time}")
    print(f"Average Turn Around Time: {avg_turn_around_time}")