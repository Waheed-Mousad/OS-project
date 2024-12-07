import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from MachineLearningModels.DataPreProcess import data_preprocess, denormlize_data
from SchedulingAlgorithms.ShortestJobFirst import SJF
from SchedulingAlgorithms.FirstComeFirstServe import FCFS
import pandas as pd
import matplotlib.pyplot as plt
# plot turn around time vs waiting time for each algorithm
def plot_turn_around_vs_waiting_time(package, extra_info = ''):
    df1 = package[0][0].copy()
    df2 = package[1][0].copy()
    avg_waiting_time1 = package[0][1]
    avg_turn_around_time1 = package[0][2]
    avg_waiting_time2 = package[1][1]
    avg_turn_around_time2 = package[1][2]
    # re index the data frame make sure they start from 0, the real index is not 0
    df1.index = range(1, len(df1) + 1)
    df2.index = range(1, len(df2) + 1)
    # plot waiting time only in Y axis and JobID in X axis , take job ID = index starting from 1
    plt.plot(df1.index, df1['SJF waiting_time'], label='SJF waiting time', marker='o', linestyle='-')
    plt.plot(df2.index, df2['FCFS waiting_time'], label='FCFS waiting time', marker='o', linestyle='-')
    #print the average waiting time of each algorithm on the side of the plot
    plt.text(0, avg_waiting_time1, f'Avg Waiting Time SJF: {avg_waiting_time1}', fontsize=12, color='blue')
    plt.text(0, avg_waiting_time2, f'Avg Waiting Time FCFS: {avg_waiting_time2}', fontsize=12, color='orange')
    plt.xlabel('JobID')
    plt.ylabel('Time as unit of time')
    title = 'Waiting Time SJF vs FCFS'
    if len(extra_info) != 0:
        title = title + ' for ' + extra_info
    plt.title(title)
    plt.legend()
    plt.show()
    # plot turn around time only in Y axis and JobID in X axis , take job ID = index starting from 1
    plt.plot(df1.index, df1['SJF turn_around_time'], label='SJF turn around time', marker='o', linestyle='-')
    plt.plot(df2.index, df2['FCFS turn_around_time'], label='FCFS turn around time', marker='o', linestyle='-')
    #print the average turn around time of each algorithm on the side of the plot
    plt.text(0, avg_turn_around_time1, f'Avg Turn Around Time SJF: {avg_turn_around_time1}', fontsize=12, color='blue')
    plt.text(0, avg_turn_around_time2, f'Avg Turn Around Time FCFS: {avg_turn_around_time2}', fontsize=12, color='orange')
    plt.xlabel('JobID')
    plt.ylabel('Time as unit of time')
    title = 'Turn Around Time SJF vs FCFS'
    if len(extra_info) != 0:
        title = title + ' for ' + extra_info
    plt.title(title)
    plt.legend()
    plt.show()





if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), '..', 'MachineLearningModels', 'processes_datasets.csv')
    df = pd.read_csv(file_path)
    # Re index so index = JobID
    df.index = df.index + 1
    data1, mapping1 = data_preprocess(df, target_col=[
        'SubmitTime', 'WaitTime', 'AverageCPUTimeUsed', 'Used Memory',
        'ReqTime: ', 'ReqMemory', 'Status', 'UserID', 'RunTime '
    ], samples=200)

    data1 = denormlize_data(data1, mapping1)

    temp = [[],[]]
    df , avg_waiting_time, avg_turn_around_time = SJF(data1, data1.columns[-1])
    temp[0] = [df, avg_waiting_time, avg_turn_around_time]
    df , avg_waiting_time, avg_turn_around_time = FCFS(data1, data1.columns[-1])
    temp[1] = [df, avg_waiting_time, avg_turn_around_time]
    plot_turn_around_vs_waiting_time(temp)
    #print average 'RunTime ' for the data frame
    print(data1['RunTime '].mean())



