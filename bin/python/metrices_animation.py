import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import random
import pandas as pd

def animate(i, data_1, data_2, line1_fig, line2_fig):
    temp1 = data_1.iloc[:int(i+1)]
    temp2 = data_2.iloc[:int(i+1)]

    line1_fig.set_data(temp1.index, temp1['values']) # (values) column
    line2_fig.set_data(temp2.index, temp2['values']) # (values) column

    return (line1_fig, line2_fig)


def create_animation(model_type, data_1, data_2):
    fig = plt.figure() # init fig
    plt.title(f'Accuracy & Loss', fontsize=15) # Main Title
    plt.xlabel('Epochs', fontsize=20) # Bottom Title
    plt.ylabel('Loss VS Accuracy', fontsize=15) # Y Label

    plt.xlim(min(data_1.index.min(), data_2.index.min()), max(data_1.index.max(), data_2.index.max())) # set min-max range of x-axis
    plt.ylim(min(data_1.values.min(), data_2.values.min()), max(data_1.values.max(), data_2.values.max())) # set min-max range of y-axis

    l1_fig, = plt.plot([], [], 'o-', label='Train Accuracy', color='b', markevery=[-1])
    l2_fig, = plt.plot([], [], 'o-', label='Train Loss', color='r', markevery=[-1])
    plt.legend(loc='center right', fontsize='medium')



    ani = animation.FuncAnimation(fig, animate, fargs=(data_1, data_2, l1_fig, l2_fig), repeat=True, interval=50, repeat_delay=50)
    plt.show()


# create datasets
def init():
    try:
        data_1 = pd.read_csv('bin\\metrices\\accuracy_logs.csv')
        data_2 = pd.read_csv('bin\\metrices\\loss_logs.csv')
    except FileNotFoundError:
        print("Animation Failed!, Files are missing...")
        exit(-1)

    data_1.reset_index(inplace=True)
    data_1.drop('index', axis=1, inplace=True)
    data_2.reset_index(inplace=True)
    data_2.drop('index', axis=1, inplace=True)

    create_animation('test', data_1, data_2)

if __name__ == "__main__":
    init()