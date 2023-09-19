import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from pathlib import Path

acc_names = ['ax', 'ay', 'az']
rate_gyro_names = ['gx', 'gy', 'gz']
sensors = ["Accelerometer", "Gyroscope"]


def sns_cm(y_true, y_pred):
    sns.heatmap(
        confusion_matrix(y_true=y_true, y_pred=y_pred),
        annot=True, fmt='d',
        cmap='Blues',
        cbar=False,
        square=True,
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true)
    )


def get_set(folder, positions: list[str] = None, aslist=True):
    temp = []
    temp_dict = {}

    for _class in Path(f"../Binaries/{folder}").rglob("*.pkl"):
        if positions is None:
            if aslist:
                temp.append(pd.read_pickle(_class))
            else:
                temp_dict[pd.read_pickle(_class)['class'][0]] = pd.read_pickle(_class)
        else:
            for stat_pos in positions:
                if str(_class).__contains__(stat_pos):
                    if aslist:
                        temp.append(pd.read_pickle(_class))
                    else:
                        temp_dict[pd.read_pickle(_class)['class'][0]] = pd.read_pickle(_class)

    return temp if aslist else temp_dict


def display_accuracy(valid, test):
    print(f"Validation set accuracy: {valid}\nTest set accuracy: {test}")
