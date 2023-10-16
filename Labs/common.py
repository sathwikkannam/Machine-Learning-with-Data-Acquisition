import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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


def scatter_plot(sets: list[pd.DataFrame]):
    ax = None
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'b']

    for i, t in enumerate(sets):
        ax = t.plot(kind='scatter', x='ax', y='ay', color=colors[i], label=str(t['class'][0]), ax=ax)

    return ax


def plot_loss_accuracy(history, name="", same_graph=False):
    plt.style.use('ggplot')

    if same_graph:
        # Both training and validation in the same plot
        plt.figure(figsize=(15, 5))
        plt.title(f'{name} - Training and Validation', fontsize=16, fontname='Arial')
        plt.plot(history[f'loss'], label=f'loss', linestyle='--', linewidth=2, color='blue')
        plt.plot(history[f'accuracy'], label=f'accuracy', linestyle='-', linewidth=2, color='red')
        plt.plot(history[f'val_loss'], label=f'val_loss', linestyle='-.', linewidth=2, color='cyan')
        plt.plot(history[f'val_accuracy'], label=f'val_accuracy', linestyle=':', linewidth=2, color='magenta')
        plt.xlabel('Epochs', fontsize=12, fontname='Arial')
        plt.ylabel('Loss / Accuracy', fontsize=12, fontname='Arial')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        # Training set
        plt.figure(figsize=(15, 5))
        plt.title(f'{name} - loss and accuracy', fontsize=16, fontname='Arial')
        plt.plot(history[f'loss'], label=f'loss', linestyle='--', linewidth=2, color='blue')
        plt.plot(history[f'accuracy'], label=f'accuracy', linestyle='-', linewidth=2, color='red')
        plt.xlabel('Epochs', fontsize=12, fontname='Arial')
        plt.ylabel('Loss / Accuracy', fontsize=12, fontname='Arial')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Validation set
        plt.figure(figsize=(15, 5))
        plt.title(f'{name} - val_loss and val_accuracy', fontsize=16, fontname='Arial')
        plt.plot(history[f'val_loss'], label=f'val_loss', linestyle='--', linewidth=2, color='blue')
        plt.plot(history[f'val_accuracy'], label=f'val_accuracy', linestyle='-', linewidth=2, color='red')
        plt.xlabel('Epochs', fontsize=12, fontname='Arial')
        plt.ylabel('Loss / Accuracy', fontsize=12, fontname='Arial')
        plt.grid(True)
        plt.legend()
        plt.show()


def display_loss_accuracy(scores, title=None):
    if title:
        print(f"title={title}")

    print(f"loss: {scores[0]} accuracy: {scores[1]}")