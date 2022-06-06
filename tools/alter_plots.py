import os
import argparse
import pandas as pd

from matplotlib import pyplot as plt


def alter_plots(args):
    folder, epoch = args['folder'], args['epoch']

    log = pd.read_csv(os.path.join(folder, "log.csv"))

    epochs = log.iloc[0:epoch]["Epoch"].tolist()
    avg_train_loss = log.iloc[0:epoch]["Avg Train Loss"].tolist()
    avg_val_loss = log.iloc[0:epoch]["Avg Val Loss"].tolist()
    train_acc_1 = log.iloc[0:epoch]["Train Acc@1"].tolist()
    train_acc_5 = log.iloc[0:epoch]["Train Acc@5"].tolist()
    val_acc_1 = log.iloc[0:epoch]["Val Acc@1"].tolist()
    val_acc_5 = log.iloc[0:epoch]["Val Acc@5"].tolist()

    items = [epochs, avg_train_loss, avg_val_loss, train_acc_1, train_acc_5, val_acc_1, val_acc_5]

    # Loss plot
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_axes([0.1,0.1,0.75,0.75]) # axis starts at 0.1, 0.1
    ax.set_title("Loss per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.plot(items[0], items[1])
    ax.plot(items[0], items[2])

    ax.legend(['Training Loss', 'Validation Loss'], loc='upper right')

    ax.set_ylim(ymax=7)
    ax.set_ylim(ymin=0)
    fig.savefig(os.path.join(folder, 'loss_alt.jpg'))
    plt.close(fig)

    # Accuracy plot
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_axes([0.1,0.1,0.75,0.75]) # axis starts at 0.1, 0.1
    ax.set_title("Accuracy per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    ax.plot(items[0], items[3])
    ax.plot(items[0], items[4])
    ax.plot(items[0], items[5])
    ax.plot(items[0], items[6])

    ax.legend(['Train Acc@1', 'Train Acc@5', 'Valid Acc@1', 'Valid Acc@5'], loc='lower right')

    ax.set_ylim(ymax=1)
    ax.set_ylim(ymin=0)
    fig.savefig(os.path.join(folder, 'accuracy_alt.jpg'))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='', help='path to model folder')
    parser.add_argument('--epoch', type=int, default=50, help='epochs to include')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    alter_plots(args)