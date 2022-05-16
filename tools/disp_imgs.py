import os
import cv2
import random
import argparse

from matplotlib import pyplot as plt


def run(args):
    data = args.data

    fig = plt.figure(figsize=(14, 10))
    train = os.path.join(data, "train")
    images = []

    folders = os.listdir(os.path.join(data, train))
    for folder in folders:
        imgs = os.listdir(os.path.join(data, train, folder))
        for img in imgs:
            images.append(os.path.join(data, train, folder, img))

    random.shuffle(images)

    for i in range(64):
        image = cv2.imread(images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig.add_subplot(8, 8, i+1)
        plt.imshow(image)
        plt.axis('off')

        path = os.path.normpath(images[i])
        path = path.split(os.sep)

        plt.title(path[-2], y=0.91, fontsize=6)

    plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='path to BIRDS_400')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()
    run(args)