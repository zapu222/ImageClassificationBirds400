import os
import cv2
import random
import argparse

from matplotlib import pyplot as plt


def run(args):
    data, num = args.data, args.num

    fig = plt.figure(figsize=(14, 10))

    train = os.path.join(data, "train")
    valid = os.path.join(data, "valid")
    test = os.path.join(data, "test")

    splits = [train, valid, test]
    images = []

    for split in splits:
        folders = os.listdir(os.path.join(data, split))
        for folder in folders:
            imgs = os.listdir(os.path.join(data, split, folder))
            for img in imgs:
                images.append(os.path.join(data, split, folder, img))

    random.shuffle(images)

    for i in range(num*num):
        image = cv2.imread(images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig.add_subplot(num, num, i+1)
        plt.imshow(image)
        plt.axis('off')

        path = os.path.normpath(images[i])
        path = path.split(os.sep)

        plt.title(path[-2], y=0.91, fontsize=6)

    plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='path to BIRDS_400')
    parser.add_argument('--num', type=int, default=8, help='height and width of grid')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()
    run(args)