import os
import argparse

def run(args):
    data = args.data

    train = os.path.join(data, "train")
    valid = os.path.join(data, "valid")
    test = os.path.join(data, "test")

    train_folders = os.listdir(train)
    train_tot = []
    for folder in train_folders:
        train_tot.append(len(os.listdir(os.path.join(data, train, folder))))

    valid_folders = os.listdir(valid)
    valid_tot = []
    for folder in valid_folders:
        valid_tot.append(len(os.listdir(os.path.join(data, valid, folder))))

    test_folders = os.listdir(test)
    test_tot = []
    for folder in test_folders:
        test_tot.append(len(os.listdir(os.path.join(data, test, folder))))

    print(f"\nNumber of classes: {len(train_tot)}\n")
    print(f"Total valid images: {sum(train_tot):>5} - average images per class {round(sum(train_tot)/len(train_tot), 1):>5}")
    print(f"Total valid images: {sum(valid_tot):>5} - average images per class {round(sum(valid_tot)/len(valid_tot), 1):>5}")
    print(f"Total test images:  {sum(test_tot):>5} - average images per class {round(sum(test_tot)/len(test_tot), 1):>5}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='path to BIRDS_400')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()
    run(args)