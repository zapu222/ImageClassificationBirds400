import argparse
import os
import json
import argparse

import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
  
from utils import create_model

from torchsummary import summary


def main(args):
    model, classes = args.model, args.classes

    print("")
    model = create_model(model, classes)
    model.to('cuda')
    summary(model, (3, 224, 224))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='model type')
    parser.add_argument('--classes', type=int, default=400, help='number of classes')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()
    main(args)