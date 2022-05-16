import os
import json
import torch
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import Birds400
from utils import create_model


def test(args):
    data_path, model_path, model, classes, device = \
        args['data_path'], args['model_path'], args['model'], args['classes'], args['device']

    # Datasets
    testset = Birds400(data_path, task="test")

    # Testloader
    testloader = DataLoader(testset, batch_size=16, num_workers=2, shuffle=True)

    model = create_model(model, classes)
    model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Testing
    correct, total = 0, 0
    with torch.no_grad():
        print(f"\n  Testing on {testset.__len__()} images...")
        for _, data in enumerate(tqdm(testloader, 0, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
            inputs, labels = data
            
            if device == "cuda":
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)

            _, labels = torch.max(labels.squeeze().data, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct/total

    print(f"  Accuracy: {correct} / {total} = {round(100*test_acc, 3)} %\n")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, default='', help='test_args.json path')

    opt = parser.parse_args()
    with open(opt.args_path, 'r') as f:
        args = json.load(f)
    return args


if __name__ == "__main__":
    args = parse_opt()
    test(args)