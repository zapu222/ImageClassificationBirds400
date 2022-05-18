import os
import json
import torch
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import Birds400
from utils import create_model
from utils import count_parameters


def test(args):
    # Parameters
    data_path, model_path, model_name, classes, device = \
        args['data_path'], args['model_path'], args['model'], args['classes'], args['device']

    # Datasets
    testset = Birds400(data_path, task="test")

    # Testloader
    testloader = DataLoader(testset, batch_size=16, num_workers=2, shuffle=True)

    # Model
    model = create_model(model_name, classes)
    model.to(device)
    print(f"\nModel: {model_name.upper()}\nTrainable parameters: {count_parameters(model)}\nLoaded from: {model_path}")

    # Load model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Testing
    correct_1, correct_5, total = 0, 0, 0
    with torch.no_grad():
        print(f"\nTesting on {testset.__len__()} images")
        for _, data in enumerate(tqdm(testloader, desc="Testset", bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
            
            inputs, labels = data   # images and labels
            
            if device == "cuda":
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)   # model outputs

            _, labels = torch.max(labels.squeeze().data, 1)   # label indices
            _, top_1 = torch.max(outputs.data, 1)   # top prediction
            _, top_5 = torch.topk(outputs, 5)   # top 5 predictions

            total += labels.size(0)   # total predictions
            correct_1 += (top_1 == labels).sum().item()   # correct predictions
            for i in range(top_5.shape[0]):
                if labels[i] in top_5[i]:
                    correct_5 += 1   # correct predictions @ 5

    test_acc_1 = correct_1/total
    test_acc_5 = correct_5/total

    print(f"\nAcc@1: {correct_1} / {total} = {round(100*test_acc_1, 3)} %\nAcc@5: {correct_5} / {total} = {round(100*test_acc_5, 3)} %")


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