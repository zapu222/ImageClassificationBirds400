import os
import json
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import Birds400
from utils import create_model


def train(args):
    # Parameters
    data_path, save_path, model, classes, img_size, batch_size, workers, lr, epochs, device = \
        args['data_path'], args['save_path'], args['model'], args['num_classes'], args["img_size"],\
            args["batch_size"], args['workers'], args['lr'], args['epochs'], args['device']

    # Datasets
    trainset = Birds400(data_path, task="train", img_size=img_size)
    valset = Birds400(data_path, task="valid")

    # Train and Valid loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=workers, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, num_workers=workers, shuffle=True)

    # Model
    model = create_model(model, classes)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Logging
    log = []
    best = 0

    # Begin training...
    for epoch in range(epochs):
        print(f"  Epoch: {epoch + 1} of {epochs}")
        running_loss = []

        # Training
        tcorrect, ttotal = 0, 0
        for _, data in enumerate(tqdm(trainloader, 0, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
            inputs, labels = data

            if device == "cuda":
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

            _, labels = torch.max(labels.squeeze().data, 1)
            _, predicted = torch.max(outputs.data, 1)
            ttotal += labels.size(0)
            tcorrect += (predicted == labels).sum().item()

        # Training accuracy
        train_acc = tcorrect/ttotal

        # Save model
        torch.save(model.state_dict(), os.path.join(save_path, "last.pth"))
    
        # Validation
        vcorrect, vtotal = 0, 0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                
                if device == "cuda":
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)

                _, labels = torch.max(labels.squeeze().data, 1)
                _, predicted = torch.max(outputs.data, 1)
                vtotal += labels.size(0)
                vcorrect += (predicted == labels).sum().item()

        # Validation accuracy
        val_acc = vcorrect/vtotal

        # Average loss for epoch
        avg_loss = sum(running_loss) / len(running_loss)

        # Append stats to log list
        log.append([epoch+1, avg_loss, round(train_acc, 5), round(val_acc, 5)])

        # Save best model
        if vcorrect/vtotal > best:
            best = vcorrect/vtotal
            torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))

        # Print results
        print(f"  Average Loss: {round(avg_loss, 5)}    ", end="")
        print(f"Training accuracy: {tcorrect} / {ttotal} = {round(100*train_acc, 3)} %    ", end="")
        print(f"Validation accuracy: {vcorrect} / {vtotal} = {round(100*val_acc, 3)} %\n")

    # Log to .csv
    log_df = pd.DataFrame(log, columns=['Epoch', 'Avg Train Loss', 'Train Acc', 'Val Acc'])
    log_df.to_csv(os.path.join(save_path, 'log.csv'), index=False)

    print('Finished Training')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, default='', help='train_args.json path')

    opt = parser.parse_args()
    with open(opt.args_path, 'r') as f:
        args = json.load(f)

    if not os.path.isdir(args['save_path']):
        os.mkdir(args['save_path'])

    with open(args['save_path'] + '\\hyp.json', 'w') as g:
        json.dump(args, g, indent=2)
    return args


if __name__ == "__main__":
    args = parse_opt()
    train(args)