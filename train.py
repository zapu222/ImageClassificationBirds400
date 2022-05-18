import os
import json
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from torch import nn, optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset import Birds400
from utils import create_model
from utils import count_parameters


def train(args):
    # Parameters
    data_path, save_path, model_name, classes, img_size, batch_size, workers, lr, epochs, device = \
        args['data_path'], args['save_path'], args['model'], args['num_classes'], args["img_size"],\
            args["batch_size"], args['workers'], args['lr'], args['epochs'], args['device']

    # Datasets
    trainset = Birds400(data_path, task="train", img_size=img_size)
    valset = Birds400(data_path, task="valid")

    # Train and Valid loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=workers, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, num_workers=workers, shuffle=True)

    # Model
    model = create_model(model_name, classes)
    model.to(device)
    print(f"\nModel: {model_name.upper()}\nTrainable parameters: {count_parameters(model)}")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Logging
    log = []
    best = 0
    if not os.path.isdir(os.path.join(save_path, "weights\\")):
        os.mkdir(os.path.join(save_path, "weights\\"))

    # Begin training...
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} of {epochs}")
        running_tloss = []   # training losses
        running_vloss = []   # validation losses

        # Training
        tcorrect, ttotal = 0, 0
        for _, data in enumerate(tqdm(trainloader, desc='Trainset: ', bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
            inputs, labels = data   # images and labels

            if device == "cuda":
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()   # zero gradients

            outputs = model(inputs)   # model outputs

            loss = criterion(outputs, labels.squeeze())   # calculate loss
            loss.backward()   # backpropigation
            optimizer.step()   # optimize

            running_tloss.append(loss.item())   # append loss 

            _, labels = torch.max(labels.squeeze().data, 1)   # labels indices
            _, predicted = torch.max(outputs.data, 1)   # predicted label indices

            _, top_5 = torch.topk(outputs, 5)

            ttotal += labels.size(0)   # total predictions
            tcorrect += (predicted == labels).sum().item()   # correct predictions
    
        # Validation
        vcorrect, vtotal = 0, 0
        with torch.no_grad():
            for _, data in enumerate(tqdm(valloader, desc='Valset:   ', bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
                inputs, labels = data   # images and labels
                
                if device == "cuda":
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)   # model outputs

                loss = criterion(outputs, labels.squeeze())   # calculate loss
                running_vloss.append(loss.item())   # append loss 

                _, labels = torch.max(labels.squeeze().data, 1)   # labels indices
                _, predicted = torch.max(outputs.data, 1)   # predicted label indices

                vtotal += labels.size(0)   # total predictions
                vcorrect += (predicted == labels).sum().item()   # correct predictions

        # Save model
        torch.save(model.state_dict(), os.path.join(save_path, "weights\last.pth"))

        # Training and val accuracy
        train_acc = tcorrect/ttotal
        val_acc = vcorrect/vtotal

        # Save best model
        if vcorrect/vtotal > best:
            best = vcorrect/vtotal
            torch.save(model.state_dict(), os.path.join(save_path, "weights\\best.pth"))

        # Average training and valid loss for epoch
        avg_tloss = sum(running_tloss) / len(running_tloss)
        avg_vloss = sum(running_vloss) / len(running_vloss)

        # Append stats to log list
        log.append([epoch+1, avg_tloss, avg_vloss, round(train_acc, 5), round(val_acc, 5)])
        cols = ['Epoch', 'Avg Train Loss', 'Avg Val Loss', 'Train Acc@1', 'Val Acc@1']

        # Log to csv
        log_df = pd.DataFrame(log, columns=cols)
        log_df.to_csv(os.path.join(save_path, 'log.csv'), index=False)

        # Save plots
        for i in range(1,5):
            x = [item[0] for item in log]
            y = [item[i] for item in log]

            fig = plt.figure(figsize=(15,10))
            ax = fig.add_axes([0.1,0.1,0.75,0.75]) # axis starts at 0.1, 0.1
            ax.set_xlabel("Epoch")
            ax.set_ylabel(cols[i])
            ax.plot(x, y)
            fig.savefig(os.path.join(save_path, cols[i].lower().replace(" ", "_") + '.jpg'))

        # Print results
        print(f"Average Loss: {round(avg_tloss, 5)}    ", end="")
        print(f"Training accuracy: {tcorrect} / {ttotal} = {round(100*train_acc, 3)} %    ", end="")
        print(f"Validation accuracy: {vcorrect} / {vtotal} = {round(100*val_acc, 3)} %")

    print('\nFinished Training')


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