import os
import json
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import Birds400
from utils import count_parameters, create_model, create_plot


def train(args):
    # Parameters
    data_path, save_path, model_type, pretrained, classes, img_size, augment, batch_size, workers, lr, epochs, device = \
        args['data_path'], args['save_path'], args['model_type'], args['pretrained'], args['num_classes'], args["img_size"],\
            args['augment'], args["batch_size"], args['workers'], args['lr'], args['epochs'], args['device']

    print(f'\nBeginning Training - Models and metrics will be saved to: {save_path}')

    # Datasets
    trainset = Birds400(data_path, task="train", img_size=img_size, augment=augment)
    valset = Birds400(data_path, task="valid")

    # Train and Valid loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=workers, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, num_workers=workers, shuffle=True)

    # Model
    model = create_model(model_type, classes, pretrained)
    model.to(device)
    print(f"\nModel: {model_type.upper()}\nTrainable parameters: {count_parameters(model)}")
    if not os.path.isdir(os.path.join(save_path, "weights\\")):
        os.mkdir(os.path.join(save_path, "weights\\"))

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Logging
    log = []
    best = 0
    cols = ['Epoch', 'Avg Train Loss', 'Avg Val Loss', 'Train Acc@1', 'Train Acc@5', 'Val Acc@1', 'Val Acc@5']

    # Begin training...
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} of {epochs}")
        running_tloss = []   # training losses
        running_vloss = []   # validation losses

        # Training
        tcorrect_1, tcorrect_5, ttotal = 0, 0, 0
        for _, data in enumerate(tqdm(trainloader, desc='Trainset', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
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
            _, top_1 = torch.max(outputs.data, 1)   # top prediction
            _, top_5 = torch.topk(outputs, 5)   # top 5 predictions

            ttotal += labels.size(0)   # total predictions
            tcorrect_1 += (top_1 == labels).sum().item()   # correct predictions @ 1
            for i in range(top_5.shape[0]):
                if labels[i] in top_5[i]:
                    tcorrect_5 += 1   # correct predictions @ 5
    
        # Validation
        vcorrect_1, vcorrect_5, vtotal = 0, 0, 0
        with torch.no_grad():
            for _, data in enumerate(tqdm(valloader, desc='Validset', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
                inputs, labels = data   # images and labels
                
                if device == "cuda":
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)   # model outputs

                loss = criterion(outputs, labels.squeeze())   # calculate loss
                running_vloss.append(loss.item())   # append loss 

                _, labels = torch.max(labels.squeeze().data, 1)   # labels indices
                _, top_1 = torch.max(outputs.data, 1)   # top prediction
                _, top_5 = torch.topk(outputs, 5)   # top 5 predictions

                vtotal += labels.size(0)   # total predictions
                vcorrect_1 += (top_1 == labels).sum().item()   # correct predictions
                for i in range(top_5.shape[0]):
                    if labels[i] in top_5[i]:
                        vcorrect_5 += 1   # correct predictions @ 5

        # Save model
        torch.save(model.state_dict(), os.path.join(save_path, "weights\last.pth"))

        # Training and val accuracy
        train_acc_1 = tcorrect_1/ttotal
        train_acc_5 = tcorrect_5/ttotal
        val_acc_1 = vcorrect_1/vtotal
        val_acc_5 = vcorrect_5/vtotal

        # Save best model
        if vcorrect_1/vtotal > best:
            best = vcorrect_1/vtotal
            torch.save(model.state_dict(), os.path.join(save_path, "weights\\best.pth"))

        # Average training and valid loss for epoch
        avg_tloss = sum(running_tloss) / len(running_tloss)
        avg_vloss = sum(running_vloss) / len(running_vloss)

        # Append stats to log list
        log.append([epoch+1, avg_tloss, avg_vloss, round(train_acc_1, 5), round(train_acc_5, 5), round(val_acc_1, 5), round(val_acc_5, 5)])

        # Log to csv
        log_df = pd.DataFrame(log, columns=cols)
        log_df.to_csv(os.path.join(save_path, 'log.csv'), index=False)

        # Save plots
        for i in range(1,7):
            x = [item[0] for item in log]
            y = [item[i] for item in log]
            create_plot(i, save_path, x, y, cols)

        # Print results
        print(f"Average Loss: {round(avg_tloss, 5)}    ", end="")
        print(f"Training Acc@1: {tcorrect_1} / {ttotal} = {round(100*train_acc_1, 3)} %    ", end="")
        print(f"Validation Acc@1: {vcorrect_1} / {vtotal} = {round(100*val_acc_1, 3)} %")

    print(f'\nFinished Training - Models and metrics saved to: {save_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, default='', help='train_args.json path')

    opt = parser.parse_args()
    with open(opt.args_path, 'r') as f:
        args = json.load(f)
    save_path = args['save_path']

    i = 2
    while True:
        if not os.path.isdir(os.path.join(save_path)):
            os.mkdir(os.path.join(save_path))
            break
        else:
            save_path = args['save_path'] + "_" + str(i)
            i += 1
    args['save_path'] = save_path

    with open(args['save_path'] + '\\hyp.json', 'w') as g:
        json.dump(args, g, indent=2)
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)