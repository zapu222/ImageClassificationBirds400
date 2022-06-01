import os
import cv2
import sys
import json
import torch
import inspect
import argparse
import torch.nn.functional as F

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

from dataset import Birds400
from utils import create_model

def run(args):
    image, data, model, model_path, device = args.image, args.data, args.model, args.model_path, args.device

    with open(os.path.join(model_path, 'hyp.json'), 'r') as f:
        model_args = json.load(f)

    dataset = Birds400(data, task="test")

    model = create_model(model, model_args['num_classes'])
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best.pth')))
    model.eval()

    with torch.no_grad():
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transform = ToTensor()
        x = transform(image)
        x = x.unsqueeze(0) 
        input = x
        
        if device == "cuda":
            input = input.cuda() 

        outputs = model(input)
        softmax = F.softmax(outputs, dim=1)
        softmax = torch.flatten(softmax)

        confs, indxs = torch.topk(softmax, 3)

        names = [dataset.indices[i.item()] for i in indxs]
        confs = confs.tolist()


        _, ax = plt.subplots(1)
        ax.axis("off")
        ax.imshow(image)
        ax.axis('off')
        padding = 5
        ax.annotate(
            text = names[0]+': '+str(round(confs[0]*100,1))+'%\n'+names[1]+': '+str(round(confs[1]*100,1))+'%\n'+names[2]+': '+str(round(confs[2]*100,1))+'%', 
            fontsize = 8,

            xy=(0, 196), 
            xytext=(padding-1, -(padding-1)), 
            textcoords = 'offset pixels',
            bbox=dict(facecolor='white', alpha=1, pad=padding),
            va='top',
            ha='left',
            )
        plt.show()

        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='path to image')
    parser.add_argument('--data', type=str, help='path to BIRDS_400')
    parser.add_argument('--model', type=str, help='model type')
    parser.add_argument('--model-path', type=str, help='path to model folder')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()
    run(args)