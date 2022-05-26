import os
import torch.nn as nn
import albumentations as A
import torchvision.models as models

from matplotlib import pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(name, classes, pretrained):
    try:
        """
        Current models that can be used...
        Alexnet, Resnet18, Squeezenet

        Add models here in a similar fashion...

        if name == "model"
            model = models.model()
            model.fc = nn.Linear(input, output)
        return model
        """

        if name == "alexnet":
            model = models.alexnet(pretrained=pretrained)
            model.classifier[6] = nn.Linear(4096, classes)
            return model

        elif name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(512, classes)
            return model

        elif name == "squeezenet":
            model = models.squeezenet1_1(pretrained=pretrained)
            model.classifier._modules["1"] = nn.Conv2d(512, classes, kernel_size=(1, 1))
            model.num_classes = classes
            return model

        elif name == "densenet":
            model = models.densenet121(pretrained=pretrained)
            model.classifier = nn.Linear(1024, classes)
            return model

        else:
            pass

    except:
        print("Model name not valid. Try again with valid model name. Models can be added in the create_model function of utils.py.")


def augment_image(image):
    transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, p=0.1),        
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.1)
        ])
        
    augmented_image = transform(image=image)['image']
    return augmented_image


def create_plot(i, save_path, x, y, cols):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_axes([0.1,0.1,0.75,0.75]) # axis starts at 0.1, 0.1
    ax.set_title(cols[i] + " per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(cols[i])
    ax.plot(x, y)
    ax.set_ylim(ymin=0)
    fig.savefig(os.path.join(save_path, cols[i].lower().replace(" ", "_") + '.jpg'))
    plt.close(fig)