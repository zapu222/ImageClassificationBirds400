import torch.nn as nn
import albumentations as A
import torchvision.models as models


def augment_image(image):
    transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, p=0.1),        
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.1)
        ])
        
    augmented_image = transform(image=image)['image']
    return augmented_image


def create_model(name, classes, pretrained):
    try:
        # Add models here in a similar fashion...
        """
        if name == "model"
            model = models.model()
            model.fc = nn.Linear(input, output)
        return model
        """

        if name == "alexnet":
            model = models.alexnet(pretrained=pretrained)
            model.fc = nn.Linear(512, classes)
        if name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(512, classes)
        return model

    except:
        print("Model name not valid. Try again with valid model name. Models can be added in the create_model function of utils.py.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)