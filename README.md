# Birds 400 Image Classification - Zachary Pulliam

This code was written in order to test various PyTorch image classification techniques on the Birds 400 dataset found at https://www.kaggle.com/datasets/gpiosenka/100-bird-species. Upon downloading the dataset, in class_dict.csv, "BLACK & YELLOW &nbsp;BROADBILL" at B60 should be changed to "BLACK & YELLOW BROADBILL" (one space rather than two).

The scripts data_stats.py and disp_imgs.py can be used to gain a better understanding of the size of the dataset as well as display example images within the dataset from the 400 differen classes (species).

Different PyTorch vision models can be trained using train.py. The hyperparamter options for training are located in the in the hyps/train_args.json file. Given that we wish to explore different models, different PyTroch models can easily be selected by changing "model" in train_args.json. For example, setting '"model": resnet18' will use Resnset18 for training. Torchvision models that are not already included in this repo can be added in the function *create_model* within utils.py. In order to gain a better understanding of different models and their corresponding architectures, models can be summarized before or after training using tools/model_summary.py. More informaton on each Torchvision model can be found at https://pytorch.org/vision/stable/models.html. 

Models will be saved to the folder defined as "save_path" in train_args.json. Ideally save paths would look like "ImageClassificationBirds400\models\resnet18." In this location, the model.pth files are saved in the weights subfolder along with stats and graphs reporting the model's perfomance. Both the best and last model will be saved in the weights subfolder.

Models can then be tested using test.py according to the hyperparameters defined in hyps/test_args.json. The variables "model_path" and "model" should specify the path to the model.pth file and the type of model that the model.pth file represents.