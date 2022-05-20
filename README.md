# Birds 400 Image Classification - Zachary Pulliam

This code was written in order to test various computer vision techniques on the Birds 400 dataset found at https://www.kaggle.com/datasets/gpiosenka/100-bird-species. 

Upon downloading the dataset, in class_dict.csv, "BLACK & YELLOW &nbsp;BROADBILL" at B60 should be changed to "BLACK & YELLOW BROADBILL" (one space rather than two).

The scripts data_stats.py and disp_imgs.py can be used to get a better understanding of the size of the dataset and the images within the dataset.

Computer vision models can be trained using train.py, which has hyperparamter option in the hyps/train_args.json file. Given that we wish to explore different models,
different computer vision models can easily be selected by changing "model" in train_args.json. New torchvision models can be added in the function create_model within utils.py. In order to gain a better understanding of different models and their corresponding architectures, models can be summarized before or after training using tools/model_summary.py.

Models will be saved to the folder defined as "save_path" in train_args.json. Ideally save paths would look like "...\models\resnet18." Here the model.pth file is saved in the weights subfolder along with stats and graphs reporting the model's perfomance. Both the best and last model will be saved in the weights subfolder.

Models can then be tested using test.py according to the hyperparameters defined in hyps/test_args.json. 
