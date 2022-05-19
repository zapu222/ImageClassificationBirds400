# Birds 400 Image Classification - Zachary Pulliam

This code was written in order to test various computer vision techniques on the Birds 400 dataset found at https://www.kaggle.com/datasets/gpiosenka/100-bird-species. 

Upon downloading the dataset, in class_dict.csv, "BLACK & YELLOW &nbsp;BROADBILL" at B60 should be changed to "BLACK & YELLOW BROADBILL," and the same for the folders
'BLACK & YELLOW &nbsp;BROADBILL' within the train, valid, and test folders. 

The scripts data_stats.py and disp_imgs.py can be used to get a better understanding of the size of the dataset and the images within the dataset.

Computer vision models can be trained using train.py, which has hyperparamter option in the hyps/train_args.json file. Given that we wish to explore different models,
different computer vision models can easily be selected by changing "model" in train_args.json. New models can be added in the function create_model within utils.py.
In order to understand architectures, models can be summarized before or after training using tools/model_summary.py.

Models will be saved to the folder defined as "save_path" in train_args.json. Here the model.pth file is saved along with stats and graphs reposrting the model's 
perfomance. Both the best and last model will be saved.

Models can then be tested using test.py according to the hyperparameter defined in hyps/test_args.json. 
