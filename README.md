# COMP6721ProjectAK_07
Get added to our private github repo and download the repository

https://github.com/akshaydhabale/COMP6721ProjectAK_07.git

Model and Dataset Link:

https://drive.google.com/drive/folders/1YDkx5H-YqduEOc4ri0-YjerDiSqJVbwh?usp=sharing

```

- Install Python 3.8 and pip. 
- Build and activate a python virtual environment
- Run the following command

```
pip install requirements.txt
```

To run the project: 
```
python app.py
```

To run K-fold: 
```
python kfold.py
```

#### File description

- app.py : Main file that imports other classes and methods defined by us. It first loads the preprocessed data, then builds the model and optimizer, trains the model and saves it and then the model is loaded again and evaluated

- preprocessing.py : Loads the dataset, applies various transformation, splits into training-testing and loads data into batches

- CNN.py : Contains a class that builds, trains and returns a CNN network. Also there are methods to load and save a method.

- train.py : trains the model on training data

- test.py : Evaluates the model on testing data and generates the confusion matrix

- demo.py : This file can be used to predict the label for any image using the trained model.

- kfold.py : This file can be used to do kfold evaluation and generates the confusion matrix for each fold.
