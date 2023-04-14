import os
import warnings
import torch, torchvision
from torchvision import transforms
from sklearn.model_selection import KFold
import CNN as CNN
import train as train
import test as test
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

warnings.filterwarnings('always')
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Pre-bias for calaculating K-Fold validation on dataset before bias
# DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Pre-bias')


## For Bias evaluation
# Post-bias/Dataset-Age/0-40 for calaculating K-Fold validation on age bias in range 0-40
# DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias/Dataset-Age/0-40')
# Post-bias/Dataset-Gender/Age/41-100 for calaculating K-Fold validation on age bias in range 41-100
# DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias/Dataset-Age/41-100')
# Post-bias/Dataset-Gender/Male for calaculating K-Fold validation on gender bias for Male
# DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias/Dataset-Gender/Male')
# Post-bias/Dataset-Gender/Female for calaculating K-Fold validation on gender bias for Female
# DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias/Dataset-Gender/Female')


## For Bias-elimination evaluation
# Post-bias-fix/Dataset-Age/0-40 for calaculating K-Fold validation on age bias in range 0-40
# DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias-fix/Dataset-Age/0-40')

# Post-bias-fix/Dataset-Gender/Age/41-100 for calaculating K-Fold validation on age bias in range 41-100
# DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias-fix/Dataset-Age/41-100')

# Post-bias-fix/Dataset-Gender/Male for calaculating K-Fold validation on gender bias for Male
# DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias-fix/Dataset-Gender/Male')

# Post-bias-fix/Dataset-Gender/Female for calaculating K-Fold validation on gender bias for Female
# DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias-fix/Dataset-Gender/Female')

## For Bias evaluation
# Post-bias-fix-combined for calaculating K-Fold validation on the entire dataset post bias elimination
DATASET_DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias-fix-combined')

#CLASSES = ['Cloth-Mask', 'FFP2-Mask', 'No-Mask', 'Surgical-Mask']
IMG_SIZE = 128
BATCH_SIZE = 32
SHUFFLE = True
EPOCH = 15
MEAN = [0.6616, 0.6203, 0.6006]
STD = [0.3089, 0.3153, 0.3261]

# Applying Transformation
transforms = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),  # resizing every image into the desired values
    transforms.RandomHorizontalFlip(),  # Flips images horizontally with a probability of 0.5
    transforms.ToTensor(),  # size normalization and conversation to tensor
    transforms.Normalize(mean=MEAN, std=STD)
])

data = torchvision.datasets.ImageFolder(DATASET_DIR, transform=transforms)
CLASSES = data.classes
X = data.imgs
Y = data.targets
k_fold = KFold(n_splits=10, shuffle=True)
itr = 0

accuracy_list =[]


# Python program to get average of a list
def avg(lst):
    return sum(lst) / len(lst)


for train_index, test_index in k_fold.split(data):
    itr += 1
    print(itr)
    print('--------------------------------------------------------')
    train_data = []
    test_data = []

    # python 10fold.py

    for i in range(len(data)):
        if i in test_index:
            test_data.append(data[i])
        else:
            train_data.append(data[i])

    # print(len(train_data))
    # print(len(test_data))
    x_train_fold = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    x_test_fold = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    model = CNN.AlexNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train.trainModel(model, x_train_fold, EPOCH, optimizer, DEVICE)
    # Saving model
    str = 'Model/model' + str(itr) + '.pth'
    MODEL_FILEPATH = os.path.join(os.path.abspath(os.curdir), str)
    CNN.saveModelAlexNet(model, optimizer, MODEL_FILEPATH)
    
    accuracy_list.append(test.testModel(model, x_test_fold, DEVICE))

    with torch.no_grad():
        labels_N_prediction = test.getLabelsNPrediction(model, x_test_fold, DEVICE)

    print(classification_report(labels_N_prediction[0], labels_N_prediction[1], target_names=CLASSES, zero_division=1))
    conf_matrix = confusion_matrix(labels_N_prediction[0], labels_N_prediction[1])
    print(conf_matrix)

print("\n Average accuracy across 10 folds: ", avg(accuracy_list))
print("\n Maximum accuracy across 10 folds: ", max(accuracy_list))
print("\n Minimum accuracy across 10 folds: ", min(accuracy_list))