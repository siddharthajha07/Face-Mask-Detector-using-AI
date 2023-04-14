import os
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms

DIR = os.path.join(os.path.abspath(os.curdir), 'Dataset/Post-bias/Dataset-Gender/Male')
IMG_SIZE = 128
BATCH_SIZE = 32
SHUFFLE = False
#MEAN = [0.485, 0.456, 0.406]
#STD = [0.229, 0.224, 0.225]

MEAN = [0.6616, 0.6203, 0.6006]
STD = [0.3089, 0.3153, 0.3261]
#print(torch.__version__)


# Applying Transformation
transforms = transforms.Compose([
        transforms.Resize([IMG_SIZE, IMG_SIZE]),  # resizing every image into the desired values
        transforms.RandomHorizontalFlip(0.5),  # Flips images horizontally with a probability of 0.5
        #transforms.RandomResizedCrop(IMG_SIZE),
        #transforms.RandomAdjustSharpness(sharpness_factor=0,p=0.1),
        transforms.ToTensor(), # size normalization and conversation to tensor
        transforms.Normalize(mean=MEAN, std=STD)
        ])

# Loads the images and labels from the specified folder and applies the given transformation
data = torchvision.datasets.ImageFolder(DIR, transform=transforms)
CLASSES = data.classes
#print(CLASSES)

"""
def mean_std(loader):
    images, labels = next(iter(loader))
    # shape of images = [b,c,w,h]
    mean_images, std_images = images.mean([0, 2, 3]), images.std([0, 2, 3])
    return mean_images, std_images


data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
MEAN, STD = mean_std(data_loader)
print("mean and std: \n", MEAN, STD)
"""

trainDataSize = round(len(data) * 0.80)
testDataSize = round(len(data) * 0.20)

# Splitting data into test and train
trainData, testData = torch.utils.data.random_split(data, [trainDataSize, testDataSize])

# Loading train data into a generator which provides images in a batch
def getTrainingData():
    return torch.utils.data.DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=SHUFFLE)


# Loading test data into a generator which provides images in a batch
def getTestingData():
    return torch.utils.data.DataLoader(testData, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

# Testing
def displayImage():
    images, labels = next(iter(getTrainingData()))
    print(images.shape, labels.shape)
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
    # plt.imshow(images[6].permute(1, 2, 0))
    plt.show()
displayImage()




