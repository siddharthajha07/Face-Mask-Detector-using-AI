import os
import torch
import preprocessing as preprocess
import CNN as CNN
import train as train
import test as test
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
MODEL_FILEPATH = os.path.join(os.path.abspath(os.curdir), 'Model/model.pth')
EPOCH = 15


# Loading training and testing preprocessed data
trainingData = preprocess.getTrainingData()
testingData = preprocess.getTestingData()


#Loading the model and uploading it to the system device (CPU or GPU)
# model = CNN.NeuralNet().to(DEVICE)
model = CNN.AlexNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print(summary(model, input_size=(3, preprocess.IMG_SIZE, preprocess.IMG_SIZE)))


# Training and saving
train.trainModel(model, trainingData, EPOCH, optimizer, DEVICE)
CNN.saveModelAlexNet(model, optimizer, MODEL_FILEPATH)


# loading and testing
model = CNN.loadModel(MODEL_FILEPATH, DEVICE)
test.testModel(model, testingData, DEVICE)


# Evaluation
with torch.no_grad():
    labelsNPrediction = test.getLabelsNPrediction(model, testingData, DEVICE)

print(classification_report(labelsNPrediction[0], labelsNPrediction[1], target_names = preprocess.CLASSES))
conf_matrix = confusion_matrix(labelsNPrediction[0], labelsNPrediction[1])
plt.figure(figsize=(10, 10))
test.displayConfusionMatrix(conf_matrix, preprocess.CLASSES)
