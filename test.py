import torch
import preprocessing as preprocess
from torch.nn.functional import cross_entropy
import itertools
import numpy as np
import matplotlib.pyplot as plt

def testModel(model, testing_data, DEVICE):

    testingLoss = 0
    correctPrediction = 0
    dataSize = 0

    for batch in testing_data:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            dataSize += len(images)
            prediction = model(images)
            # print(prediction.shape)
            # print(prediction.argmax(dim=1))
            testingLoss += cross_entropy(prediction, labels).item()
            correctPrediction += (prediction.argmax(dim=1) == labels).sum().item()

    accuracy = correctPrediction/dataSize
    testingLoss = testingLoss/dataSize

    print('\nTesting:')
    print(f"Correct prediction: {correctPrediction}/{dataSize} and accuracy: {accuracy} and loss: {testingLoss}")
    return accuracy



def getLabelsNPrediction(model, data, DEVICE):

    allLabels = []
    allPrediction = []

    for batch in data:
        images, labels = batch
        images = images.to(DEVICE)
        prediction = model(images).to(torch.device("cpu")).argmax(dim=1).detach().numpy()
        labels = labels.to(torch.device("cpu")).detach().numpy()
        allPrediction = np.append(allPrediction, prediction)
        allLabels = np.append(allLabels, labels)

    return [allLabels, allPrediction]


def displayConfusionMatrix(conf_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(conf_matrix)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

