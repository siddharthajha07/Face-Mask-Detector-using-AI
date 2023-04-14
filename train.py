import preprocessing as preprocess
from torch.nn.functional import cross_entropy
import matplotlib.pyplot as plt

def trainModel(model, training_data, EPOCH, optimizer, DEVICE):
    # performance measurement array
    print(DEVICE)
    training_loss = []
    training_accuracy = []
    for epoch in range(EPOCH):
        trainingLoss = 0
        correctPrediction = 0
        dataSize = 0
        for batch in training_data:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            dataSize += len(images)

            predictions = model(images)

            loss = cross_entropy(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainingLoss += loss.item()
            correctPrediction += (predictions.argmax(dim=1) == labels).sum().item()
        accuracy = correctPrediction / dataSize
        trainingLoss = trainingLoss / dataSize
        training_loss.append(trainingLoss)
        training_accuracy.append(accuracy)
        print(f"Epoch {epoch + 1}: Correct Prediction: {correctPrediction}/{dataSize} and accuracy: {accuracy} and loss: {trainingLoss}")

    # plot the learning characteristics
    print('Accuracy and Loss Curve')
    f = plt.figure(figsize=(10, 10))
    plt.plot(training_accuracy, label="Training Accuracy")
    plt.plot(training_loss, label="Training Loss")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.title('Accuracy and Loss Curve')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])



