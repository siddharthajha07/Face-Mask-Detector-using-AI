import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
import CNN
import os
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODEL_FILEPATH = os.path.join(os.path.abspath(os.curdir), 'Model/model.pth')
#DEMO_FILEPATH = os.path.join(os.path.abspath(os.curdir), 'Demo')

classDict = {0: 'Cloth-Mask', 1: 'FFP2-Mask', 2: 'FFP2-Valve-Mask', 3: 'No-Mask', 4: 'Surgical-Mask'}

import preprocessing
from PIL import Image

img = Image.open('D:/Concordia Academics/Winter 2022/COMP 6721/COMP6721ProjectAK_07/Demo/64_2.jpg')
#img = Image.open(DEMO_FILEPATH)
plt.imshow(img)
plt.show()
image = preprocessing.transforms(img)
model = CNN.loadModel(MODEL_FILEPATH, DEVICE)
#print(model)
#unsqueeze to add channel as Conv layer except four dimension
image = image.to(DEVICE).unsqueeze(0)
prediction = model(image).to(torch.device("cpu")).argmax(dim=1).detach().numpy()
#print(prediction)
# numpy array to number
predClass = prediction[0]
print("Predicted Class for Image:", classDict[predClass])




