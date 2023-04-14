import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 5, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride= 1, padding= 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=984, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=984, out_channels=4048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=4048, out_channels=2024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=2024, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features= 12544, out_features= 512),
            nn.ReLU(),

            nn.Linear(in_features= 512, out_features= 256),
            nn.ReLU(),

            nn.Linear(in_features=256 , out_features=5)
        )

    def forward(self, tensor):
        tensor = self.conv_layer(tensor)
        tensor = tensor.view(tensor.size(0), -1)
        tensor = self.fc_layer(tensor)
        return tensor


# saving the model into the folder model
def saveModelAlexNet(model, optimizer, MODEL_FILEPATH):
    model_info = {
        'model': AlexNet(),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(model_info, MODEL_FILEPATH)

# loading the model
def loadModel(MODEL_FILEPATH, DEVICE):
    model_info = torch.load(MODEL_FILEPATH)
    model = model_info['model'].to(DEVICE)
    model.load_state_dict(model_info['state_dict'])
    return model