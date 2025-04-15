from torch import nn
import torch
from emotion.motor_prediction.utils import get_device

device = get_device()
class GraceModel(nn.Module):
    def __init__(self, feature_type='face_embed'):
        super(GraceModel, self).__init__()
        self.feature_type = feature_type
        
        if feature_type == 'ldmk':
            # Smaller model for landmarks
            self.fc1 = nn.Linear(136, 128)
            self.fc2 = nn.Linear(128, 256)
            self.fc5 = nn.Linear(256, 512)
            self.fc3 = nn.Linear(512, 64)
            self.fc4 = nn.Linear(64, 26)
        else:  # face_embed
            # Larger model for face embeddings
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 512)
            self.fc5 = nn.Linear(512, 1024)
            self.fc3 = nn.Linear(1024, 128)
            self.fc4 = nn.Linear(128, 26)
            
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x