from torch import nn
from pytorch_face_landmark.models.mobilefacenet import MobileFaceNet
import torch
from emotion.motor_prediction.utils import get_device

device = get_device()
class GraceModel(nn.Module):
    def __init__(self, node1=256, node2=512, node3=1024, node4=128, image_embed=False):
        super(GraceModel, self).__init__()
        self.image_embed = image_embed
        if image_embed is False:
            self.fc1 = nn.Linear(136, node1)
            self.fc2 = nn.Linear(node1, node2)
        else:
            self.model = MobileFaceNet([112, 112], 136)
            checkpoint = torch.load('grace_emo/pytorch_face_landmark/checkpoint/mobilefacenet_model_best.pth.tar',
                                    map_location=device)
            self.model.load_state_dict(checkpoint['state_dict'])
            #model.eval() # try finetune/freeze
        self.fc5 = nn.Linear(node2, node3)
        self.fc3 = nn.Linear(node3, node4)
        self.fc4 = nn.Linear(node4, 26)
        self.relu = nn.LeakyReLU()

    def forward(self, x, images):
        if self.image_embed is False:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
        else:
            x = self.model(images, returnFeature=True)
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x