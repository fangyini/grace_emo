from torch import nn

class GraceModel(nn.Module):
    def __init__(self, node1=256, node2=512, node3=1024, node4=128):
        super(GraceModel, self).__init__()
        self.fc1 = nn.Linear(136, node1)
        self.fc2 = nn.Linear(node1, node2)
        self.fc5 = nn.Linear(node2, node3)
        self.fc3 = nn.Linear(node3, node4)
        self.fc4 = nn.Linear(node4, 26)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x