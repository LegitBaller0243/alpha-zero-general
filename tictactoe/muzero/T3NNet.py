import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import Action

class T3NNet:
    def __init__(self):
        ##create architecture
        self.action_size = 9
        self.hidden_size = 128

        # Convert Input Borads into Tensors
        
        self.representations = RepresentationsNet(self.hidden_size)
        self.predictions = PredictionsNet(self.action_size, self.hidden_size)
        #p.output.length = first input for d
        self.dynamics = DynamicsNet(self.action_size, self.hidden_size)
    def getnets(self):
        return self.representations, self.predictions, self.dynamics


class RepresentationsNet(nn.Module):
    def __init__(self, hidden):
        super(RepresentationsNet, self).__init__()
        ##will run parallel in batch_size based on input tensor
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128*3*3, hidden)
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 3:  # If x is (3,3), add batch and channel dims
            x = x.unsqueeze(0) # Shape: (1,1,3,3)

        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
class PredictionsNet(nn.Module):
    def __init__(self, action_size, hidden_size):
        super(PredictionsNet, self).__init__()
        ## takes a hidden state and ouputs a policy and value prediction
        ## given some hidden state which is 128 dimensions -> create a policy and a value
        self.l1 = nn.Linear(hidden_size, 64)
        self.policy = nn.Linear(64, action_size)
        self.value = nn.Linear(64, 1)
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.l1(x))
        policy_logits = F.softmax(self.policy(x), dim=-1)
        value = torch.tanh(self.value(x))
        return policy_logits, value
class DynamicsNet(nn.Module):
    def __init__(self, action_size, hidden_size):
        super(DynamicsNet, self).__init__()
        ##given hidden state + action - return new hidden state + immediate reward
        self.l1 = nn.Linear(hidden_size+1, hidden_size)
        self.reward = nn.Linear(hidden_size, 1)
        self.l2 = nn.Linear(hidden_size, hidden_size)
    def forward(self, hidden, action):
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.tensor(hidden, dtype=torch.float32)
        
        # Only needed for self-play (single actions)
        if isinstance(action, (int, np.int64, Action)):  # Add np.int64 to types we handle
            action_value = action.index if isinstance(action, Action) else float(action)  # Convert numpy.int64 to float
            action = torch.tensor([[action_value]])
            action = action.expand(hidden.size(0), -1)  # Match batch size
        else:
            # During training - already batched correctly
            action = action.float().view(-1, 1)

        x = torch.cat([hidden, action], dim=-1)
        x = F.relu(self.l1(x))
        new_hidden = F.relu(self.l2(x))
        reward = self.reward(x)
        return new_hidden, reward



    