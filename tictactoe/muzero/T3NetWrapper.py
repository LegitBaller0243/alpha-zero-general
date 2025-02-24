from T3NNet import T3NNet
import torch
import torch.optim as optim
from utils import *
import os
from helpers import NetworkOutput

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class T3NetWrapper:
    def __init__(self):
        self.nnet = T3NNet()
        self.representations, self.predictions, self.dynamics = self.nnet.getnets()

        self.training_step_count = 0

    def initial_inference(self, image):
    # representation + prediction function

        #turn image into a tensor
        hidden_state = self.representations(image)
        policy_logits, value = self.predictions(hidden_state)
        return NetworkOutput(value, torch.tensor(0.0), policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action):
        # dynamics + prediction function
        new_hidden, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.predictions(new_hidden)
        return NetworkOutput(value, reward, policy_logits, new_hidden)

    def get_weights(self):
        # Returns the weights of this network as a flat list of parameters.
        return (
            list(self.representations.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.predictions.parameters())
        )
    def training_steps(self):
        # How many steps / batches the network has been trained for.
        return self.training_step_count
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        try:
            filepath = os.path.join(folder, filename)
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save({
                "representation": self.representations.state_dict(),
                "dynamics": self.dynamics.state_dict(),
                "prediction": self.predictions.state_dict()
            }, filepath)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")


    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        checkpoint = torch.load(filepath, map_location='cuda' if args.cuda else 'cpu')
        self.representations.load_state_dict(checkpoint['representation'])
        self.dynamics.load_state_dict(checkpoint['dynamics'])
        self.predictions.load_state_dict(checkpoint['prediction'])
