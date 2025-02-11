import collections
import math
import typing
from typing import Dict, List, Optional

import numpy as np
from Game import TicTacToeGame as T3Game
from utils import Action
import torch

MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value
  
class MuZeroConfig(object):
  def __init__(self,
               action_space_size: int,
               max_moves: int,
               discount: float,
               dirichlet_alpha: float,
               num_simulations: int,
               batch_size: int,
               td_steps: int,
               num_actors: int,
               lr_init: float,
               lr_decay_steps: float,
               visit_softmax_temperature_fn,
               num_episodes,
               known_bounds: Optional[KnownBounds] = None):
    ### Self-Play
    self.action_space_size = action_space_size
    self.num_actors = num_actors

    self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount
    self.num_iterations = 5

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.training_steps = 2
    self.checkpoint_interval = 5
    self.num_episodes = num_episodes
    self.window_size = 10
    self.batch_size = batch_size
    self.num_unroll_steps = 3
    self.td_steps = td_steps

    self.weight_decay = 1e-4
    self.momentum = 0.9

    # Exponential learning rate schedule
    self.lr_init = lr_init
    self.lr_decay_rate = 0.1
    self.lr_decay_steps = lr_decay_steps

  def new_game(self):
    return T3Game()



class NetworkOutput(typing.NamedTuple):
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: Dict[Action, float]  # or torch.Tensor if needed
    hidden_state: torch.Tensor


def softmax_sample(distribution, temperature: float):

    if temperature == 0:  
        _, action = max(distribution)
        return 1.0, action  

    visit_counts = np.array([v for v, _ in distribution], dtype=np.float32)
    actions = [a for _, a in distribution]

    # Compute softmax probabilities: N(a)^(1/T) / sum(N(b)^(1/T))
    exponents = np.power(visit_counts, 1 / temperature)
    probabilities = exponents / np.sum(exponents)

    # Sample an action 
    selected_action = np.random.choice(actions, p=probabilities)

    return probabilities[actions.index(selected_action)], selected_action

def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 5:
            return 1.0
        else:
            return 0.0  # Play according to the max.

def make_tictactoe_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float):

    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=9,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=5,  
        batch_size=4,  
        td_steps=max_moves,  #mone carlo
        num_actors=1,
        lr_init=lr_init,
        lr_decay_steps=500,  
        visit_softmax_temperature_fn=visit_softmax_temperature,
        num_episodes=10,  
        known_bounds=KnownBounds(-1, 1))

import logging

# Set up logging (ensures consistent logging across all files)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Change to DEBUG for detailed logs
)

logger = logging.getLogger(__name__)  # Logger object
