from typing import List
import matplotlib.pyplot as plt

class Action:
    def __init__(self, index: int):
        self.index = index

    def __eq__(self, other):
        # Handle comparison with both Action objects and integers
        if isinstance(other, Action):
            return self.index == other.index
        elif isinstance(other, int):
            return self.index == other
        return False

    def __hash__(self):
        return hash(self.index)

    def __gt__(self, other):
      return self.index > other.index


class Player(object):
  pass


class Node(object):
  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history, action_space_size):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action):
    self.history.append(action)

  def last_action(self):
    return self.history[-1]

  def action_space(self):
    return range(self.action_space_size)  # Just return range of ints

  def to_play(self) -> Player:
    return Player()

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def plot_results(wins, losses, draws):
    """
    Plots the results of the games played.

    :param wins: Number of wins.
    :param losses: Number of losses.
    :param draws: Number of draws.
    """
    labels = ['Wins', 'Losses', 'Draws']
    sizes = [wins, losses, draws]
    colors = ['#4CAF50', '#F44336', '#FFC107']
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, sizes, color=colors)
    plt.title('Results of Best Network vs Random Player')
    plt.ylabel('Number of Games')
    plt.xlabel('Outcome')
    plt.ylim(0, max(sizes) + 10)  # Add some space above the highest bar
    
    total_games = wins + losses + draws
    if total_games > 0:
        winning_percentage = ((wins + draws) / total_games) * 100
        plt.text(0, wins + 1, f'Winning Percentage: {winning_percentage:.2f}%', ha='center')

    plt.show()
