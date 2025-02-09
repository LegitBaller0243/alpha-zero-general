import numpy as np
class ReplayBuffer(object):

  def __init__(self, config):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.history[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps))
            for (g, i) in game_pos]

  def sample_game(self):
    if len(self.buffer) == 0:
      raise ValueError("Replay buffer is empty, cannot sample game.")
    return self.buffer[np.random.randint(len(self.buffer))]  # Select a game randomly

  def sample_position(self, game) -> int:
    if len(game.history) == 0:
        return 0 
    return np.random.randint(0, len(game.history))
