'''
To Create:
- Main.py (creates a coach)
- Coach.py (runs self-play and trains neural networks)
- MCTS.py (for self-play)
- ReplayBuffer (called by Coach when needed)
        - not running shared storage cuz we don't have multiprocessor
- NNet (for architecture)
- NNet Wrapper (initial inference/recurrent inference)
'''


from ReplayBuffer import ReplayBuffer
from Coach import Coach
from T3NetWrapper import T3NetWrapper
from Game import TicTacToeGame
from helpers import MuZeroConfig, KnownBounds
import traceback  



def main(config):
    nnet = T3NetWrapper()
    replay_buffer = ReplayBuffer(config)
    coach = Coach(config, nnet, replay_buffer)

    # Track metrics
    training_losses = []
    self_play_rewards = []

    try:
        for iteration in range(config.num_iterations):
            print(f'\nStarting Iteration {iteration + 1}')
            
            print('Starting Self-Play...')
            total_reward = coach.run_selfplay()  # Modify run_selfplay to return total reward
            self_play_rewards.append(total_reward)
            print(f"Total Self-Play Reward: {total_reward}")
            
            print('Starting Training...')
            avg_loss = coach.train_network()
            training_losses.append(avg_loss)
            print(f"Average Loss: {avg_loss:.4f}")
            
            if iteration % config.checkpoint_interval == 0:
                nnet.save_checkpoint()
                
    except Exception as e:
        print(f"Error during training: {e}")
        print("Full traceback:")
        traceback.print_exc()  # This will print the full stack trace with line numbers
        nnet.save_checkpoint(filename='error_checkpoint.pth.tar')
            
    return nnet

def make_tictactoe_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float):

  def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 5:
      return 1.0
    else:
      return 0.0  # Play according to the max.

  return MuZeroConfig(
      action_space_size=action_space_size,
      max_moves=9,
      discount=1.0,
      dirichlet_alpha=dirichlet_alpha,
      num_simulations=50,
      batch_size=32,
      td_steps=max_moves,  # Always use Monte Carlo return.
      num_actors=100,
      lr_init=lr_init,
      lr_decay_steps=1000,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      num_episodes=100,
      known_bounds=KnownBounds(-1, 1))

if __name__ == "__main__":
    config = make_tictactoe_config(9, 9, .3, .001)
    best_network = main(config)