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
from Arena import Arena
from T3NetWrapper import T3NetWrapper
from Game import TicTacToeGame
from helpers import make_tictactoe_config
from utils import plot_results
from SharedStorage import SharedStorage
import multiprocessing
from multiprocessing import Value, Manager, Lock, Queue
import logging
from logging_config import setup_logging, log_listener  

def actor(global_steps, global_lock, training_losses, self_play_rewards, config, replay_buffer, storage):

    process_name = multiprocessing.current_process().name
    logging.info(f"Process {process_name} started.")

    coach = Coach(config, replay_buffer, global_steps, global_lock)  
    for iteration in range(config.num_iterations):
        logging.info(f'{process_name}: Starting Iteration {iteration + 1}')
        
        logging.info('Starting Self-Play...')
        total_reward = coach.run_selfplay(storage)
        
        # Ensure safe write to shared list
        with global_lock:
            self_play_rewards.append(total_reward)

        logging.info(f'Total Self-Play Reward: {total_reward}')

        logging.info('Starting Training...')
        avg_loss = coach.train_network(storage)

        with global_lock:
            training_losses.append(avg_loss)

        logging.info(f"Average Loss: {avg_loss:.4f}")


def main(config):
    queue = Queue()
    setup_logging(queue)

    listener = multiprocessing.Process(target=log_listener, args=(queue,))
    listener.start()

    nnet = T3NetWrapper()
    replay_buffer = ReplayBuffer(config)
    storage = SharedStorage()
    storage.save_network(0, nnet)

    global_training_steps = Value('i', 0)
    global_training_lock = Lock()

    manager = Manager()
    training_losses = manager.list()  
    self_play_rewards = manager.list()  

    processes = []
    for _ in range(config.num_actors):
        p = multiprocessing.Process(target=actor, args=(
            global_training_steps, global_training_lock, training_losses, self_play_rewards, config, replay_buffer, storage))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # Wait for all processes to finish

    # Stop the logging listener
    queue.put("STOP")
    listener.join()
    
    ##print([(i, reward) for i, reward in enumerate(self_play_rewards)])
    ##print([(i, loss) for i, loss in enumerate(training_losses)])
    return storage.latest_network()

if __name__ == "__main__":
    config = make_tictactoe_config(9, 9, .6, .01)
    best_network = main(config)

    # Play against a random player
    arena = Arena(best_network, T3NetWrapper())
    wins, losses, draws = arena.playRandom(best_network, TicTacToeGame(), num_games=125)
    combined_percentage = (wins + draws) / 125
    log_message = f"""
    📊 {'='*30} 📊
    🎮 Arena Results Summary:
    ------------------------------
    ✅ Wins: {wins} 
    ❌ Losses: {losses} 
    🤝 Draws: {draws}
    ------------------------------
    ✨ Win + Draw Percentage: {combined_percentage:.2f}% ✨
    📊 {'='*30} 📊
    """
    print(log_message)