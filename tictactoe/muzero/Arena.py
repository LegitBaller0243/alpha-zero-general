import numpy as np
from tqdm import tqdm
import logging
from Game import TicTacToeGame

class Arena:
    def __init__(self, player1, player2, display=None):
        """
        Initializes the Arena where two players compete against each other.

        :param player1: Neural network wrapper for player 1.
        :param player2: Neural network wrapper for player 2.
        :param game: The game environment (e.g., Tic-Tac-Toe).
        :param display: Optional function to render the game state (used for verbose mode).
        """
        self.player1 = player1  # First player (e.g., old model)
        self.player2 = player2  # Second player (e.g., new model)
        self.display = display  # Function to visualize game state (optional)

    def playGame(self, starting_player, game, verbose=False):
        """
        Plays one full game between player1 and player2.

        :param starting_player: The player who starts the game (1 or -1).
        :param verbose: If True, prints each move and board state.
        :return: +1 if player1 wins, -1 if player2 wins, 0 for a draw.
        """
        current_player = starting_player  # Player 1 starts
        turn = 0  # Track turns

        if verbose:
            print("\n--- New Game Started ---\n")
            self.display() if self.display else None

        while not game.terminal() and turn < 10:  # While the game is not over
            turn += 1
            curr_observation = game.make_image(-1)
            # Choose action based on the current player
            if current_player == 1:
                _, _, policy_logits, _ = self.player1.initial_inference(curr_observation)
            else: 
                _, _, policy_logits, _ = self.player2.initial_inference(curr_observation)

            policy = policy_logits.detach().numpy()
            policy = policy[0]

            temperature = 0.2
            legal_actions = game.legal_actions()

            if temperature == 0:
                # Mask illegal actions by setting logits to -inf
                masked_logits = np.full_like(policy, -np.inf)
                for a in legal_actions:
                    masked_logits[a] = policy[a]
                action = np.argmax(masked_logits)
            else:
                # Softmax with temperature and masking
                masked_logits = np.full_like(policy, -np.inf)
                for a in legal_actions:
                    masked_logits[a] = policy[a]
                policy_probs = self.softmax_with_temperature(masked_logits, temperature)
                action = np.random.choice(len(policy_probs), p=policy_probs)


            game.apply(action)

            current_player = -current_player  # Switch turns

        result = game.get_reward()
        return result

    def playRandom(self, network, game, coach, num_games=20):
        """
        Plays one full game between player1 and player2.

        :param starting_player: The player who starts the game (1 or -1).
        :param verbose: If True, prints each move and board state.
        :return: +1 if player1 wins, -1 if player2 wins, 0 for a draw.
        """
        current_player = 1  # Network is 1
        wins, losses, draws = 0, 0, 0
        logger = logging.getLogger(__name__)
        for i in range(num_games):
            game = TicTacToeGame()  
            

            while not game.terminal():  # While the game is not over
                if current_player == 1:
                    curr_observation = game.make_image(-1)
                    _, _, policy_logits, _ = network.initial_inference(curr_observation)

                    policy = policy_logits.detach().numpy()[0]
                    policy_probs = np.exp(policy) / np.sum(np.exp(policy))

                    #logger.info(f"\nPolicy Probs:\n{policy_probs.reshape(3, 3)}")

                    legal_actions = game.legal_actions()

                    # Mask illegal actions by setting their probs to -inf
                    masked_policy = np.full_like(policy_probs, -np.inf)
                    for a in legal_actions:
                        masked_policy[a] = policy_probs[a]

                    action = np.argmax(masked_policy)
                    if i % 25 == 0:
                        logger.info(f"\nCurrent Board:\n{game.board}")
                        logger.info(f"Chosen Action (Argmax): {action} | Legal Actions: {legal_actions}")
                    game.apply(action)
                else:
                    action = np.random.choice(game.legal_actions())
                    game.apply(action)


                current_player = -current_player 
            result = game.get_reward()
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

        return wins, losses, draws

    def playGames(self, num_games, verbose=False):
        """
        Plays multiple games between the two players and tracks results.

        :param num_games: Number of games to play.
        :param verbose: If True, prints each game's moves and results.
        :return: (player1 wins, player2 wins, draws)
        """
        p1_wins, p2_wins, draws = 0, 0, 0

        for i in tqdm(range(num_games), desc="Playing Games"):
            print(f"\n--- Game {i+1}/{num_games} ---") if verbose else None
            starting = 1 if (i % 2 == 0) else -1  # Corrected ternary operation
            result = self.playGame(starting_player=starting, game=TicTacToeGame(), verbose=verbose)

            if result == 1:
                p1_wins += 1
            elif result == -1:
                p2_wins += 1
            else:
                draws += 1

        # Final stats
        print("\n--- Match Results ---")
        print(f"Player 1 Wins: {p1_wins}")
        print(f"Player 2 Wins: {p2_wins}")
        print(f"Draws: {draws}")

        return p1_wins, p2_wins, draws

    def softmax_with_temperature(self, logits, temperature):
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

