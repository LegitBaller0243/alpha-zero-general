import numpy as np
from typing import List
from utils import ActionHistory, Action
import logging
logger = logging.getLogger(__name__)



class Game:
    """A single episode of interaction with the environment (MuZero-compatible)."""

    def __init__(self, action_space_size: int, discount: float):
        self.history = [] 
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

        self.states = [np.zeros((3, 3), dtype=np.float32)]     # Stores board states for make_image


    def terminal(self):
        """(override in subclass)"""
        pass

    def legal_actions(self):
        """(override in subclass)."""
        return []

    def apply(self, action, new_state, reward):
        #DeepMind - found reward by stepping through environment
        self.history.append(action)
        self.states.append(new_state)  # Store board state
        self.rewards.append(reward)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        ## this is different - deepMind did (Action(index))
        action_space = [Action(i) for i in range(self.action_space_size)]
        
        
        temperature = 0.25  # Lower values make distributions sharper

        visits = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in action_space
        ], dtype=np.float32)

        visits = visits ** (1 / temperature)
        normalized_visits = (visits / np.sum(visits)).tolist()

        ##logger.info(f"[Search Statistics] Child visits: {normalized_visits}")
    # Store visits
        self.child_visits.append(normalized_visits)
        self.root_values.append(root.value())


    def make_image(self, state_index):
        if not self.states:
            return np.zeros((1, 3, 3), dtype=np.float32)  # Empty board as fallback
        if state_index >= len(self.states):
            return self.states[-1].reshape(1, 3, 3)
        return self.states[state_index].reshape(1, 3, 3)  # Shape: (1, height, width)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index], self.child_visits[current_index]))
            else:
                targets.append((0, 0, []))  # Absorbing state
        return targets

    def to_play(self) -> int:
        return 1 if len(self.history) % 2 == 0 else -1  # Alternating turns

    def action_history(self):
        return ActionHistory(self.history, self.action_space_size)


class TicTacToeGame(Game):
    """A Tic-Tac-Toe environment inheriting from MuZero's Game class."""

    def __init__(self, action_space_size=9, discount=1.0, n=3):
        super().__init__(action_space_size, discount)
        self.n = n
        self.board = np.zeros((n, n), dtype=np.float32)  # Start with an empty board

    def get_init_board(self):
        """Returns the initial empty board state."""
        return np.zeros((self.n, self.n), dtype=np.float32)

    def legal_actions(self) -> List[int]:
        """Returns a list of valid move indices."""
        return [i for i in range(self.n * self.n) if self.board[i // self.n, i % self.n] == 0]

    def apply(self, action: int):
        """Applies an action, updates board state, and records history."""
        row, col = divmod(action, self.n)
        self.board[row, col] = self.to_play()
        reward = self.get_reward()
        super().apply(action, np.copy(self.board), reward)

    def get_reward(self):
        """Returns the reward for the current board state."""
        if self.is_win(1):
            return 1
        if self.is_win(-1):
            return -1
        return 0  # No winner yet

    def is_win(self, player: int) -> bool:
        """Checks if a player has won."""
        for row in self.board:
            if np.all(row == player):
                return True
        for col in range(self.n):
            if np.all(self.board[:, col] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def terminal(self) -> bool:
        """Returns True if the game is over (win or draw)."""
        return self.is_win(1) or self.is_win(-1) or not any(0 in row for row in self.board)

    def string_representation(self):
        """Returns a string representation of the board for debugging."""
        return str(self.board)

    def display(self):
        """Prints the board to the console."""
        for row in self.board:
            print(" ".join(["X" if x == 1 else "O" if x == -1 else "-" for x in row]))
        print()
