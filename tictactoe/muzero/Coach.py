## complete constructor (__init__) method
import torch
import torch.nn.functional as F
import numpy
from helpers import *
from utils import Node
import logging

from T3NetWrapper import T3NetWrapper


class Coach():
    def __init__(self, config, replay_buffer, global_training_steps, global_lock):
        self.config = config
        self.replay_buffer = replay_buffer
        self.global_training_steps = global_training_steps
        self.global_lock = global_lock  # Store the lock

    ####### Self-Play ########
    def run_selfplay(self, storage):
        total_reward = 0
        for _ in range(self.config.num_episodes):
            network = storage.latest_network()
            game = self.play_game(network)
            self.replay_buffer.save_game(game)
            total_reward += game.get_reward()
            logging.info(f"Completed Self-Play Game: Total Moves: {len(game.history)}, Total Reward: {game.get_reward()}")
        
        return total_reward


    def play_game(self, nnet):
        game = self.config.new_game()

        while not game.terminal() and len(game.history) < self.config.max_moves:
            root = Node(0)
            current_observation = game.make_image(-1)
            self.expand_node(root, game.to_play(), game.legal_actions(),
                    nnet.initial_inference(current_observation))
            self.add_exploration_noise(root)

            self.run_mcts(root, game.action_history(), nnet)
            action = self.select_action(len(game.history), root, nnet)
            game.apply(action)
            game.store_search_statistics(root)
        return game


   
    def run_mcts(self, root, action_history, nnet):
        min_max_stats = MinMaxStats(self.config.known_bounds)

        for _ in range(self.config.num_simulations):
            history = action_history.clone()
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                history.add_action(action)
                search_path.append(node)


            parent = search_path[-2]
            network_output = nnet.recurrent_inference(parent.hidden_state,
                                                            history.last_action())
            self.expand_node(node, history.to_play(), history.action_space(), network_output)

            self.backpropagate(search_path, network_output.value, history.to_play(),
                            self.config.discount, min_max_stats)


    def select_action(self, num_moves, node, nnet):
        visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
        ]
        t = self.config.visit_softmax_temperature_fn(
            num_moves=num_moves, training_steps=nnet.training_steps())
        _, action = softmax_sample(visit_counts, t)
        return action


    def select_child(self, node, min_max_stats):
        _, action, child = max(
            (self.ucb_score(node, child, min_max_stats), action,
            child) for action, child in node.children.items())
        return action, child



    def ucb_score(self, parent, child, min_max_stats):
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) /
                        self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())
        return prior_score + value_score


    def expand_node(self, node, to_play, actions, network_output):
        node.to_play = to_play
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward

        policy_logits = torch.softmax(network_output.policy_logits, dim=-1)
        policy_logits_list = policy_logits.squeeze(0).tolist()

        policy = {a: policy_logits_list[i] for i, a in enumerate(actions)}
        
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)



    def backpropagate(self, search_path, value, to_play, discount, min_max_stats):
        for node in search_path:
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + discount * value

    def add_exploration_noise(self, node):
        actions = list(node.children.keys())
        noise = numpy.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


    ####### Part 2: Training #########

    def train_network(self, storage):
        total_loss = 0
        n_steps = 0
        network = T3NetWrapper()
        
        learning_rate = self.config.lr_init * self.config.lr_decay_rate**(
            self.global_training_steps.value / self.config.lr_decay_steps)
        optimizer = torch.optim.SGD(list(network.representations.parameters()) + 
                                        list(network.dynamics.parameters()) + 
                                        list(network.predictions.parameters()), 
                                        lr=learning_rate, momentum=self.config.momentum, 
                                        weight_decay=0.0001
                                    )

        for i in range(self.config.training_steps):
            if i % self.config.checkpoint_interval == 0:
                with self.global_lock:
                    storage.save_network(self.global_training_steps.value, network)
            batch = self.replay_buffer.sample_batch(self.config.num_unroll_steps, self.config.td_steps)
            loss = self.update_weights(optimizer, batch, self.config.weight_decay, network)
            total_loss += loss
            n_steps += 1
            logging.info(f"Training Step: {self.global_training_steps.value}, Learning Rate: {learning_rate}")
            with self.global_lock:
                storage.save_network(self.global_training_steps.value, network)

        return total_loss / n_steps if n_steps > 0 else 0


    def update_weights(self, optimizer, batch, weight_decay, network):
        loss = 0
        for image, actions, targets in batch:
            value, reward, policy_logits, hidden_state = network.initial_inference(image)
            predictions = [(1.0, value, reward, policy_logits)]

            for action in actions:
                value, reward, policy_logits, hidden_state = network.recurrent_inference(
                    hidden_state, action)
                predictions.append((1.0 / len(actions), value, reward, policy_logits))

            hidden_state = hidden_state * 0.5

            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                # Skip empty targets (end of game)
                if not target_policy:  
                    continue

                target_value = torch.tensor(target_value, dtype=torch.float32).view(1, 1)
                target_reward = torch.tensor(target_reward, dtype=torch.float32).view(1, 1)
                
                target_policy_tensor = torch.argmax(torch.tensor(target_policy, dtype=torch.float32), dim=0).view(1)

                l = (
                    self.scalar_loss(value, target_value) +
                    self.scalar_loss(reward, target_reward) +
                    F.cross_entropy(policy_logits, target_policy_tensor))

                loss += l * gradient_scale

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.get_weights(), max_norm=1.0)
        optimizer.step()

        with self.global_lock:  
            self.global_training_steps.value += 1  
            logging.info(f"Training Step: {self.global_training_steps.value}, Loss: {loss.item()}")

        return loss.item()


    def scalar_loss(self, prediction, target):
        return F.mse_loss(prediction, target)

    ######### End Training ###########

