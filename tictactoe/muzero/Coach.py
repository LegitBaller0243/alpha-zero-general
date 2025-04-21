## complete constructor (__init__) method
import torch, copy
import torch.nn.functional as F
import numpy, random
from helpers import *
from utils import Node
import logging
logger = logging.getLogger(__name__)
from Game import TicTacToeGame
from Arena import Arena

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
            ##logging.info(f"Completed Self-Play Game: Total Moves: {len(game.history)}, Total Reward: {game.get_reward()}")
        
        return total_reward


    def play_game(self, nnet):
        game = self.config.new_game()
        while not game.terminal() and len(game.history) < self.config.max_moves:
            root = Node(0)
            current_observation = game.make_image(-1)
            ##print out for debugging
            output = nnet.initial_inference(current_observation)
            self.expand_node(root, game.to_play(), game.legal_actions(),
                    output)
            self.add_exploration_noise(root)

            self.run_mcts(root, game.action_history(), nnet)
            action = self.select_action(len(game.history), root, nnet)
            game.apply(action)
            game.store_search_statistics(root)
        return game


   
    def run_mcts(self, root, action_history, nnet):
        min_max_stats = MinMaxStats(self.config.known_bounds)

        for i in range(self.config.num_simulations):
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
            
            

            legal_actions = list(set(history.action_space()) - set(history.history))
            self.expand_node(node, history.to_play(), legal_actions, network_output)

            self.backpropagate(search_path, network_output.value, history.to_play(),
                            self.config.discount, min_max_stats)


    def select_action(self, num_moves, node, nnet):
        visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
        ]
        ##if num_moves > 2 and random.random() > .80:  # Avoid early noise
            ##logger.info(f"MCTS visit counts at move {num_moves}: {visit_counts}")


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


    def expand_node(self, node, to_play, legal_actions, network_output):
        node.to_play = to_play
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward

        logits = network_output.policy_logits.squeeze(0).detach()
        mask = torch.full_like(logits, float('-inf'))

        for a in legal_actions:
            mask[a] = logits[a]

        policy_probs = F.softmax(mask, dim=0).cpu().numpy()

        ##logger.info(f"For these legal_actions: {legal_actions} use this policy {policy_probs}")

        for a in legal_actions:
            node.children[a] = Node(policy_probs[a])






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
        old_network = storage.latest_network()
        new_network = copy.deepcopy(old_network)
        


        learning_rate = self.config.lr_init * self.config.lr_decay_rate**(
            self.global_training_steps.value / self.config.lr_decay_steps)
        optimizer = torch.optim.Adam(
            list(new_network.representations.parameters()) + 
            list(new_network.dynamics.parameters()) + 
            list(new_network.predictions.parameters()), 
            lr=learning_rate, 
            weight_decay=self.config.weight_decay
        )

        for i in range(self.config.training_steps):
            if i % self.config.checkpoint_interval == 0:
                with self.global_lock:
                    storage.save_network(self.global_training_steps.value, new_network)
            batch = self.replay_buffer.sample_batch(self.config.num_unroll_steps, self.config.td_steps)
            loss = self.update_weights(optimizer, batch, new_network)
            total_loss += loss
            n_steps += 1

        # Evaluate the new network against the old one
        arena = Arena(new_network, old_network, display=None)
        new_wins, old_wins, draws = arena.playGames(20, verbose=False)

        if (new_wins + draws/2) / 20 > self.config.threshold:  
            with self.global_lock:
                storage.save_network(self.global_training_steps.value, new_network)
        else:
            with self.global_lock:
                storage.save_network(self.global_training_steps.value, old_network)  

        return total_loss / n_steps if n_steps > 0 else 0


    def update_weights(self, optimizer, batch, network):
        loss = 0
        training_progress = min(1.0, self.global_training_steps.value / 5000)
        value_loss_coef = 1.0
        policy_coef = 3.0 + 3.0 * training_progress  


        '''if random.random() < 0.01:
            logger.info(f"[Replay Buffer Check] Sampled target policies:")
            for _, _, targets in batch[:3]:
                logger.info([t[2] for t in targets])  # Show first few target policies'''

        for image, actions, targets in batch:
            value, reward, policy_logits, hidden_state = network.initial_inference(image)
            predictions = [(1.0, value, reward, policy_logits)]

            for action in actions:
                value, reward, policy_logits, hidden_state = network.recurrent_inference(
                    hidden_state, action)
                predictions.append((1.0 / len(actions), value, reward, policy_logits))

            for prediction, target in zip(predictions, targets):
                gradient_scale, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                if not target_policy:  
                    continue

                target_value = torch.tensor(target_value, dtype=torch.float32).view(1, 1)
                target_reward = torch.tensor(target_reward, dtype=torch.float32).view(1, 1)
                
                target_policy_tensor = torch.tensor(target_policy, dtype=torch.float32).unsqueeze(0)
                if random.random() >= .9999:
                    predicted_action = int(torch.argmax(policy_logits).item())
                    target_action = int(np.argmax(target_policy))
                    logger.info(f"[Policy Match] Predicted: {predicted_action} | Target: {target_action} | Match: {predicted_action == target_action}")
                    logger.info(
                        f"Predicted Policy (softmax): {F.softmax(policy_logits, dim=-1).detach().cpu().numpy()}, "
                        f"Target Policy: {target_policy}"
                    )

                target_index = torch.tensor([int(np.argmax(target_policy))])


                ce_loss = F.cross_entropy(policy_logits, target_index)

                l = (
                    self.scalar_loss(value, target_value) * value_loss_coef +
                    self.scalar_loss(reward, target_reward) + policy_coef 
                    * ce_loss)




                loss += l * gradient_scale

        optimizer.zero_grad()
        loss.backward()
        weights = network.get_weights()

        '''for i, param in enumerate(network.predictions.parameters()):
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                logger.info(f"Param {i} - Gradient Norm: {grad_norm:.6f}")
            else:
                logger.warning(f"Param {i} - No Gradient!")'''
        


        torch.nn.utils.clip_grad_norm_(network.get_weights(), max_norm=3.0)
        old_params = [p.clone().detach() for p in network.get_weights()]

        optimizer.step()
        
        new_params = [p.clone().detach() for p in network.get_weights()]
        total_change = sum((o - n).abs().sum().item() for o, n in zip(old_params, new_params))

        if random.random() < 0.05:
            logger.info(f"Total parameter change after update: {total_change:.6f}")
            if total_change < 1e-6:
                logger.warning("❗ Model weights not changing! Check gradients and optimizer.")


        with self.global_lock:  
            self.global_training_steps.value += 1  

        return loss.item()


    def scalar_loss(self, prediction, target):
        prediction_tensor = prediction.view(1, 1) 
        return F.mse_loss(prediction_tensor, target)


    ######### End Training ###########

