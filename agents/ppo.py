"""
This module implements utilities for working with the distributional RL algorithm,
including functions to convert between scalar and categorical representations of
value distributions and an implementation of the Proximal Policy Optimization (PPO)
algorithm.
"""

import pickle
import warnings
from datetime import datetime
from typing import Union

import numpy as np
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.multiprocessing import Process, Pipe

import graph.make_graph
from configuration.i_configuration import IConfiguration
from game_inputs import GameInputs
from utils.print_utils.printer import Printer
from utils.stats import MovingAverageScore, write_to_file, append_to_file


# methods support_to_scalar and scalar_to_support are implemented
# by Davaud Werner in https://github.com/werner-duvaud/muzero-general

# noinspection DuplicatedCode
def support_to_scalar(par_logits, par_support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(par_logits, dim=1)

    support = (torch.tensor(list(range(-par_support_size, par_support_size + 1)))
               .expand(probabilities.shape)
               .float()
               .to(device=probabilities.device))

    x_return = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x_return = torch.sign(x_return) * (
            ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x_return) + 1 + 0.001)) - 1) / (2 * 0.001))
            ** 2
            - 1
    )
    return x_return


# noinspection DuplicatedCode
def scalar_to_support(par_x, par_support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the par_scale (defined in https://arxiv.org/abs/1805.11593)
    par_x = torch.sign(par_x) * (torch.sqrt(torch.abs(par_x) + 1) - 1) + 0.001 * par_x
    # Encode on a vector
    par_x = torch.clamp(par_x, -par_support_size, par_support_size)
    floor = par_x.floor()
    prob = par_x - floor
    logits = torch.zeros(par_x.shape[0], par_x.shape[1], 2 * par_support_size + 1).to(par_x.device)
    logits.scatter_(
        2, (floor + par_support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + par_support_size + 1
    prob = prob.masked_fill_(2 * par_support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * par_support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=unused-variable
# pylint: disable=too-many-statements
# noinspection DuplicatedCode
def worker(connection, env_param, env_func, count_of_iterations, count_of_envs,
           count_of_steps, gamma, gae_lambda, start_iteration_number) -> None:
    """
    worker function for Proximal Policy Optimization (PPO) agent training.

    Args:
    connection (Connection): Connection object for communication with the main process.
    env_param (dict): A dictionary containing environment parameters.
    env_func (callable): A callable function that returns an environment.
    count_of_iterations (int): Number of iterations for training.
    count_of_envs (int): Number of parallel environments to use.
    count_of_steps (int): Number of steps for each environment per iteration.
    gamma (float): Discount factor for rewards.
    gae_lambda (float): Lambda parameter for generalized advantage estimation.

    Returns:
    None.
    """
    envs = [env_func(*env_param) for _ in range(count_of_envs)]
    observations = torch.stack([torch.from_numpy(
        env.reset(start_iteration_number)
    ) for env in envs])
    game_score = np.zeros(count_of_envs)
    steps_taken_storage = np.zeros(count_of_steps)

    mem_log_probs = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_actions = torch.zeros((count_of_steps, count_of_envs, 1), dtype=torch.long)
    mem_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
    mem_rewards = torch.zeros((count_of_steps, count_of_envs, 1))

    for iteration in range(count_of_iterations):
        mem_non_terminals = torch.ones((count_of_steps, count_of_envs, 1))
        scores = []
        steps_taken_list = []

        for step in range(count_of_steps):
            Printer.print_basic("STEP: " + str(step), "AGENT")
            connection.send(observations.float())
            logits, values = connection.recv()
            probs = F.softmax(logits, dim=-1)
            actions = probs.multinomial(num_samples=1)
            log_probs = F.log_softmax(logits, dim=-1).gather(1, actions)

            mem_log_probs[step] = log_probs
            mem_actions[step] = actions
            mem_values[step] = values

            for idx in range(count_of_envs):
                observation, reward, terminal, steps_took_to_complete = envs[idx].step(
                    actions[idx, 0].item())
                mem_rewards[step, idx, 0] = reward
                game_score[idx] += reward
                Printer.print_info("Single Reward: " + str(reward), "AGENT")
                Printer.print_info("Cumulative Reward: " + str(game_score[idx]), "AGENT")
                if reward < 0:
                    mem_non_terminals[step, idx, 0] = 0
                if terminal:
                    mem_non_terminals[step, idx, 0] = 0
                    scores.append(game_score[idx])
                    game_score[idx] = 0
                    steps_taken_storage[idx] = steps_took_to_complete
                    steps_taken_list.append(steps_taken_storage[idx])
                    observation = envs[idx].reset(start_iteration_number + iteration)
                # observations[idx] = observation.clone().detach()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    observations[idx] = torch.tensor(observation)

        connection.send(observations.float())
        # noinspection PyUnboundLocalVariable
        mem_values[step + 1] = connection.recv()
        mem_rewards = torch.clamp(mem_rewards, -1.0, 1.0)
        advantages = torch.zeros((count_of_steps, count_of_envs, 1))
        values = torch.zeros((count_of_steps, count_of_envs, 1))
        t_gae = torch.zeros((count_of_envs, 1))

        for step in reversed(range(count_of_steps)):
            delta = mem_rewards[step] + gamma * mem_values[step + 1] * mem_non_terminals[step] \
                    - mem_values[step]
            t_gae = delta + gamma * gae_lambda * t_gae * mem_non_terminals[step]
            values[step] = t_gae + mem_values[step]
            advantages[step] = t_gae.clone()

        connection.send([mem_log_probs, mem_actions, values, advantages, scores, steps_taken_list])
    connection.recv()
    connection.close()


# pylint: disable=too-many-instance-attributes
# noinspection DuplicatedCode
class Agent:
    """
    A reinforcement learning agent using Proximal Policy Optimization with Generalized Advantage
        Estimation.

    :param model: The neural network model to use for the agent.
    :param optimizer: The optimizer to use for training the neural network model.
    :param gamma: The discount factor for future rewards.
    :param epsilon: The clip parameter for the PPO loss function.
    :param coef_value: The coefficient for the value loss in the PPO loss function.
    :param coef_entropy: The coefficient for the entropy bonus in the PPO loss function.
    :param gae_lambda: The lambda parameter for Generalized Advantage Estimation.
    :param path: The path to the directory where training results will be saved.
    :param device: The device to use for running the neural network model.
    :param value_support_size: The number of values in the support used for value estimation.
    """

    def __init__(self, model, optimizer, gamma=0.997, epsilon=0.1,
                 coef_value=0.5, coef_entropy=0.001, gae_lambda=0.95,
                 path='results/', device='cpu',
                 value_support_size=1):

        self.model = model
        self.optimizer = optimizer

        self.gamma = gamma
        self.coef_value = coef_value
        self.coef_entropy = coef_entropy
        self.gae_lambda = gae_lambda

        self.lower_bound = 1 - epsilon
        self.upper_bound = 1 + epsilon

        self.path = path
        self.device = device

        self.value_support_size = value_support_size
        self.support_to_value = value_support_size > 1
        self.value_support_interval = value_support_size * 2 + 1

        self.start_iteration_value: int = 0
        self.loaded_score_path: str = ""

        self.start_time = datetime.now()

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    def train(self,
              env_param: tuple[GameInputs, IConfiguration],
              env_func, count_of_actions,
              count_of_iterations=10000, count_of_processes=2,
              count_of_envs=16, count_of_steps=128, count_of_epochs=4,
              batch_size=512, input_dim: Union[int, tuple] = 4):
        """
        Trains the agent using Proximal Policy Optimization with Generalized Advantage Estimation.

        :param env_param: An instance of the GameInputs class containing the inputs for the game.
        :type env_param: GameInputs
        :param env_func: A function that returns the environment to be used for training.
        :param count_of_actions: The number of possible actions in the environment.
        :param count_of_iterations: the number of training iterations to run
        :param count_of_processes: the number of parallel processes to use for training
        :param count_of_envs: the number of environments to run in each process
        :param count_of_steps: the number of steps to run in each environment per iteration
        :param count_of_epochs: the number of times to update the network using the collected data
        :param batch_size: the size of the batches used to update the network
        :param input_dim: the dimensionality of the observation space
        """

        input_dim_readjusted: tuple
        if isinstance(input_dim, int):
            input_dim_readjusted = (input_dim,)
        else:
            input_dim_readjusted = input_dim

        Printer.print_info("Training is starting", "AGENT")

        logs_score = 'iteration,episode,avg_score,best_avg_score,hours_took,steps_took'
        logs_loss = 'iteration,episode,policy,value,entropy'

        score = MovingAverageScore()

        if self.loaded_score_path != "":
            with open(self.loaded_score_path, "rb") as loaded_score:
                score = pickle.load(loaded_score)

        buffer_size = count_of_processes * count_of_envs * count_of_steps
        batches_per_iteration = count_of_epochs * buffer_size / batch_size

        processes, connections = [], []
        for _ in range(count_of_processes):
            parr_connection, child_connection = Pipe()
            process = Process(target=worker, args=(
                child_connection, env_param, env_func, count_of_iterations,
                count_of_envs, count_of_steps, self.gamma, self.gae_lambda,
                self.start_iteration_value))
            connections.append(parr_connection)
            processes.append(process)
            process.start()

        mem_dim = (count_of_processes, count_of_steps, count_of_envs)
        mem_observations = torch.zeros((mem_dim + input_dim_readjusted), device=self.device)
        mem_actions = torch.zeros((*mem_dim, 1), device=self.device, dtype=torch.long)
        mem_log_probs = torch.zeros((*mem_dim, 1), device=self.device)
        if self.support_to_value:
            mem_values = torch.zeros((*mem_dim, self.value_support_interval), device=self.device)
        else:
            mem_values = torch.zeros((*mem_dim, 1), device=self.device)
        mem_advantages = torch.zeros((*mem_dim, 1), device=self.device)

        for iteration in range(self.start_iteration_value,
                               self.start_iteration_value + count_of_iterations):
            for step in range(count_of_steps):
                observations = [conn.recv() for conn in connections]
                observations = torch.stack(observations).to(self.device)
                mem_observations[:, step] = observations

                with torch.no_grad():
                    logits, values = self.model(observations.view(-1, *input_dim_readjusted))

                # If you selected actions in the main process, your iteration
                # would last about 0.5 seconds longer (measured on 2 processes)
                logits = logits.view(-1, count_of_envs, count_of_actions).cpu()

                if self.support_to_value:
                    values = support_to_scalar(values, self.value_support_size).view(
                        -1, count_of_envs, 1).cpu()
                else:
                    values = values.view(-1, count_of_envs, 1).cpu()

                for idx in range(count_of_processes):
                    connections[idx].send([logits[idx], values[idx]])

            observations = [conn.recv() for conn in connections]
            observations = torch.stack(observations).to(self.device)

            with torch.no_grad():
                _, values = self.model(observations.view(-1, *input_dim_readjusted))

            if self.support_to_value:
                values = support_to_scalar(values, self.value_support_size).view(
                    -1, count_of_envs, 1).cpu()
            else:
                values = values.view(-1, count_of_envs, 1).cpu()

            for idx in range(count_of_processes):
                connections[idx].send(values[idx])

            iteration_steps_took = 0

            for idx in range(count_of_processes):
                log_probs, actions, values, advantages, scores, steps_taken_list = connections[
                    idx].recv()
                mem_actions[idx] = actions.to(self.device)
                mem_log_probs[idx] = log_probs.to(self.device)
                if self.support_to_value:
                    mem_values[idx] = scalar_to_support(
                        values.to(self.device).view(-1, 1),
                        self.value_support_size
                    ).view(-1, count_of_envs, self.value_support_interval)
                else:
                    mem_values[idx] = values.to(self.device)
                mem_advantages[idx] = advantages.to(self.device)
                score.add(scores)
                iteration_steps_took = np.mean(steps_taken_list)

            avg_score, best_score = score.mean()
            print('iteration: ', iteration, '\taverage score: ', avg_score)
            if best_score:
                print('New best avg score has been achieved', avg_score)
                torch.save(self.model.state_dict(), self.path + 'model' + str(iteration) + '.pt')
                with open(self.path + 'score' + str(iteration), "wb") as loaded_score:
                    pickle.dump(score, loaded_score)
            elif iteration % 100 == 0:
                torch.save(self.model.state_dict(), self.path + 'model' + str(iteration) + '.pt')
                with open(self.path + 'score' + str(iteration), "wb") as loaded_score:
                    pickle.dump(score, loaded_score)

            mem_observations = mem_observations.view(-1, *input_dim_readjusted)
            mem_actions = mem_actions.view(-1, 1)
            mem_log_probs = mem_log_probs.view(-1, 1)

            if self.support_to_value:
                mem_values = mem_values.view(-1, self.value_support_interval)
            else:
                mem_values = mem_values.view(-1, 1)
            mem_advantages = mem_advantages.view(-1, 1)
            mem_advantages = (mem_advantages - mem_advantages.mean()) / (
                    mem_advantages.std() + 1e-5)

            s_policy, s_value, s_entropy = 0.0, 0.0, 0.0

            for epoch in range(count_of_epochs):
                perm = torch.randperm(buffer_size, device=self.device).view(-1, batch_size)
                for idx in perm:
                    logits, values = self.model(mem_observations[idx])
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs.gather(1, mem_actions[idx])

                    entropy_loss = (log_probs * probs).sum(1, keepdim=True).mean()

                    if self.support_to_value:
                        values_log_probs = F.log_softmax(values, dim=-1)
                        value_loss = torch.sum(- mem_values[idx] * values_log_probs, dim=1).mean()
                    else:
                        value_loss = F.mse_loss(values, mem_values[idx])

                    ratio = torch.exp(new_log_probs - mem_log_probs[idx])
                    surr_policy = ratio * mem_advantages[idx]
                    surr_clip = torch.clamp(ratio, self.lower_bound, self.upper_bound) \
                                * mem_advantages[idx]
                    policy_loss = - torch.min(surr_policy, surr_clip).mean()

                    s_policy += policy_loss.item()
                    s_value += value_loss.item()
                    s_entropy += entropy_loss.item()

                    self.optimizer.zero_grad()
                    loss = policy_loss + self.coef_value * value_loss \
                           + self.coef_entropy * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

            mem_observations = mem_observations.view((mem_dim + input_dim_readjusted))
            mem_actions = mem_actions.view((*mem_dim, 1))
            mem_log_probs = mem_log_probs.view((*mem_dim, 1))
            if self.support_to_value:
                # noinspection PyUnusedLocal
                mem_values = mem_values = mem_values.view((*mem_dim, self.value_support_interval))
            else:
                mem_values = mem_values.view((*mem_dim, 1))
            mem_advantages = mem_advantages.view((*mem_dim, 1))

            elapsed_time = datetime.now() - self.start_time
            hours_taken = elapsed_time.total_seconds() / 3600

            actual_score: str = '\n' + str(iteration) + ',' \
                                + str(score.get_count_of_episodes()) + ',' \
                                + str(avg_score) + ',' \
                                + str(score.get_best_avg_score()) + ',' \
                                + str(round(hours_taken, 2)) + ',' \
                                + str(round(iteration_steps_took))

            logs_score += actual_score

            logs_loss += '\n' + str(iteration) + ',' \
                         + str(avg_score) + ',' \
                         + str(s_policy / batches_per_iteration) + ',' \
                         + str(s_value / batches_per_iteration) + ',' \
                         + str(s_entropy / batches_per_iteration)

            if iteration % 10 == 0:
                write_to_file(logs_score, self.path + 'logs_score.txt')
                write_to_file(logs_loss, self.path + 'logs_loss.txt')
                append_to_file(
                    par_filename=self.path + 'logs_final.txt',
                    par_log_without_header=actual_score,
                    par_log_with_header=logs_score
                )
                graph.make_graph.scatter_plot_save(self.path + 'logs_final.txt', self.path)
                torch.save(self.model.state_dict(), self.path + 'latest_model' + '.pt')
                with open(self.path + 'latest_score', "wb") as loaded_score:
                    pickle.dump(score, loaded_score)

        print('Training has ended, best avg score is ', score.get_best_avg_score())

        for connection in connections:
            connection.send(1)
        for process in processes:
            process.join()

    def load_model(self, path, par_start_iter_number: int):
        """
        Load a PyTorch model from the specified file path and update instance variables.

        Args:
            path (str): The file path to the saved model.
            par_start_iter_number (int): The starting iteration number of the model.

        Returns:
            None
        """
        self.model.load_state_dict(torch.load(path + 'model' + str(par_start_iter_number) + '.pt'))
        self.loaded_score_path = path + 'score' + str(par_start_iter_number)
        self.start_iteration_value = par_start_iter_number + 1
