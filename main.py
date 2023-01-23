import os
import threading

import torch
import graph.make_graph

from Game import Game
from Utils.stats import write_to_file
from models.short_race import PolicyValueModel

import os
import sys
import torch
import plotly.graph_objects as go
import graph.make_graph
from torch.multiprocessing import set_start_method
from agents.ppo import Agent
from envs.short_race_env import create_env
import time


def game_loop_thread() -> None:
    tmp_game.main_loop()


def agent_loop() -> None:
    settings = {
        'create_scatter_plot': False,
        'load_previous_model': False
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    name = 'test2'
    env_params = tmp_game

    count_of_iterations = 20000
    count_of_processes = 1
    count_of_envs = 1
    count_of_steps = 512
    batch_size = 512

    count_of_epochs = 4
    lr = 2.5e-4

    value_support_size = 4

    path = 'results/short_race/' + name + '/'

    path_logs_score = path + 'logs_score_results.txt'

    path_model = path + 'model4.pt'

    if os.path.isdir(path):
        print('directory has already existed')
    else:
        os.mkdir(path)
        print('new directory has been created')

    dim1 = 3
    count_of_actions = 8
    count_of_features = 8448

    model = PolicyValueModel(count_of_actions, count_of_features)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Drawing Scatter Plot From Existing Data
    if settings.get('create_scatter_plot'):
        graph.make_graph.scatter_plot_show(path_logs_score)

    agent = Agent(model, optimizer, coef_value=0.25,
                  value_support_size=value_support_size,
                  device=device, path=path)

    # Loading Existing Model
    if settings.get('load_previous_model'):
        agent.load_model(path_model)

    iteration_number = 0
    open(path + 'times_rudolf_1.txt', "w").close()
    results_time = ''

    while not tmp_game.is_race_initialised():
        print("Waiting for Race to Initialise")
        time.sleep(1)

    for i in range(count_of_iterations):
        iteration_number = iteration_number + 1
        time_started = time.perf_counter()
        count_of_iterations = 106


        print()

        agent.train(env_params, create_env, count_of_actions,
                    count_of_iterations=count_of_iterations,
                    count_of_processes=count_of_processes,
                    count_of_envs=count_of_envs,
                    count_of_steps=count_of_steps,
                    count_of_epochs=count_of_epochs,
                    batch_size=batch_size, input_dim=(dim1))
        # except Exception as e:
        #     i = i - 1
        #     continue
        time.sleep(3)
        time_elapsed = time.perf_counter() - time_started
        results_time += '\n' + str(iteration_number) + ',' + str(time_elapsed)
        write_to_file(results_time, path + 'times_rudolf_1.txt')
        print("Elapsed: " + str(time_elapsed))
    write_to_file(results_time, path + 'times_rudolf_1.txt')


if __name__ == '__main__':
    tmp_game: Game = Game()

    tmp_game_thread: threading.Thread = threading.Thread(target=game_loop_thread)
    tmp_agent_thread: threading.Thread = threading.Thread(target=agent_loop)

    tmp_game_thread.start()
    tmp_agent_thread.start()

    tmp_game_thread.join()
    tmp_agent_thread.join()
