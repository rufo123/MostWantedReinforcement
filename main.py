import multiprocessing
import os
import time

import torch

import graph.make_graph
from agents.ppo import Agent
from envs.short_race_env import create_env
from game import Game
from game_inputs import GameInputs
from models.short_race import PolicyValueModel
from utils.stats import write_to_file


def game_loop_thread(par_game_inputs: GameInputs) -> None:
    """
    A function representing a thread that runs the game loop.

    Args:
        par_game_inputs (GameInputs): An instance of the GameInputs class containing the inputs for the game.

    Returns:
        None: This function doesn't return anything.
    """
    tmp_game: Game = Game()
    tmp_game.main_loop(par_game_inputs)


def agent_loop(par_game_inputs: GameInputs) -> None:
    """
    A function representing the agent loop.

    Args:
        par_game_inputs (GameInputs): An instance of the GameInputs class containing the inputs for
            the game.

    Returns:
        None: This function doesn't return anything.
    """
    settings = {
        'create_scatter_plot': False,
        'load_previous_model': True
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    # set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    name = 'test2'
    env_param = par_game_inputs
    count_of_iterations = 20000
    count_of_processes = 1
    count_of_envs = 1
    count_of_steps = 100
    batch_size = 100

    count_of_epochs = 4
    tmp_learning_rate = 2.5e-4

    value_support_size = 1

    path = 'results/short_race/' + name + '/'

    path_logs_score = path + 'logs_score_results.txt'

    tmp_model_start_iter_number: int = 47
    path_model = path + 'model' + str(tmp_model_start_iter_number) + '.pt'

    if os.path.isdir(path):
        print('directory has already existed')
    else:
        os.mkdir(path)
        print('new directory has been created')

    dim1 = 4
    count_of_actions = 8
    count_of_features = 8448

    model = PolicyValueModel(count_of_actions, count_of_features)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tmp_learning_rate)

    # Drawing Scatter Plot From Existing Data
    if settings.get('create_scatter_plot'):
        graph.make_graph.scatter_plot_show(path_logs_score)

    agent = Agent(model, optimizer, coef_value=0.25,
                  value_support_size=value_support_size,
                  device=device, path=path)

    # Loading Existing Model

    if settings.get('load_previous_model'):
        agent.load_model(path_model, tmp_model_start_iter_number + 1)

    iteration_number = 0
    open(path + 'times_rudolf_1.txt', "w").close()
    results_time = ''

    tmp_game_variables: tuple = par_game_inputs.game_initialization_inputs.get()

    tmp_is_game_started: bool = tmp_game_variables[0]

    par_game_inputs.game_initialization_inputs.put(tmp_game_variables)

    while not tmp_is_game_started:
        print("Waiting for Race to Initialise")

        tmp_game_variables: tuple = par_game_inputs.game_initialization_inputs.get()

        tmp_is_game_started: bool = tmp_game_variables[0]

        par_game_inputs.game_initialization_inputs.put(tmp_game_variables)

        time.sleep(1)

    for i in range(count_of_iterations):
        iteration_number = iteration_number + 1
        time_started = time.perf_counter()
        count_of_iterations = 106

        print()

        agent.train(env_param, create_env, count_of_actions,
                    count_of_iterations=count_of_iterations,
                    count_of_processes=count_of_processes,
                    count_of_envs=count_of_envs,
                    count_of_steps=count_of_steps,
                    count_of_epochs=count_of_epochs,
                    batch_size=batch_size, input_dim=dim1)
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
    # graph.make_graph.scatter_plot_show('vysledky.txt')

    tmp_queue_env_inputs: multiprocessing.Queue = multiprocessing.Queue()
    tmp_queue_game_started_inputs: multiprocessing.Queue = multiprocessing.Queue()
    tmp_queue_restart_game_input: multiprocessing.Queue = multiprocessing.Queue()

    game_inputs: GameInputs = GameInputs(
        tmp_queue_env_inputs,
        tmp_queue_game_started_inputs,
        tmp_queue_restart_game_input
    )

    tmp_game_thread = multiprocessing.Process(target=game_loop_thread, args=(game_inputs,))
    tmp_agent_thread = multiprocessing.Process(target=agent_loop, args=(game_inputs,))

    tmp_game_thread.start()
    tmp_agent_thread.start()

    tmp_game_thread.join()
    tmp_agent_thread.join()
