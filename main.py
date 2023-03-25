"""
Module providing a main.py
If you are importing this module you are doing something wrong!
"""
import multiprocessing
import os
import time

import torch

import graph.make_graph
from agents.ppo import Agent
from envs.short_race_env import create_env
from game_api.game import Game
from game_inputs import GameInputs
from models.short_race import PolicyValueModel
from utils.stats import write_to_file


def game_loop_thread(par_game_inputs: GameInputs) -> None:
    """
    A function representing a thread that runs the game loop.

    Args:
        par_game_inputs (GameInputs): An instance of the GameInputs class containing the 
            inputs for the game.

    Returns:
        None: This function doesn't return anything.
    """
    tmp_game: Game = Game()
    try:
        tmp_game.main_loop(par_game_inputs)
    except Exception as exception:
        print(f"An error occurred in Game Api: {exception}")
        raise


# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def agent_loop(par_game_inputs: GameInputs) -> None:
    """
    A function representing the agent loop.

    Args:
        par_game_inputs (GameInputs): An instance of the GameInputs class containing the inputs for
            the game.
a
    Returns:a
        None: This function doesn't return anything.
    """
    settings = {
        'create_scatter_plot': False,
        'load_previous_model': False,
        'previous_model_iter_number': 0
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.version.cuda)
    print('device: ', device)
    # set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    name = 'fifth_iteration_training_convo'
    env_param = par_game_inputs
    count_of_iterations = 20000
    count_of_processes = 1
    count_of_envs = 1
    count_of_steps = 150
    # the batch size needs to be a factor or divisor of 'count_of_steps' for it to work properly.
    # Choosing batch sizes that are multiples or factors of the step count can help ensure that the
    # batches align properly with the steps, preventing issues with the input size.
    # Factors and Divisors of 150:
    # [1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150]
    batch_size = 150

    count_of_epochs = 4
    tmp_learning_rate = 2.5e-4

    value_support_size = 1

    path = 'h:/diplomka_vysledky/results/short_race/' + name + '/'

    path_logs_score = path + 'logs_score_results.txt'

    if os.path.isdir(os.path.abspath(path)):
        print('directory has already existed')
    else:
        os.mkdir(os.path.abspath(path))
        print('new directory has been created')

    dim1 = 35
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
        agent.load_model(path, settings.get('previous_model_iter_number'))

    iteration_number = 0
    with open(path + 'times_rudolf_1.txt', "w", encoding="utf-8"):
        pass
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

    iteration_number = iteration_number + 1
    time_started = time.perf_counter()

    print()

    try:
        agent.train(env_param, create_env, count_of_actions,
                    count_of_iterations=count_of_iterations,
                    count_of_processes=count_of_processes,
                    count_of_envs=count_of_envs,
                    count_of_steps=count_of_steps,
                    count_of_epochs=count_of_epochs,
                    batch_size=batch_size, input_dim=dim1)
    except Exception as exception:
        print(f"An exception occurred during training:: {exception}")
        raise
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
    #graph.make_graph.scatter_plot_show(os.path.abspath('H:\\diplomka_vysledky\\results\\'
    #                                                  'short_race\\fourth_iteration_training'
    #                                                  '\\vysledne_skore.txt'))
    #exit()

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
