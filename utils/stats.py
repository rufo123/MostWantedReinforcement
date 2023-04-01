"""
Module: This module contains a MovingAverageScore class and a write_to_file function for logging.
"""
import math
import os

import numpy as np


def write_to_file(par_log: str, par_filename: str):
    """
     Write log string to file.

     Args:
         par_log (str): Log string to write to file
         par_filename (str): Name of the file to write to
     """
    with open(par_filename, "w", encoding="utf-8") as file:
        file.write(par_log)
        file.close()


def append_to_file(par_filename: str, par_log_without_header: str, par_log_with_header: str = ''):
    """
    Append log string to file.

    Args:
        par_log_without_header (str): Log string to append to file if file already exists and is
         not empty
        par_log_with_header (str): Log string to write to file if file does not exist or is empty
        par_filename (str): Name of the file to append to
    """
    if os.path.isfile(par_filename) and os.stat(par_filename).st_size != 0:
        with open(par_filename, "a", encoding="utf-8") as file:
            file.write(par_log_without_header + '\n')
    else:
        with open(par_filename, "w", encoding="utf-8") as file:
            file.write(par_log_with_header + '\n')


class MovingAverageScore:
    """
    Calculates the moving average score over a given window of episodes.

    Attributes:
        memory: An array that stores the scores.
        index: The current index of the memory array.
        memory_size: The size of the memory array.
        full_memory: A flag indicating whether the memory array is full.
        best_avg_score: The best average score achieved so far.
        best_score: The best score achieved so far.
        count_of_episodes: The total count of episodes seen.
    """

    def __init__(self, count: int = 100):
        """
        Initialize MovingAverageScore class.

        Args:
            count (int): Number of scores to keep in memory
        """
        self.memory = np.zeros(count)

        self.index = 0
        self.memory_size = count
        self.full_memory = False

        self.best_avg_score = - math.inf
        self.best_score = - math.inf
        self.count_of_episodes = 0

    def add(self, scores: np.ndarray):
        """
        Add scores to memory.

        Args:
            scores (numpy.ndarray): Array of scores to add to memory
        """
        length = len(scores)
        if length > 0:
            scores = np.array(scores)
            # noinspection PyArgumentList
            self.best_score = max(self.best_score, scores.max())
            self.count_of_episodes += length

            if length + self.index <= self.memory_size:
                new_index = self.index + length
                self.memory[self.index:new_index] = scores

                if new_index == self.memory_size:
                    self.index = 0
                    self.full_memory = True
                else:
                    self.index = new_index
            else:
                length_to_end = self.memory_size - self.index
                length_from_start = length - length_to_end

                self.memory[self.index:] = scores[:length_to_end]
                self.memory[:length_from_start] = scores[length_to_end:]

                self.index = length_from_start
                self.full_memory = True

    def mean(self) -> tuple[float, bool]:
        """
        Calculate mean score from memory.

        Returns:
            Tuple[float, bool]: Mean score and whether the best average score was updated
        """
        if self.full_memory:
            mean = self.memory.mean()
        else:
            if self.index == 0:
                return -math.inf, False
            mean = self.memory[:self.index].mean()

        if self.best_avg_score < mean:
            self.best_avg_score = mean
            return mean, True
        return mean, False

    def get_best_avg_score(self):
        """
        Get the best average score.

        Returns:
            float: The best average score
        """
        return self.best_avg_score

    def get_count_of_episodes(self):
        """
        Get the total count of episodes.

        Returns:
            int: The count of episodes
        """
        return self.count_of_episodes
