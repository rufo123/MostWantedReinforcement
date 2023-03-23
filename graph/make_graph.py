"""
This module provides a scatter plot functions to show and save the scatter plot (Graph)
"""
import os

import pandas as pd
import plotly.graph_objects as go


def scatter_plot_show(from_filepath):
    """
    Plots a scatter plot of the data from a CSV file and shows it in a window.

    Args:
        from_filepath (str): Filepath to the CSV file to plot.
    """
    d_f = pd.read_csv(from_filepath)

    fig = go.Figure([go.Scatter(x=d_f['iteration'], y=d_f['avg_score'])])
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Average score",
        font={"family": 'Courier New, monospace', "size": 18}
    )
    fig.show()


def scatter_plot_save(from_filepath, save_to_filepath):
    """
    Plots a scatter plot of the data from a CSV file and saves it as a PNG image.

    Args:
        from_filepath (str): Filepath to the CSV file to plot.
        save_to_filepath (str): Filepath to the directory to save the image to.
    """
    d_f = pd.read_csv(from_filepath)

    fig = go.Figure([go.Scatter(x=d_f['iteration'], y=d_f['avg_score'])])
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Average score",
        font={"family": 'Courier New, monospace', "size": 18}
    )

    plots_images_path = save_to_filepath

    if os.path.isdir(plots_images_path):
        print('directory has already existed')
    else:
        os.mkdir(plots_images_path)
        print('new directory has been created')

    fig.write_image(plots_images_path + "scatter_plot.png")
