import os

import pandas as pd
import plotly.graph_objects as go


def scatter_plot_show(from_filepath):
    df = pd.read_csv(from_filepath)

    fig = go.Figure([go.Scatter(x=df['iteration'], y=df['avg_score'])])
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Average score",
        font=dict(
            family="Courier New, monospace",
            size=18
        )
    )
    fig.show()


def scatter_plot_save(from_filepath, save_to_filepath):
    df = pd.read_csv(from_filepath)

    fig = go.Figure([go.Scatter(x=df['iteration'], y=df['avg_score'])])
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Average score",
        font=dict(
            family="Courier New, monospace",
            size=18
        )
    )

    plots_images_path = save_to_filepath

    if os.path.isdir(plots_images_path):
        print('directory has already existed')
    else:
        os.mkdir(plots_images_path)
        print('new directory has been created')

    fig.write_image(plots_images_path + "scatter_plot.png")
