import plotly.graph_objects as go
import pandas as pd


def scatter_plot(filepath):
    df = pd.read_csv(filepath)

    fig = go.Figure([go.Scatter(x=df['iteration'], y=df['avg_score'])])
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Average score",
        font=dict(
            family="Courier New, monospace",
            size=18
        )
    )
    fig.show();
