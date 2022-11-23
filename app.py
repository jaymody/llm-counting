from dataclasses import dataclass

import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from model import Transformer, count, repeat, tree_map

TRANSFORMER_PARAMS = {
    "N": {"min_val": 1, "max_val": 1e6, "default": 1, "step": 1},
    "l": {"min_val": 1, "max_val": 2**16, "default": 2048, "step": 1},
    "d_model": {"min_val": 1, "max_val": 2**16, "default": 1024, "step": 1},
    "d_ff": {"min_val": 1, "max_val": 2**16, "default": 4096, "step": 1},
    "n_vocab": {"min_val": 1, "max_val": 1e6, "default": 51200, "step": 1},
    "n_layers": {"min_val": 1, "max_val": 1000, "default": 24, "step": 1},
    "n_heads": {"min_val": 1, "max_val": 1000, "default": 16, "step": 1},
}

app = Dash(__name__)

graph = dcc.Graph(id="graph")
sliders = [
    html.Label(
        dcc.Slider(
            v["min_val"],
            v["max_val"],
            v["step"],
            id=f"{k}-slider",
            value=v["default"],
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode="drag",
        ),
        title=k,
    )
    for k, v in TRANSFORMER_PARAMS.items()
]
app.layout = html.Div([html.H1("JupyterDash Demo"), graph, *sliders])


def analysis(
    N,
    l,
    d_model,
    d_ff,
    n_vocab,
    n_layers,
    n_heads,
):
    def count_dict(N, l, d_model, d_ff, n_vocab, n_layers, n_heads):
        return repeat(
            Transformer(l, d_model, d_ff, n_vocab, n_layers, n_heads).flops_dict, N
        )

    keys = {
        "Embeddings",
        "LayerNorm_Final",
        "LayerNorms",
        "PositionWiseFFN",
        "ScaledDotProductAttention",
    }

    _count = lambda d: sum(_count(v) if isinstance(v, dict) else v for v in d.values())

    def gather(d):
        if not isinstance(d, dict):
            return d
        return {k: _count(v) if k in keys else gather(v) for k, v in d.items()}

    flops = count_dict(N, l, d_model, d_ff, n_vocab, n_layers, n_heads)
    total_flops = count(flops)

    counts = gather(flops)
    counts["LayerNorms"] = counts.pop("LayerNorm_Final") + counts["Blocks"].pop(
        "LayerNorms"
    )
    counts.update(counts.pop("Blocks"))
    counts.update(counts.pop("MultiHeadAttention"))

    percentages = tree_map(d=counts, f=lambda x: x / total_flops)
    return counts, percentages, total_flops


# Define callback to update graph
@app.callback(
    Output("graph", "figure"),
    [Input(f"{k}-slider", "value") for k in TRANSFORMER_PARAMS],
)
def update_figure(*args):
    counts, percentages, total_flops = analysis(*args)
    x, y = zip(*percentages.items())
    fig = go.Figure(go.Bar(x=x, y=y))
    fig.update_yaxes(range=[0, 1])
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
