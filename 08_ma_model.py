import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="08. The MA Model")


@app.cell
def _(mo):
    mo.md(r"""
    # 08. The MA Model
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import statsmodels.graphics.tsaplots as sgt
    import statsmodels.tsa.stattools as sts
    from statsmodels.tsa.arima.model import ARIMA

    from tools.metrics_generator import MetricsGenerator
    from tools.model_generator import ModelGenerator
    return mo, sns


@app.cell
def _(sns):
    # set style using seaborn, although charts are handled by matplotlib.pyplot
    sns.set_theme(context="notebook", style="white")
    return


if __name__ == "__main__":
    app.run()
