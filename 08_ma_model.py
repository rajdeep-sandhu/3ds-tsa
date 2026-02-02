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
    return Path, mo, pd, sns


@app.cell
def _(sns):
    # set style using seaborn, although charts are handled by matplotlib.pyplot
    sns.set_theme(context="notebook", style="white")

    # CSS style for results
    RESULT_CSS_STYLE = {
        "font-family": "monospace",
        "white-space": "pre-wrap",
        "padding": "15px",
        "max-width": "100%",
        "font-size": "1.1em",
        "background-color": "#f9f9f9",
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and Preprocess Dataset
    """)
    return


@app.cell
def _(Path, mo, pd):
    @mo.cache
    def load_data(file_path: Path) -> pd.DataFrame:
        print("Reading from disk")
        return pd.read_csv(file_path)
    return (load_data,)


@app.cell
def _(Path, load_data, pd):
    csv_file: Path = Path.cwd().joinpath("Index2018.csv")
    raw_csv_data: pd.DataFrame = load_data(csv_file)
    return


if __name__ == "__main__":
    app.run()
