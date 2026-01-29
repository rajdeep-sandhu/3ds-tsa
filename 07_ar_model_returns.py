import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="07. The AR Model - Returns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 07. The AR Model - Prices
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### **Description**

    - Load and simplify price data to use only FTSE prices.
    """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.graphics.tsaplots as sgt
    import statsmodels.tsa.stattools as sts
    from statsmodels.tsa.arima.model import ARIMA

    from tools.metrics_generator import MetricsGenerator
    from tools.model_generator import ModelGenerator

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load and Preprocess dataset
    """
    )
    return


@app.cell
def _(Path, mo, pd):
    @mo.cache
    def load_data(file_path: Path) -> pd.DataFrame:
        print("Reading from disk")
        return pd.read_csv(file_path)

    return


if __name__ == "__main__":
    app.run()
