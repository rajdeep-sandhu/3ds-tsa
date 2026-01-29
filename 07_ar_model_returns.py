import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="07. The AR Model - Returns")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 07. The AR Model - Prices
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import statsmodels.graphics.tsaplots as sgt
    import statsmodels.tsa.stattools as sts
    from statsmodels.tsa.arima.model import ARIMA

    from tools.metrics_generator import MetricsGenerator
    from tools.model_generator import ModelGenerator
    return (mo,)


if __name__ == "__main__":
    app.run()
