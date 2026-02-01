import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="03. Introduction to Time Series")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03. Introduction to Time Series
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy
    import seaborn as sns
    return mo, pd, plt, scipy, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get closing prices of market indices.
    These measure the stability of the following stock exchanges: US, Germany, London, Japan.

    - spx: S&P500
    - dax: DAX 30
    - ftse: FTSE 100
    - nikkei: NIKKEI 225
    """)
    return


@app.cell
def _(pd):
    raw_csv_data = pd.read_csv("Index2018.csv")
    return (raw_csv_data,)


@app.cell
def _(raw_csv_data):
    df_comp = raw_csv_data.copy()
    return (df_comp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Examine the Data
    """)
    return


@app.cell
def _(df_comp):
    df_comp
    return


@app.cell
def _(df_comp, mo):
    mo.vstack(
        [
            mo.md("Get an overview of the numeric fields"),
            df_comp.describe(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The mean of `spx` is much lower than for other indices.
    - Its maximum is lower than the minimums of `ftse` and `nikkei`.
    - Tha values for `dax` and `ftse` are similar, while those of `spx` are far smaller and, `nikkei`, much larger.
    - This difference in magnitude needs to be taken into account when comparing multiple time series.
    """)
    return


@app.cell
def _(df_comp):
    df_comp.info()
    return


@app.cell
def _(df_comp):
    df_comp["spx"].isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Line charts
    """)
    return


@app.cell
def _(sns):
    sns.set_theme(context="notebook", style="white")
    return


@app.cell
def _(df_comp, plt):
    # Plot spx
    df_comp["spx"].plot(figsize=(20, 5), title="S&P500")
    plt.show()

    # Plot ftse
    df_comp["ftse"].plot(figsize=(20, 5), title="FTSE")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The first two peaks are the dotcom and the housing market bubbles respectively. These are mirrored by S&P00 and FTSE. This can be explained by the parallels between the US and the UK stock exchange markets.
    """)
    return


@app.cell
def _(df_comp, plt):
    plt.figure(figsize=(20, 5))
    df_comp["spx"].plot(label="S&P500")
    df_comp["ftse"].plot(label="FTSE100")
    plt.title("S&P500 vs FTSE100")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - When plotted together, the S&P curve flattens. This does not imply relative stability and is due to the difference in magnitude of values between the two indices. This is one of the reasons why time series analysis involves compounded returns along with prices.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The QQ Plot
    """)
    return


@app.cell
def _(df_comp, plt, scipy):
    scipy.stats.probplot(df_comp["spx"], plot=plt)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dataset is not normally distributed, which means that the statistics of normal distributions cannot be used to make successful forecasts. However, this is as expected for time series data.
    """)
    return


if __name__ == "__main__":
    app.run()
