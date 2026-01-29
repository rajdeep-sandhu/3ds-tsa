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
def _(mo):
    mo.md(r"""
    #### **Description**

    - Load and simplify price data to use only FTSE prices.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import statsmodels.graphics.tsaplots as sgt
    import statsmodels.tsa.stattools as sts
    from statsmodels.tsa.arima.model import ARIMA

    from tools.metrics_generator import MetricsGenerator
    from tools.model_generator import ModelGenerator
    return Path, mo, pd, plt, sgt, sns, sts


@app.cell
def _(sns):
    # set style using seaborn, although charts are handled by matplotlib.pyplot
    sns.set_theme(context="notebook", style="white")
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
    return (raw_csv_data,)


@app.cell
def _(pd):
    def set_date_index_frequency(data: pd.DataFrame) -> pd.DataFrame:
        """
        Set date to datetime index.
        Set frequency to business days
        """

        data_out: pd.DataFrame = data.copy()
        data_out["date"] = pd.to_datetime(data_out["date"], dayfirst=True)
        data_out = data_out.set_index("date")
        data_out = data_out.asfreq("b")

        return data_out
    return (set_date_index_frequency,)


@app.cell
def _(pd, set_date_index_frequency):
    def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
        """Clean the provided dataset."""
        df_cleaned: pd.DataFrame = data.copy()

        # Set date as index with frequency as business days.
        df_cleaned = set_date_index_frequency(data=df_cleaned)

        # Forward fill missing values
        df_cleaned = df_cleaned.ffill()

        return df_cleaned
    return (clean_dataset,)


@app.cell
def _(pd):
    def simplify_dataset(data: pd.DataFrame) -> pd.DataFrame:
        """Simplify dataset to a single ftse market value column."""

        data_copy: pd.DataFrame = data.copy()
        data_copy["market_value"] = data_copy.ftse

        del data_copy["spx"]
        del data_copy["dax"]
        del data_copy["ftse"]
        del data_copy["nikkei"]

        return data_copy
    return (simplify_dataset,)


@app.cell
def _(clean_dataset, pd, raw_csv_data: "pd.DataFrame"):
    df_comp: pd.DataFrame = clean_dataset(raw_csv_data)
    return (df_comp,)


@app.cell
def _(df_comp: "pd.DataFrame", pd, simplify_dataset):
    df_ftse: pd.DataFrame = simplify_dataset(df_comp)
    df_ftse
    return (df_ftse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate test:train split
    """)
    return


@app.cell
def _(df_ftse: "pd.DataFrame"):
    size = int(len(df_ftse) * 0.8)
    df, df_test = df_ftse.iloc[:size].copy(), df_ftse.iloc[size:].copy()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stationarity ADF Test
    """)
    return


@app.cell
def _(df, sts):
    sts.adfuller(df["market_value"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The t-statistic (-1.90) is higher than the 5% critical value.
    - The p-value is higher than 0.05.
    - The null hypothesis **cannot be rejected** and the time series is **non-stationary**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use Returns instead of Prices
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Because price data is non-stationary, an AR model is not suitable.
    - However, it can be transformed into returns so that it fits the assumptions of stationarity.
    """)
    return


@app.cell
def _(df, pd):
    # Calculate simple returns and convert to a percentage
    df_returns: pd.DataFrame = df.copy()
    df_returns["returns"] = (
        df_returns["market_value"].pct_change(periods=1).mul(100)
    )

    # Remove the first row as returns cannot be calculated for the first row
    df_returns = df_returns[1:]
    df_returns
    return (df_returns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perform an ADF on the returns
    """)
    return


@app.cell
def _(df_returns: "pd.DataFrame", sts):
    sts.adfuller(df_returns["returns"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The t-statistic (-12.77) is more negative than the 5% critical value.
    - The computed p-value is lower than 0.05.
    - Both are significant. The null hypothesis can therefore be rejected, indicating that the data is meets the assumptions of stationarity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ACF and PACF for Returns
    """)
    return


@app.cell
def _(df_returns: "pd.DataFrame", mo, plt, sgt):
    sgt.plot_acf(df_returns["returns"], zero=False, lags=40, auto_ylims=True)
    plt.title("ACF: FTSE Returns", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation Coefficient")
    mo.as_html(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The ACF graph is very different from that for prices.
    - The coefficients vary in sign, magnitude and significance.
    - The first few lags are predomnantly significant and predominantly negative. This indicates that consecutive returns move in different directions.
    - This suggests that returns oevr the entire week are relevant to the current one. (NB A business weekis 5 days.)
    - The negative relationship can be interpreted as some form of natural adjustment occuring in the market.
    """)
    return


if __name__ == "__main__":
    app.run()
