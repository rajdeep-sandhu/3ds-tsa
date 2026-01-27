import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="07. The AR Model - Prices")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 07. The AR Model - Prices""")
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
    import seaborn as sns
    import statsmodels.graphics.tsaplots as sgt
    import statsmodels.tsa.stattools as sts
    from statsmodels.tsa.arima.model import ARIMA

    from tools.metrics_generator import MetricsGenerator
    from tools.model_generator import ModelGenerator
    return ARIMA, ModelGenerator, Path, mo, pd, plt, sgt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load and Preprocess dataset""")
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
    df_comp
    return (df_comp,)


@app.cell
def _(df_comp: "pd.DataFrame", pd, simplify_dataset):
    df_ftse: pd.DataFrame = simplify_dataset(df_comp)
    return (df_ftse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate test:train split""")
    return


@app.cell
def _(df_ftse: "pd.DataFrame"):
    size = int(len(df_ftse) * 0.8)
    df, df_test = df_ftse.iloc[:size].copy(), df_ftse.iloc[size:].copy()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The ACF""")
    return


@app.cell
def _(df, mo, plt, sgt):
    sgt.plot_acf(df["market_value"], zero=False, lags=40, auto_ylims=True)
    plt.title("ACF: FTSE Prices", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation Coefficient")
    mo.as_html(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The coefficients slowly decline with lags.
    - All are positive.
    - All are significant.
    - This is similar to the ACF for the S&P500 conducted previously.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - A higher number of lags means a more coefficients and a better fit but makes the model prone to overfitting and poor generalisation.
    - A parsimonious model with fewer lags is better.
    - An efficient model should only include lags which have a **direct**, **significant** effect on the present value. This is determined using the PACF.
    """
    )
    return


if __name__ == "__main__":
    app.run()
