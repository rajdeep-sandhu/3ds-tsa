import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="04. Update Dataset")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Update Dataset from Yahoo! Finance
    This illustrates updating the dataset from the Yahoo! Finance API. However, the data is not saved, as the provided dataset will be used for analysis.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import yfinance
    import warnings
    return mo, pd, warnings, yfinance


@app.cell
def _(warnings):
    # Ignoring warning messages
    warnings.filterwarnings("ignore")
    return


@app.cell
def _(mo, pd, yfinance):
    # Download the data
    @mo.cache
    def download_data() -> pd.DataFrame:
        """Download stock data from Yahoo! Finance."""
        return yfinance.download(
            tickers="^GSPC ^FTSE ^N225 ^GDAXI",
            start="1994-01-07",
            end="2019-09-27",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
        )

    raw_data: pd.DataFrame = download_data()
    return (raw_data,)


@app.cell
def _(pd, raw_data: "pd.DataFrame"):
    # Work on a copy of the downloaded data
    df_comp: pd.DataFrame = raw_data.copy()
    df_comp
    return (df_comp,)


@app.cell
def _(pd):
    def get_close_prices(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create columns for closing prices.
        Delete original columns.
        """
        # Adding new columns to the data set
        data['spx'] = data['^GSPC'].Close
        data['dax'] = data['^GDAXI'].Close
        data['ftse'] = data['^FTSE'].Close
        data['nikkei'] = data['^N225'].Close

        # Delete original columns
        del data['^N225'], data['^GSPC'], data['^GDAXI'], data['^FTSE']

        return data
    return (get_close_prices,)


@app.cell
def _(df_comp: "pd.DataFrame", get_close_prices, pd):
    df_comp_1: pd.DataFrame = get_close_prices(df_comp)
    df_comp_1 = df_comp.iloc[1:]

    df_comp_1 = df_comp_1.asfreq('b')
    df_comp_1 = df_comp_1.fillna(method='ffill')
    df_comp_1
    return (df_comp_1,)


@app.cell
def _(df_comp_1: "pd.DataFrame"):
    print(df_comp_1.head())
    print(df_comp_1.tail())
    return


if __name__ == "__main__":
    app.run()
