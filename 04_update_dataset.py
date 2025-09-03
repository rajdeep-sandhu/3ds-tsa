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

        close_prices = data.copy()
    
        # Add new columns to the data set
        close_prices['spx'] = close_prices['^GSPC'].Close
        close_prices['dax'] = close_prices['^GDAXI'].Close
        close_prices['ftse'] = close_prices['^FTSE'].Close
        close_prices['nikkei'] = close_prices['^N225'].Close

        # Delete original columns
        del close_prices['^N225'], close_prices['^GSPC'], close_prices['^GDAXI'], close_prices['^FTSE']

        return close_prices
    return (get_close_prices,)


@app.cell
def _(pd):
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """Return cleaned dataset."""
        df_comp_cleaned: pd.DataFrame = data.copy()

        # Remove first row to start on a Monday
        df_comp_cleaned = df_comp_cleaned.iloc[1:]

        # Set index to business days.
        df_comp_cleaned = df_comp_cleaned.asfreq("b")

        # Forward fill missing data
        df_comp_cleaned = df_comp_cleaned.fillna(method="ffill")

        return df_comp_cleaned
    return (clean_data,)


@app.cell
def _(clean_data, df_comp: "pd.DataFrame", get_close_prices, pd):
    df_comp_close: pd.DataFrame = clean_data(get_close_prices(df_comp))
    df_comp_close
    return


if __name__ == "__main__":
    app.run()
