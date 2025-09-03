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
    return mo, warnings, yfinance


@app.cell
def _(warnings):
    # Ignoring warning messages
    warnings.filterwarnings("ignore")
    return


@app.cell
def _(mo, yfinance):
    # Download the data
    @mo.cache
    def download_data():
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

    raw_data = download_data()
    return (raw_data,)


@app.cell
def _(raw_data):
    # Work on a copy of the downloaded data
    df_comp = raw_data.copy()
    df_comp
    return (df_comp,)


@app.cell
def _(df_comp):
    # Adding new columns to the data set
    df_comp['spx'] = df_comp['^GSPC'].Close
    df_comp['dax'] = df_comp['^GDAXI'].Close
    df_comp['ftse'] = df_comp['^FTSE'].Close
    df_comp['nikkei'] = df_comp['^N225'].Close
    df_comp.head()
    return


@app.cell
def _(df_comp):
    df_comp_1 = df_comp.iloc[1:]
    del df_comp_1['^N225']
    del df_comp_1['^GSPC']
    del df_comp_1['^GDAXI']
    del df_comp_1['^FTSE']
    df_comp_1 = df_comp_1.asfreq('b')
    df_comp_1 = df_comp_1.fillna(method='ffill')
    df_comp_1
    return (df_comp_1,)


@app.cell
def _(df_comp_1):
    print(df_comp_1.head())
    print(df_comp_1.tail())
    return


if __name__ == "__main__":
    app.run()
