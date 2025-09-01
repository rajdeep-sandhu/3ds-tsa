import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="04. Create a Time Series Object")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 04. Create a Time Series Object in Python""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import the data""")
    return


@app.cell
def _(pd):
    raw_csv_data = pd.read_csv("Index2018.csv")
    df_comp = raw_csv_data.copy()
    return (df_comp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Describe the dataset and convert text date to datetime""")
    return


@app.cell
def _(df_comp, mo):
    mo.vstack(
        [
            mo.md(f"Before conversion to `datetime`"),
            df_comp["date"].describe(),
        ]
    )
    return


@app.cell
def _(df_comp, pd):
    df_comp["date"] = pd.to_datetime(df_comp["date"], dayfirst=True)
    df_comp
    return


@app.cell
def _(df_comp, mo):
    mo.vstack(
        [
            mo.md("After conversion to `datetime`"),
            df_comp["date"].describe(),
            mo.stat(
                value=df_comp["date"].nunique(),
                label="Unique Dates",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Setting the Index""")
    return


@app.cell
def _(df_comp):
    df_indexed = df_comp.set_index("date")
    df_indexed.head()
    return (df_indexed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Setting the Desired Frequency""")
    return


@app.cell
def _(df_indexed):
    df_indexed_1 = df_indexed.asfreq("b")
    df_indexed_1.head()
    return (df_indexed_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Handling Missing Values
    - This is for illustration only. Practically, a consistent approach should be used.
    """
    )
    return


@app.cell
def _(df_indexed_1):
    df_filled = df_indexed_1.copy()
    df_filled.isna().sum()
    return (df_filled,)


@app.cell
def _(df_filled):
    # Forward fill spx
    df_filled["spx"] = df_filled["spx"].ffill()
    df_filled.isna().sum()
    return


@app.cell
def _(df_filled):
    # Backfill ftse
    df_filled["ftse"] = df_filled["ftse"].bfill()
    df_filled.isna().sum()
    return


@app.cell
def _(df_filled):
    # Fill dax and nikkei with their mean values
    for ticker in ["dax", "nikkei"]:
        mean_price = df_filled[ticker].mean()
        df_filled[ticker] = df_filled[ticker].fillna(mean_price)

    df_filled.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Simplifying the Dataset""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Simplify the dataframe to keep only `spx` market values.""")
    return


@app.cell
def _(df_indexed_1):
    df_spx = df_indexed_1[["spx"]].rename({"spx": "market_value"}, axis="columns")
    df_spx.head()
    return (df_spx,)


@app.cell
def _(df_spx):
    df_spx.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Splitting the Data""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Time series data cannot be shuffled.
    - A cut off point is used. Data prior to this is assigned to the training set. Data following this is assigned to the testing set.
    - An 80:20 split for training and testing data is reasonable to prevent overfitting and maintain accuracy.
    """
    )
    return


@app.cell
def _(df_spx):
    train_size = int(df_spx.shape[0] * 0.8)
    train_size
    return (train_size,)


@app.cell
def _(df_spx, train_size):
    df_train = df_spx.iloc[:train_size]
    df_test = df_spx.iloc[train_size:]
    return df_test, df_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Compare the tail of df_train with the head of df_test to ensure the last value does not overlap"""
    )
    return


@app.cell
def _(df_train):
    df_train.tail()
    return


@app.cell
def _(df_test):
    df_test.head()
    return


if __name__ == "__main__":
    app.run()
