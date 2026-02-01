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
    import pandas as pd

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import the data""")
    return


@app.cell
def _(mo, pd):
    # Cache to avoid reloading each time a cell is rerun.
    @mo.cache
    def load_csv_data():
        return pd.read_csv("Index2018.csv")

    df_comp = load_csv_data().copy()
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
    mo.md(r"""## Set the Index to `date`""")
    return


@app.cell
def _(df_comp):
    # Set the index
    df_indexed = df_comp.set_index("date")
    df_indexed
    return (df_indexed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Set the frequency to business days
    This is based on financial data being on weekdays.
    """
    )
    return


@app.cell
def _(df_indexed):
    df_indexed_1 = df_indexed.asfreq("b")
    df_indexed_1
    return (df_indexed_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Handle Missing Values
    - Different approaches are used here for illustration only. In practice, a consistent approach should be used, depending on how best to handle missing values in that partivular series or dataset.
    """
    )
    return


@app.cell
def _(mo, pd):
    def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using different approaches (for demonstration).
        For marimo, all steps involving reassigning to the same variable need to be in a single cell,
        or wrapped in a function.
        Otherwise, rerunning after changing code uses the final value and causes errors.
        """

        df_filled = data.copy()
        mo.output.replace(df_filled.isna().sum())

        # Forward fill spx
        df_filled["spx"] = df_filled["spx"].ffill()
        mo.output.append(
            mo.vstack([mo.md("Forward fill `spx`"), df_filled.isna().sum()])
        )

        # Backfill ftse
        df_filled["ftse"] = df_filled["ftse"].bfill()
        mo.output.append(mo.vstack([mo.md("Backfill `ftse`"), df_filled.isna().sum()]))

        # Fill dax and nikkei with their mean values
        for ticker in ["dax", "nikkei"]:
            mean_price = df_filled[ticker].mean()
            df_filled[ticker] = df_filled[ticker].fillna(mean_price)

        mo.output.append(
            mo.vstack(
                [
                    mo.md("Fill `dax` and `nikkei` with their mean values"),
                    df_filled.isna().sum(),
                ]
            )
        )

        return df_filled

    return (fill_missing_values,)


@app.cell
def _(df_indexed_1, fill_missing_values):
    df_filled = fill_missing_values(data=df_indexed_1)
    return (df_filled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Simplify the Dataset
    Keep only `spx` market values.
    """
    )
    return


@app.cell
def _(df_filled, mo):
    df_spx = df_filled[["spx"]].rename({"spx": "market_value"}, axis="columns")
    mo.vstack(
        [
            mo.md("Keep only `spx`"),
            df_spx.head(),
            mo.md("Decribe the data"),
            df_spx.describe(),
        ]
    )
    return (df_spx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Split the Data
    - Time series data cannot be shuffled.
    - A cut off point is used. Data prior to this is assigned to the training set. Data following this is assigned to the testing set.
    - An 80:20 split for training and testing data is reasonable to prevent overfitting and maintain accuracy.
    """
    )
    return


@app.cell
def _(df_spx, mo):
    train_size = int(df_spx.shape[0] * 0.8)
    mo.stat(value=train_size, label="train_size", caption="Training Set Size")
    return (train_size,)


@app.cell
def _(df_spx, train_size):
    df_train = df_spx.iloc[:train_size]
    df_test = df_spx.iloc[train_size:]
    return df_test, df_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(df_test, df_train, mo):
    mo.vstack(
        [
            mo.md(
                "Compare the tail of df_train with the head of df_test to ensure the last value does not overlap"
            ),
            mo.hstack([df_train.tail(), df_test.head()], justify="space-around"),
        ],
    )
    return


if __name__ == "__main__":
    app.run()
