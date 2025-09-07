import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="05. Working with Time Series")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 05. Working with Time Series""")
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import statsmodels.graphics.tsaplots as sgt
    import statsmodels.tsa.stattools as sts

    from statsmodels.tsa.seasonal import seasonal_decompose
    return Path, mo, np, pd, plt, seasonal_decompose, sgt, sns, sts


@app.cell
def _(sns):
    sns.set_theme(context="notebook", style="white")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load, transform and simplify the dataset""")
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
        Set date todatetime index.
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
        """Simplify dataset to a single spx market value column."""

        data_copy: pd.DataFrame = data.copy()
        data_copy["market_value"] = data_copy.spx

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
    df_spx: pd.DataFrame = simplify_dataset(df_comp)
    return (df_spx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate test:train split""")
    return


@app.cell
def _(df_spx: "pd.DataFrame"):
    # Generate test train split
    size: int = int(len(df_spx) * 0.8)
    df, df_test = df_spx.iloc[:size].copy(), df_spx.iloc[size:].copy()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## White Noise""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Generate random normally distributed data.
    - For this to be comparable to the spx data, its mean and standard deviation need to be set to that of the actual set.
    """
    )
    return


@app.cell
def _(np, pd):
    def add_white_noise(data: pd.DataFrame) -> pd.DataFrame:
        """Add a white noise column to the input data"""
        data_out = data.copy()

        # Generate random normally distributed data
        # with mean and std comparable to source data
        data_out.loc[:, "white_noise"] = np.random.normal(
            loc=data_out["market_value"].mean(),
            scale=data_out["market_value"].std(),
            size=len(data_out),
        )

        return data_out
    return (add_white_noise,)


@app.cell
def _(add_white_noise, df, mo, pd):
    df_white_noise: pd.DataFrame = add_white_noise(df)

    mo.vstack(
        [
            mo.md("DataFrame description"),
            df_white_noise.describe(),
            df_white_noise.head(),
        ]
    )
    return (df_white_noise,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- Because each value is generated individually, the mean and standard deviation of the generated data are similar but not necessarily the same as spx.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Plot the data""")
    return


@app.cell
def _(df_white_noise: "pd.DataFrame", plt):
    plt.figure(figsize=(20, 5))
    df_white_noise["white_noise"].plot(label="White Noise")
    plt.title("White Noise", size=24)
    y_limits = plt.ylim()  # Needs to be called before plt.show()
    plt.show()
    return (y_limits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For the graphs to be comparable, set the y-axis limits of the S&P500 graph to be the same as the white noise graph.""")
    return


@app.cell
def _(df_white_noise: "pd.DataFrame", plt, y_limits):
    plt.figure(figsize=(20, 5))
    df_white_noise["market_value"].plot(label="S&P500")
    plt.title("S&P500", size=24)
    plt.ylim(y_limits)  # Use the same y limits as the white noise graph
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot both graphs together.""")
    return


@app.cell
def _(df, df_white_noise: "pd.DataFrame", plt):
    plt.figure(figsize=(20, 5))
    df_white_noise["white_noise"].plot(label="White Noise")
    df["market_value"].plot(label="S&P00")
    plt.title("White Noise", size=24)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Random Walk""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - For this workbook, load the existing random walk dataset and add it to the dataframe.
    - The dataset represents prices generated using a random walk.
    """
    )
    return


@app.cell
def _(Path, load_data, pd, set_date_index_frequency):
    # Load random walk data
    random_walk_file: Path = Path.cwd().joinpath("RandWalk.csv")
    random_walk: pd.DataFrame = load_data(random_walk_file)

    # Set index to datetime with business day frequency
    random_walk = set_date_index_frequency(data=random_walk)
    return (random_walk,)


@app.cell
def _(mo, random_walk: "pd.DataFrame"):
    mo.hstack(
        [
            mo.vstack([mo.md("Dataframe"), random_walk.head()]),
            mo.vstack([mo.md("Dataframe description"), random_walk.describe()]),
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Add the random walk data to `df` as a new column.""")
    return


@app.cell
def _(df, random_walk: "pd.DataFrame"):
    df["random_walk"] = random_walk["price"]
    df.head()
    return


@app.cell
def _(df, plt):
    plt.figure(figsize=(20, 5))
    df["market_value"].plot(label="S&P500")
    df["random_walk"].plot(label="Random Walk")
    plt.title("S&P500 vs Random Walk", size=24)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Both show cyclical changes and have variations over time.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Stationarity""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### S&P500""")
    return


@app.function
def print_dickey_fuller(result):
    print(f"t-statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"lags used in regression: {result[2]}")
    print(f"No. of observations used in the analysis: {result[3]}")
    print(f"Augmented Dickey-Fuller critical values: {result[4]}")


@app.cell
def _(df, sts):
    result = sts.adfuller(df["market_value"])
    result
    return (result,)


@app.cell
def _(result):
    print_dickey_fuller(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The t-statistic (-1.73698) is greater than any of the critical values, which means that the null hypothesis that the data is non-stationary cannot be rejected.
    - The p-value (0.41) indicates a 41% chance of not rejecting the null hypothesis, which means that it cannot be confirmed that the data is stationary.
    - Therefore there is no evidence of stationarity.
    - The number of lags used in the regression when determining the t-statistic indicates autocorrelation going back 18 periods. This needs to be taken into account when picking the appropriate model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### White Noise""")
    return


@app.cell
def _(df, sts):
    result_1 = sts.adfuller(df["white_noise"])
    result_1
    return (result_1,)


@app.cell
def _(result_1):
    print_dickey_fuller(result_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The t-statistic is smaller than all the critical values.
    - The p-value is 0.
    - There are no lags. (NB As white noise is stochastic, the number of lags may sometimes vary slightly.)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Random Walk""")
    return


@app.cell
def _(df, sts):
    result_2 = sts.adfuller(df["random_walk"])
    result_2
    return (result_2,)


@app.cell
def _(result_2):
    print_dickey_fuller(result_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The p-value is 0.61, indicating a 61% chance that the data comes from a non-stationary process.
    - The t-statistic is more than any of the critical values.
    - This indicates insufficient evidence that the data is stationary.
    - More often than not, random walk intervals of the same size differ significantly due to the uncertainty of the process. So, each days price might go up or down, but the starting position is always different.
    - Chance dictates that there will be intervals of alternating ups and downs and those with constant runs of increase or decrease. The covariances of two such intervalswith identical size will very rarely be equal. Unlike white noise, these are expected to be a non-stationary process.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Seasonality""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Additive Decomposition""")
    return


@app.cell
def _(df, plt, seasonal_decompose):
    s_dec = seasonal_decompose(df["market_value"], model="additive")
    s_dec.plot()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Trend
      - The trend closely resembles the observed series, as the decomposition function uses the previous period values as the trendsetter.
      - We have already determined that the current period's prices are the best predictor for the next period's prices. If seasonal patterns are observed, we will have other prices as better predictors, e.g. if prices are consistently higher at the beginning of the month compared to the end, it would be better to use values from around 30 periods ago than 1 period ago.
    - Seasonal
      - This appears as a rectangle as the values are constantly oscillating between -0.2 and 0.1, and the figure size is too small. Therefore, there is no concrete cyclical pattern evident using naiive decomposition.
      - Residual
        - The residuals vary greatly around 2000 and 2008, which can be explained by the instability caused by the dotcom and the housing prices bubbles respectively.

    Overall, the results of the additive decomposition suggest no seasonality in the data.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Multiplicative Decomposition""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The results are very similar, which provides further proof that there is no seasonality within S&P500 prices."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### ACF""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - `statsmodels.graphics.tsaplots.plot_acf()` is used.
    - In time series analysis, it is conventional to analyse the first 40 lags.
    - zero is set to False to exclude the current period, as its correlation with itself is 1, and makes the graph more difficult comprehend correctly.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### S&P500""")
    return


@app.cell
def _(df, plt, sgt):
    sgt.plot_acf(df["market_value"], lags=40, zero=False)
    plt.title("ACF: S&P500", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation Coefficient")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The autocorrelation values are outside the blue curve which indicates 95% significance. These are significantly different from 0, which suggests the existence of autocorrelation for that specific lag.
    - The level of significance area expands as the lags increase, indicating that the greater the time difference, the less likely is the autocorrelation to persist. e.g. today’s prices are more likely to be similar to yesterday’s prices than prices a month ago.
    - Therefore, the autocorrelation for higher lags should be significantly different from 0.
    - Since all the lines are significant, it indicates time dependence in the data.
    - The autocorrelation barely decreases upto 40 lags, which suggests that prices even a month back can serve as decent estimators. (**NB** It does change when plotted over a larger range of values)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### White Noise""")
    return


@app.cell
def _(df, plt, sgt):
    sgt.plot_acf(df["white_noise"], lags=40, zero=False, auto_ylims=True)
    plt.title("ACF: White Noise", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation Coefficient")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The autocorrelation coefficients are both positive and negative.
    - Most values fall within the confidence intervals and do not reach significance. This suggests that there is no autocorrelation for any lag.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Random Walk""")
    return


@app.cell
def _(df, plt, sgt):
    sgt.plot_acf(df["random_walk"], lags=40, zero=False, auto_ylims=False)
    plt.title("ACF: Random Walk", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation Coefficient")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- This shows a pattern similar to the S&P500 price data.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### PACF""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### S&P500""")
    return


@app.cell
def _(df, plt, sgt):
    sgt.plot_pacf(
        df["market_value"], lags=40, zero=False, method="ols", auto_ylims=True
    )
    plt.title("PACF: S&P500", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Partial Autocorrelation Coefficient")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Only the first few elements, and a few intermittent elements, are significantly different from 0.
    - Some values, e.g. the 9th lag are negative, meaning higher values 9 days ago resulted in lower values today.
    - The value for lag 1 is the same for ACF and PACF. This is because there are no other channels affecting this lag.
    - Most values after this do not reach significance. Because they are essentially 0 and not significant, being positive or negative is somewhat random without lasting effects.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### White Noise""")
    return


@app.cell
def _(df, plt, sgt):
    sgt.plot_pacf(
        df["white_noise"], lags=40, zero=False, method="ols", auto_ylims=True
    )
    plt.title("PACF: White Noise", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Partial Autocorrelation Coefficient")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - Most of the coefficients do not reach significance.
    - This fits with there being no autocorrelation within white noise.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Random Walk""")
    return


@app.cell
def _(df, plt, sgt):
    sgt.plot_pacf(
        df["random_walk"], lags=40, zero=False, method="ols", auto_ylims=True
    )
    plt.title("PACF: Random Walk", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Partial Autocorrelation Coefficient")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- The pattern is similar to that of S&P500.""")
    return


if __name__ == "__main__":
    app.run()
