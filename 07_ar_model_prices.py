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
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import statsmodels.graphics.tsaplots as sgt
    import statsmodels.tsa.stattools as sts
    from statsmodels.tsa.arima.model import ARIMA

    from pathlib import Path
    from tools.metrics_generator import MetricsGenerator
    from tools.model_generator import ModelGenerator
    from typing import Any
    return (
        ARIMA,
        Any,
        MetricsGenerator,
        ModelGenerator,
        Path,
        mo,
        pd,
        plt,
        sgt,
        sts,
    )


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The PACF""")
    return


@app.cell
def _(df, mo, plt, sgt):
    sgt.plot_pacf(
        df["market_value"],
        alpha=0.05,
        zero=False,
        lags=40,
        method="ols",
        auto_ylims=True,
    )

    plt.title("PACF: FTSE Prices", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Partial Autocorrelation Coefficient")
    mo.as_html(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The first lag coefficient is greatly significant and **must** be included in the model.
    - Coefficients from lag 25 onwards are not significant and can be ignored. Since their values will be very close to 0, their impact on the model will be minimal.
    - The model should therefore include less than 25 lags.
    - A business month is 22 days, which means there will be cyclical changes. Values a month ago negatively affect the values today. However, these are overshadowed by more recent lags and their contribution should not be overanalysed.
    - NB Patterns are not always so convenient to spot.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The AR(1) Model""")
    return


@app.cell
def _(ARIMA, df):
    model_prices = ARIMA(df["market_value"], order=(1, 0, 0))
    result_prices = model_prices.fit()
    result_prices.summary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - `const` refers to $C$. `coef` is its value.
    - `ar.L1` refers to `'market_value`. `coef` refers to $\phi_1$, which is the coefficient for the autoregressive value for 1 time period ago ($t-1$).
      - The coefficient is close to 1, which is similar to what the ACF and PACF graph indicate.

    - As the $P$ values for both are 0.00, both $C$ and $\phi_1$ for market value are significant.

    - As the **critical values** for both do not contain 0, both the coefficients are significant.

    Since both values are significantly different from 0, a more complex model can be tried for greater accuracy.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Higher-Lag AR Models""")
    return


@app.cell
def _(ARIMA, Any, ModelGenerator):
    def generate_models(data: Any, max_lags: int) -> ModelGenerator:
        model_generator = ModelGenerator(data=data)
        max_lags = 9
        param_grid = [{"order": (p, 0, 0)} for p in range(1, max_lags + 1)]
        model_generator.generate_models(
            model_function=ARIMA, model_name_prefix="AR", param_grid=param_grid
        )
        return model_generator
    return (generate_models,)


@app.cell
def _(df, generate_models):
    max_lags = 9
    model_generator_prices = generate_models(data=df["market_value"], max_lags=max_lags)
    return (model_generator_prices,)


@app.cell
def _(model_generator_prices):
    model_generator_prices.summarise_results()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **AR(2)**

    - The coefficients for $C$ and $\phi_1$ have changed. This is because some of the changes contributing to the present value can be attributed to lag 2.
    - `ar.L2` has been added for lag 2. As the $P$ value for L2 is > 0.05, it indicates that $\phi_2$ is not significantly different from 0. Also, its **critical value** range contains 0.
    - The log likelihood is slightly higher than that for AR(1).

    **AR(3)**

    - $\phi_2$ is now negative. Its P value is below 0.05, which indicates significance.
    - The P value of $\phi_3$ is 0, which indicates that it is significant.
    - The log likelihood is higher than both AR(1) and AR(2)

    **AR(4)**

    - The P value of $\phi_3$ is more than 0.05, which indicates that it is not significant.
    - The log likelihood is higher than the AR(1) to AR(3) models, which suggests that the model is capturing more variation in the data.
    - However, the insignificance of $\phi_3$ raises concerns about overfitting.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Create a dataframe to tabulate measures of interest""")
    return


@app.cell
def _(MetricsGenerator, model_generator_prices):
    metrics_prices = MetricsGenerator(models=model_generator_prices.models)
    metrics_prices.generate_metrics_table()
    metrics_prices.evaluation
    return (metrics_prices,)


@app.cell
def _(metrics_prices):
    # Find models where both the final lag and the LLR Test p-values fail to reach significance.
    metrics_prices.evaluation.query(
        "final_lag_pval >= 0.05 and llr_test_pval >= 0.05"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Plot the test metrics.**""")
    return


@app.cell
def _(metrics_prices, mo, plt):
    # Create 2 subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot final lag p-value on the first subplot
    metrics_prices.evaluation[["final_lag_pval", "llr_test_pval"]].plot(ax=axes[0])
    axes[0].set_title("P-Values")
    axes[1].set_xlabel("Model")
    axes[0].set_ylabel("P-Value")

    # Plot AIC, BIC, HQIC on the second subplot
    metrics_prices.evaluation[["aic", "bic", "hqic"]].plot(ax=axes[1])
    axes[1].set_title("Model Evaluation")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Metric Value")
    axes[1].legend(loc="best")

    plt.tight_layout()
    mo.as_html(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The first model which satisfies the p-values of both the final lag and the LLR Test being non-significant is AR(2). The next is AR(8)
    - Model AR(7) is selected and a LLR Test is performed against AR(1) to confirm significance.
    """
    )
    return


@app.cell
def _(metrics_prices):
    # Calculate degrees of freedom from the maximum lags of each model
    deg_freedom_prices = (
        metrics_prices.evaluation.loc["AR_7_0_0", "ar"]
        - metrics_prices.evaluation.loc["AR_1_0_0", "ar"]
    )

    metrics_prices.llr_test(
        metrics_prices.evaluation.loc["AR_1_0_0", "llf"],
        metrics_prices.evaluation.loc["AR_7_0_0", "llf"],
        df=deg_freedom_prices,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The returned p-value indicates that the AR_7 model is significanlty better than the AR_1 model."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Analysing the Residuals""")
    return


@app.cell
def _(df, model_generator_prices):
    # Get residuals from the AR_7 model result
    df["residuals_price"] = model_generator_prices.get_model("AR_7_0_0")[1].resid
    return


@app.cell
def _(df, mo):
    mo.vstack(
        [
            f"Mean: {df['residuals_price'].mean()}",
            f"Variance: {df['residuals_price'].var()}",
        ]
    )
    return


@app.cell
def _(df, mo):
    mo.as_html(df["residuals_price"].plot.box())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The mean is close to 0, which suggests that the model performs well.
    - However, the high variance indicates that the model might not perform well.
    """
    )
    return


@app.cell
def _(df, sts):
    # ADF Test
    sts.adfuller(df["residuals_price"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- The **ADF** t-statistic is much more negative than the the 5% critical value and the p-value is 0, both of which suggest stationarity.""")
    return


@app.cell
def _(df, mo, plt, sgt):
    # Plot the ACF for residuals
    sgt.plot_acf(df["residuals_price"], zero=False, lags=40, auto_ylims=True)
    plt.title("ACF: FTSE Price Residuals", size=24)
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation Coefficient")
    mo.as_html(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The majority of residuals are not significantly different from 0, which fits the characteristics of white noise. However, the 3 values that are significantly different from 0 indicate that there might be a better predictor.""")
    return


@app.cell
def _(df, mo, plt):
    # Plot the residual time series.
    # The first row is dropped as it is an outlier, which is expected for AR model residuals.
    df["residuals_price"][1:].plot(figsize=(20, 5))
    mo.as_html(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - The price residuals are mostly low and the time series does not indicate an obvious pattern, so the choice of model seems correct.
    - However, since an AR model is being used on non-stationary data, the predictions might still be incorrect.
    """
    )
    return


if __name__ == "__main__":
    app.run()
