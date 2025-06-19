import numpy as np
import pandas as pd

from scipy.stats.distributions import chi2


class MetricsGenerator:
    def __init__(self, models):
        """
        Initialise the MetricsGenerator.

        params:
        models: dictionary of models and their results.
        """
        self.models = models
        self.evaluation = None

    def llr_test(self, llf_1, llf_2, df=1):
        """
        Perform the Likelihood Ratio Test (LLR) to compare 2 nested models.

        Params:
        llf_1: Log-likelihood of the simpler model.
        llf_2: Log-likelihood of the more complex model.
        df: Degrees of freedom (difference in the number of parameters between models).

        Returns:
        p-value
        """
        lr_stat = 2 * (llf_2 - llf_1)
        p_value = chi2.sf(lr_stat, df)

        return p_value

    def generate_metrics_table(self):
        self.evaluation = pd.DataFrame(
            columns=[
                "model",
                "ar",
                "llf",
                "aic",
                "bic",
                "hqic",
                "final_lag",
                "final_lag_pval",
            ]
        ).set_index("model")

        for model_name, (_, result) in self.models.items():
            self.evaluation.loc[model_name] = {
                "ar": result.model_orders["ar"],
                "llf": result.llf,
                "aic": result.aic,
                "bic": result.bic,
                "hqic": result.hqic,
                "final_lag": result.pvalues.index[-2],
                "final_lag_pval": result.pvalues.values[-2].round(5),
            }

        # Label which model will be compared with which.
        # Assumptions:
        ## Models are in increasing lags of 1.
        ## Each row is tested against the previous row
        ## The first row results are therefore NaN.
        self.evaluation["llr_test_models"] = self.evaluation.apply(
            lambda row: (
                f"{row.name} vs {self.evaluation.index[self.evaluation.index.get_loc(row.name)-1]}"
                if self.evaluation.index.get_loc(row.name) > 0
                else np.nan
            ),
            axis=1,
        )

        # Test each row's llf against the previous row's llf
        self.evaluation["llr_test_pval"] = self.evaluation.apply(
            lambda row: (
                self.llr_test(
                    self.evaluation["llf"].shift(1).loc[row.name], row["llf"], df=1
                )
            ),
            axis=1,
        )
    
