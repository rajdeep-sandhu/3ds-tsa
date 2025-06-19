class ModelGenerator:
    """Generate a series of models."""

    def __init__(self, data):
        """
        Initialise the ModelGenerator.

        params:
        data: The dataset to use for model generation
        model_function: The function to generate models (e.g. ARIMA)
        """
        self.data = data
        self.models = {}

    def generate_models(self, max_lag, model_function, **kwargs):
        """
        Generate a series of models from 1 to a number of lags using the provided function.

        params:
        - max_lag: The maximum lag for generating the model series.
        - kwargs: Additional arguments to pass to the model function.
        """

        for lags in range(1, max_lag + 1):
            model_name = f"{model_function.__name__}_{lags}"
            model = model_function(self.data, order=(lags, 0, 0), **kwargs)
            result = model.fit()
            self.models[model_name] = (model, result)

    def get_model(self, name):
        """Return a tuple with a (model, result) with the given name."""
        return self.models.get(name)

    def summarise_results(self):
        "Display summaries of all results."

        for model_name in self.models:
            _, result = self.get_model(model_name)
            print(f"{model_name}:")
            print(result.summary(), end="\n" * 3)

