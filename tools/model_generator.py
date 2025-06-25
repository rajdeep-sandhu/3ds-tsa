from typing import Any, Callable


class ModelGenerator:
    """Generate a series of models."""

    def __init__(self, data: Any, target: Any = None) -> None:
        """
        Initialise the ModelGenerator.

        params:
        - data: The dataset to use for model generation
        - target: The target variable (for supervised models).
        """
        self.data = data
        self.target = target
        self.models = {}

    def generate_models(
        self,
        model_function: Callable[..., Any],
        model_name_prefix: str,
        param_grid: list[dict]
    ) -> None:
        """
        Generate models using the provided model function and parameter grid.

        params:
        - model_function: The function or class to create the model (e.g., ARIMA, LinearRegression).
        - model_name_prefix: A prefix for naming the models.
        - param_grid: A list of dictionaries of parameters to iterate over.
        """

        for i, params in enumerate(param_grid):
            model_name = f"{model_name_prefix}_{i + 1}"
            model = model_function(self.data, **params)

            if self.target is not None:
                result = model.fit(self.target)
            else:
                result = model.fit()
            
            self.models[model_name] = (model, result)

    def get_model(self, name: str) -> tuple[Any, Any] | None:
        """Return a tuple with a (model, result) with the given name."""
        return self.models.get(name)

    def summarise_results(self) -> None:
        """Display summaries of all results."""

        for model_name in self.models:
            _, result = self.get_model(model_name)
            print(f"{model_name}:")
            print(result.summary(), end="\n" * 3)
