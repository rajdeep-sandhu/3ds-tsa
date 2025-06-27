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

    def _name_model(
        self,
        model_function: Callable[..., Any],
        params: dict = None,
        index: int = None,
        prefix: str = None,
    ):
        """
        Generates model name from order.
        If not provided or missing, uses the index.

        ::parameters::
        - model_function: The function or class to create the model (e.g., ARIMA, LinearRegression).
        - prefix: str. Optional. If provided, used instead of model_function.
        - params: dict. Optional. Should contain order argument as tuple for ARIMA models.
        - index: int. Optional. Index to use if no order is available.
        """

        order = params.get("order", None) if params else None

        model_name = str(prefix) if prefix else model_function.__name__

        if (order and index) or (order is None and index is None):
            raise ValueError(
                "Cannot generate model name: Provide one of either a valid 'params' with 'order', or 'index'."
            )

        if order and isinstance(order, tuple) and len(order) == 3:
            return f"{model_name}_{order[0]}_{order[1]}_{order[2]}"

        if index and isinstance(index, int):
            return f"{model_name}_{index}"

        raise ValueError("Cannot generate model name: Invalid arguments.")

    def generate_models(
        self,
        model_function: Callable[..., Any],
        param_grid: list[dict],
        model_name_prefix: str = None,
    ) -> None:
        """
        Generate models using the provided model function and parameter grid.

        params:
        - model_function: The function or class to create the model (e.g., ARIMA, LinearRegression).
        - model_name_prefix: A prefix for naming the models.
        - param_grid: A list of dictionaries of parameters to iterate over.
        """

        for i, params in enumerate(param_grid):
            # Generate model name
            model_name: str = self._name_model(
                model_function=model_function,
                params=params,
                model_name_prefix=model_name_prefix,
            )
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
