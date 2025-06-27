import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima.model import ARIMA

from tools.model_generator import ModelGenerator


class DummyModel:
    # Class that provides a dummy name
    __name__: str = "DummyModel"


@pytest.fixture
def generator():
    data: pd.Series = pd.Series(np.random.normal(size=100))
    return ModelGenerator(data=data)


def test_name_model_with_order(generator):
    """Test that the name is generated from the function name."""
    params: dict = {"order": (1, 2, 3)}
    name: str = generator._name_model(model_function=DummyModel, params=params)
    assert name == "DummyModel_1_2_3"


def test_name_model_with_prefix(generator):
    """Test that prefix supersedes the provided function name."""
    params: dict = {"order": (1, 2, 3)}
    name: str = generator._name_model(model_function=DummyModel, params=params, prefix="TestPrefix")
    assert name == "TestPrefix_1_2_3"

def test_name_model_with_index(generator):
    name = generator._name_model(DummyModel, params={}, index=4)
    assert name == "DummyModel_4"


def test_name_model_both_order_and_index(generator):
    params = {"order": (1, 2, 3)}
    with pytest.raises(ValueError, match="Provide one of either"):
        generator._name_model(DummyModel, params=params, index=1)

def test_name_model_neither_order_or_index(generator):
    with pytest.raises(ValueError, match="Provide one of either"):
        generator._name_model(DummyModel, params={})


def test_name_model_order_wrong_length(generator):
    params = {"order": (1, 2)}  # too short
    with pytest.raises(ValueError, match="Invalid arguments"):
        generator._name_model(DummyModel, params=params)


def test_name_model_order_wrong_type(generator):
    params = {"order": "not_a_tuple"}  # too short
    with pytest.raises(ValueError, match="Invalid arguments"):
        generator._name_model(DummyModel, params=params)


def test_name_model_index_wrong_type(generator):
    with pytest.raises(ValueError, match="Invalid arguments"):
        generator._name_model(DummyModel, index="not_an_int")
