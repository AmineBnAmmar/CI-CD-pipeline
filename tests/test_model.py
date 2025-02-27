from model_pipeline import train_model
import numpy as np

def test_model_training():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = train_model(X, y)
    assert model.n_estimators == 100
