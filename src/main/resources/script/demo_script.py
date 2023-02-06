import joblib
import numpy as np

def user_open(path: str):
    return joblib.load(path)


def user_eval(model, data):
    return model.predict(np.array(data).reshape(1, -1)).tolist()[0]

