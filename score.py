import os          # <- NUEVO: para unir rutas
import json
import joblib
import pandas as pd
from azureml.core.model import Model
from sklearn.preprocessing import OrdinalEncoder


def init():
    global model
    try:
     
        model_target = Model.get_model_path("model.pkl")

        # 2) Si el registro era una carpeta, construye la ruta al archivo
        model_path = (os.path.join(model_target, "model.pkl")
                      if os.path.isdir(model_target) else model_target)

        # 3) Carga el modelo
        model = joblib.load(model_path)
        print(f"Modelo cargado desde: {model_path}")

    except Exception as e:
        model = None              # evita NameError en run()
        print("Error en init():", e)


def run(raw_data):
    try:
        data = json.loads(raw_data)["data"][0]
        data = pd.DataFrame(data)

        try:
            data.drop(["Suffix", "MiddleName", "Title"], axis=1, inplace=True)
            ord_enc = OrdinalEncoder()
            X = ord_enc.fit_transform(data)
            result = model.predict(pd.DataFrame(X)).tolist()
        except Exception as e:
            result = f"NOT PREDICTED, {e}"

        return json.dumps(result)

    except Exception as e:
        return json.dumps(str(e))