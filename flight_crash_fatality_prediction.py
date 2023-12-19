# script to create a new (normalized) observation data point for the plane crash dataset
# author tbartolo
import joblib
import numpy as np
import gradio as gr
import os

resources_dir = "resources"
normalizers_dir = "normalizers"
algorithms_dir = "algorithms"


def master_fn(year, operator, location, ac_type, passengers, crew, algorithm):
    def ac_type_normalizer(value):
        hasher = joblib.load(os.path.join(resources_dir, normalizers_dir, "ac_type_hash_model.joblib"))
        return hasher.transform([[value]]).toarray().flatten()

    def location_normalizer(value):
        hasher = joblib.load(os.path.join(resources_dir, normalizers_dir, "location_hash_model.joblib"))
        return hasher.transform([[value]]).toarray().flatten()

    def operator_normalizer(value):
        hasher = joblib.load(os.path.join(resources_dir, normalizers_dir, "operator_hash_model.joblib"))
        return hasher.transform([[value]]).toarray().flatten()

    def year_normalizer(value):
        kbins_discretizer = joblib.load(os.path.join(resources_dir, normalizers_dir, "year_qbins_model.joblib"))
        return kbins_discretizer.transform([[value]]).flatten()

    def passenger_crew_normalizer(value1, value2):
        scaler = joblib.load(os.path.join(resources_dir, normalizers_dir, "passenger_crew_scaler_model.joblib"))
        return scaler.transform([[value1, value2]])[0]

    observation = np.concatenate((year_normalizer(year), operator_normalizer(operator), location_normalizer(location),
                                  ac_type_normalizer(ac_type), passenger_crew_normalizer(passengers, crew)))

    result = float

    if algorithm == "SVR":
        svr = joblib.load(os.path.join(resources_dir, algorithms_dir, "svr_model.joblib"))
        result = svr.predict([observation])[0]
    elif algorithm == "KNN":
        knn = joblib.load(os.path.join(resources_dir, algorithms_dir, "knn_model.joblib"))
        result = knn.predict([observation])[0]
    elif algorithm == "Isolation Forest":
        random_forest = joblib.load(os.path.join(resources_dir, algorithms_dir, "random_forest_model.joblib"))
        result = random_forest.predict([observation])[0]
    else:
        raise ValueError('Unknown algorithm: {}'.format(algorithm))

    return round(result, 2)


demo = gr.Interface(
    master_fn,
    [
        "number",
        "text",
        "text",
        "text",
        "number",
        "number",
        gr.Radio(["KNN", "Isolation Forest", "SVR"])
    ],
    "number",
    title="Flight Crash Fatality Prediction",
    description="Enter the flight details and the algorithm to perform the prediction. Note that the purpose of this "
                "effort is strictly to showcase the strengths and weaknesses amongst the 3 techniques used, "
                "and by no means claims any sort of accuracy in its predictions.",
)

if __name__ == "__main__":
    demo.launch(show_api=False)
