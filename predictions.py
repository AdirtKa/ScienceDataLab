import pandas as pd
import dill
import numpy as np
import csv


def main(*args, **kwargs) -> int():
    with open("models/dill_model.pkl", "rb") as f:
        info = dill.load(f)

    model = info["model"]
    save_open(model)
    save_closed(model)

    return 0

def save_open(model):
    df_ndvi = pd.read_csv("data/test/test_public/NDVI.csv", sep=";", encoding="windows-1251").drop(columns=["index"])
    df_nir = pd.read_csv("data/test/test_public/B8A.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_nir")
    df_swir = pd.read_csv("data/test/test_public/B12.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_swir")
    df_red = pd.read_csv("data/test/test_public/B04.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_red")
    df_VegRedEdge = pd.read_csv("data/test/test_public/B05.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_vegRedEdge")
    df_blue = pd.read_csv("data/test/test_public/B02.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_blue")
    df_green = pd.read_csv("data/test/test_public/B03.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_green")

    data = pd.concat([df_ndvi, df_nir, df_swir, df_red, df_VegRedEdge, df_blue, df_green], axis=1)

    predictions: list[str] = model.predict(data).tolist()

    with open("answers/classification_openset.csv", mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["culture"])
        for item in predictions:
            writer.writerow([item])

def save_closed(model):
    close_df_ndvi = pd.read_csv("data/test/test_closed/NDVI.csv", sep=";", encoding="windows-1251").drop(columns=["index"])
    close_df_nir = pd.read_csv("data/test/test_closed/B8A.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_nir")
    close_df_swir = pd.read_csv("data/test/test_closed/B12.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_swir")
    close_df_red = pd.read_csv("data/test/test_closed/B04.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_red")
    close_df_VegRedEdge = pd.read_csv("data/test/test_closed/B05.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_vegRedEdge")
    close_df_blue = pd.read_csv("data/test/test_closed/B02.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_blue")
    close_df_green = pd.read_csv("data/test/test_closed/B03.csv", sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_green")

    close_data = pd.concat([close_df_ndvi, close_df_nir, close_df_swir, close_df_red, close_df_VegRedEdge, close_df_blue, close_df_green], axis=1)

    predictions: list[str] = model.predict(close_data).tolist()

    with open("answers/classification_closedset.csv", mode="w", newline="", encoding="windows-1251") as file:
        writer = csv.writer(file)

        writer.writerow(["culture"])
        for item in predictions:
            writer.writerow([item])



if __name__ == '__main__':
    main()