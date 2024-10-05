```python
import os
your_path = "ваш/путь/к/данным"
```
Прописываем путь где хранятся ваши данные в системе.
```python
df_ndvi = pd.read_csv(os.path.join(your_path,"NDVI.csv"), sep=";", encoding="windows-1251").drop(columns=["index"])
df_nir = pd.read_csv(os.path.join(your_path,"B8A.csv"), sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_nir")
df_swir = pd.read_csv(os.path.join(your_path,"B12.csv"), sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_swir")
df_red = pd.read_csv(os.path.join(your_path,"B04.csv"), sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_red")
df_VegRedEdge = pd.read_csv(os.path.join(your_path,"B05.csv"), sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_vegRedEdge")
df_blue = pd.read_csv(os.path.join(your_path,"B02.csv"), sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_blue")
df_green = pd.read_csv(os.path.join(your_path,"B03.csv"), sep=";", encoding="windows-1251").drop(columns=["index"]).add_suffix("_green")


data = pd.concat([df_ndvi, df_nir, df_swir, df_red, df_VegRedEdge, df_blue, df_green], axis=1)
```
После общей загрузки и обработки данных можем на их основе обучать модель. Удаляем индексы, так как pandas индексирует данные автоматически (нам изначальные индексы не нужны). Добавляем суффиксы для того чтобы иметь возможность поимённо обращаться к каждом каналу.
```python

predictions: list[str] = model.predict(data).tolist()
with open("answers/classification_openset.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["culture"])
    for item in predictions:
        writer.writerow([item])
```
Вызываем метод predict для того чтобы модель делала предсказания. Все предсказанные хозяйственные культуры экспортируются в файл .csv в "answers/classification_openset.csv".
