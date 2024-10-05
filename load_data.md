```python
import os
your_path = "ваш_путь_к_данным"
```
Прописываем путь где хранятся ваши данные в системе
```python
df_ndvi = pd.read_csv(os.path.join(your_path,"NDVI.csv"), encoding="windows-1251").drop(columns=["index"])
df_nir = pd.read_csv(os.path.join(your_path,"B8A.csv"), encoding="windows-1251").drop(columns=["index", "culture"]).add_suffix("_nir")
df_swir = pd.read_csv(os.path.join(your_path,"B12.csv"), encoding="windows-1251").drop(columns=["index", "culture"]).add_suffix("_swir")
df_red = pd.read_csv(os.path.join(your_path,"B04.csv"), encoding="windows-1251").drop(columns=["index", "culture"]).add_suffix("_red")
df_VegRedEdge = pd.read_csv(os.path.join(your_path,"B05.csv"), encoding="windows-1251").drop(columns=["index", "culture"]).add_suffix("_vegRedEdge")
df_blue = pd.read_csv(os.path.join(your_path,"B02.csv"), encoding="windows-1251").drop(columns=["index", "culture"]).add_suffix("_blue")
df_green = pd.read_csv(os.path.join(your_path,"B03.csv"), encoding="windows-1251").drop(columns=["index", "culture"]).add_suffix("_green")

labels = df_ndvi["culture"]
df_ndvi.drop(columns=["culture"], inplace=True)

data = pd.concat([df_ndvi, df_nir, df_swir, df_red, df_VegRedEdge, df_blue, df_green], axis=1)
```
После общей загрузки и обработки данных можем на их основе обучать модель