# Решение задания на хакатон от Science Data Lab

## Информация о команде
Название - **Лужескоки**

Участники студенты ДВГУПС, группа БО931ПИА:
- Морозов Даниил (https://github.com/AdirtKa)
- Стороженко Егор (https://github.com/Fayylen)
- Ковшар Давид (https://github.com/LiamHater)

## Краткое описание проделанной работы

1. Был реализован класс, заполняющие пропуски средним значением по столбцу
2. Был реализован класс, расчитывающий различные вегетационные признаки с помощью каналов
3. Выбранная модель HistGradientBoostingClassifier
4. Итоговая f1-метрика на тренировочной выборке **0.989**

## Как загружать данные в модель
инструкция указана в файле load_data.md

## Примечание

Так как библиотека *pickle* не умеет сериализировать обычные классы в *pickle*-файлы
для работы модели, открытой через эту библиотеку, необходимо иметь реализацию 
написанных классов DataImputer и VegetationIndexAdder в файле, где модель используется.

Другим же вариантом является использования библиотеки *dill*. В данном случае ее функционал
никак не отличается от *pickle* за исключением сериализации самописных классов.