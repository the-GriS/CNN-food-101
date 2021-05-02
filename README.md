# Лабораторная работа №4

В работе изспользуются графики для валидации.  
Базовый алгоритм взят из прошлой лабораторной работы - политика косинусного затухания:
```
learning_rate_cos_res = (
  tf.keras.experimental.CosineDecayRestarts(
      initial_learning_rate,
      first_decay_steps, t_mul, m_mul))
```
Где initial_learning_rate = 0.001 first_decay_steps=7700 t_mul=1.7 m_mul=0.7.


## 1. С использованием, техники обучения Transfer Learning и оптимальной политики изменения темпа обучения, определенной в ходе выполнения лабораторной #3, обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101 с использованием следующих техник аугментации данных, так же определить оптимальный набор параметров для этих техник аугментации:  

### a. Случайное горизонтальное и вертикальное отражение 

При использовании техники аугментации данных "Случайное горизонтальное и вертикальное отражение" мы взяли команду:  
```
tf.keras.layers.experimental.preprocessing.RandomFlip(
    mode=HORIZONTAL_AND_VERTICAL)
```

### Графики для валидации:
- Оранжевый - базовый алгоритм
- Синий - mode = "horizontal_and_vertical" 
- Красный - mode = "horizontal"
- Голубой - mode = "vertical"

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/categorical_accuracy_rand_flip.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/loss_rand_flip.svg)

*Пример измененного изображения с оптимальными праметрами*  
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/img_horizont.jpg)

### b. Использование случайной части изображения 

При использовании техники аугментации данных "Использование случайной части изображения" мы взяли команду:  
```
tf.keras.layers.experimental.preprocessing.RandomCrop(
    height = 224, width = 224)
```

### Графики для валидации:
- Оранжевый - базовый алгоритм
- Синий - начальный размер изображения 400*400 
- Красный - начальный размер изображения 300*300 
- Голубой - начальный размер изображения 250*250 
- Розовый - начальный размер изображения 350*350 

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/categorical_accuracy_rand_crop.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/loss_rand_crop.svg)

*Пример измененного изображения с оптимальными праметрами*  
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/img_crop.jpg)

### c. Поворот на случайный угол

При использовании техники аугментации данных "Поворот на случайный угол" мы взяли команду:  
```
tf.keras.layers.experimental.preprocessing.RandomRotation(
    factor)
```  
Где factor - это множитель угла случайного поворота(угол случайного поворота = 2Pi * factor) 

### Графики для валидации:
- Оранжевый - базовый алгоритм
- Синий - factor = 0.5, что дает угол поворота от -180 до 180 градусов
- Красный - factor = 0.25, что дает угол поворота от -90 до 90 градусов
- Голубой - factor = 0.15, что дает угол поворота от -54 до 54 градусов
- Розовый - factor = 0.05, что дает угол поворота от -18 до 18 градусов

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/categorical_accuracy_rand_rot.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/loss_rand_rot.svg)

*Пример измененного изображения с оптимальными праметрами*  
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/img.jpg)

## 2. Обучить нейронную сеть с использованием исследованных техник аугментации данных совместно

### Графики на валидации:
- Оранжевый - базовый алгоритм
- Синий - при использовании всех изученных техник аугиентации с оптимальными параметрами

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/categorical_accuracy_full.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/loss_full.svg)

*Пример измененного изображения с оптимальными праметрами*  
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_4/diagrams/lab_4/img_full.jpg)

## Анализ результатов
Сперва всмоним результаты на валидации для базового алгоритма - точность равная 67.66% при потерях 1.212, этот резуоьтат был получен на 14 эпохе.  

При использовании техники аугментации данных "Случайное горизонтальное и вертикальное отражение" mode подбирался("horizontal_and_vertical", "horizontal", "vertical"). Судя по графикам оптимальным параметром является mode - "horizontal", т. к. при таких значениях получается лучшее значение на валидации равное 66.65% при потерях 1.224, достигнутое на 30 эпохе.  

При использовании техники аугментации данных "Использование случайной части изображения" начальный размер изображений подбирался(400*400, 350*350, 300*300, 250*250). Судя по графикам оптимальным начальным размером является 250*250, т. к. на 18 эпохе мы достигли результатов равных 68.37% на графики точности при валидации с потерями равными 1.181. 

При использовании техники аугментации данных "Поворот на случайный угол" factor подбирался(0.5, 0.25, 0.15, 0.05). И для данной техники оптимальным параметром является factor = 0.05, что соответсвует углу поворота изображения в промежутке от -18 до 18 градусов. При этом мы достигли максимального значения равного 67.31% при потерях 1.226 на 16 эпохе.   

При использовании всех техник аугментации с оптимальными параметрами одновременно мы получили следующий результат: на 28 эпохе график точность при валидации показал следующее значение 68.14% при потерях равных 1.201, что является максимыльным значением для данного эксперимента.  

И вконце проведем сравнение полученных результатов с базовым алгоритмом, полученном в предыдущей лабораторной работе:
- Техника аугментации данных "Случайное горизонтальное и вертикальное отражение" ухудшила результат на 0.01%
- Техника аугментации данных "Использование случайной части изображения" улучшила результат на 0.71%
- Техника аугментации данных "Поворот на случайный угол" ухудшила результат на 0.35%
- Свосестное использование данных техник аугментации привело к положительному изменению результата, разница составила 0.48%
