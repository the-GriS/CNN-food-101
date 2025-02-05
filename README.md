# Лабораторная работа №2

## 1. Обучить нейронную сеть EfficientNet-B0 (случайное начальное приближение) для решения задачи классификации изображений Food-101

### Графики обучения:
- Синяя линия - валидация
- Оранжевая линия - обучение

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/master/diagrams/lab_2_EfficientNet-B0/categorical_accuracy.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/master/diagrams/lab_2_EfficientNet-B0/loss.svg)

## 2. С использованием техники обучения Transfer Learning обучить нейронную сеть EfficientNet-B0 (предобученную на базе изображений imagenet) для решения задачи классификации изображений Food-101

### Графики обучения:
- Синяя линия - валидация
- Оранжевая линия - обучение

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/master/diagrams/lab_2_TransferLearning/epoch_categorical_accuracy.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/master/diagrams/lab_2_TransferLearning/epoch_loss.svg)

## Анализ результатов
При обучении нейроной сети с использованием техники обучения Transfer Learning точность получилась значительно выше(66,76% при валидации и 83,77 при обучении), чем при обучении со случайным начальным приближением (31.26% при валидации и 28.16% при обучении). Использование техники обучения Transfer Learning лучше на 35.5% при валидации и на 55.61% при обучении. Из этого мы делаем вывод, что обучение с использование  Transfer Learning эффективнее нежели метод со случайным начальным приближением. Также время затраченное на обучение Transfer Learning меньше, чем для аналогичной операции для метода случайного начального приближения: 1.2 часа против 6 часов соответственно.
