# Лабораторная работа №2

## 1. Обучить нейронную сеть EfficientNet-B0 (случайное начальное приближение) для решения задачи классификации изображений Food-101

### Графики обучения:
- Синяя линия - валидация
- Оранжевая линия - обучение

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/master/diagrams/lab_2_TransferLearning/epoch_categorical_accuracy.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/master/diagrams/lab_2_TransferLearning/epoch_loss.svg)

## 2. С использованием техники обучения Transfer Learning обучить нейронную сеть EfficientNet-B0 (предобученную на базе изображений imagenet) для решения задачи классификации изображений Food-101

### Графики обучения:
- Синяя линия - валидация
- Оранжевая линия - обучение

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/master/diagrams/lab_2_TransferLearning/epoch_categorical_accuracy.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/master/diagrams/lab_2_TransferLearning/epoch_loss.svg)

## Анализ результатов
При обучении нейроной сети с использованием техники обучения Transfer Learning точность получилась значительно ниже, чем при обучении со случайным начальным приближением(на графики точности разница достигает 60% при обучении и 56% при валидации). Из этого мы делаем вывод, что обучение с использование  Transfer Learning эффективнее нежели метод со случайным начальным приближением. Также время затраченное на обучение Transfer Learning меньше, чем для анологичной операции для метода случайного начального приближения: 1.5 часа против 5 часов соответственно.
