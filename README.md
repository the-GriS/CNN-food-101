# Лабораторная работа №3

## 1. С использованием техники обучения Transfer Learning обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101 с использованием фиксированных темпов обучения 0.01, 0.001, 0.0001

### Графики обучения:
- Красный - темп 0.01 на обучении / Голубой - темп 0.01 на валидации
- Оранжевый - темп 0.001 на обучении / Синий - темп 0.001 на валидации
- Розовый - темп 0.0001 на обучении / Зеленый - темп 0.0001 на валидации

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_categorical_accuracy.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_loss.svg)

## 2. Реализовать и применить в обучении Косинусное затухание (Cosine Decay), а также определить оптимальные параметры для данной политики

При использовании политики косинусного затухания мы взяли функцию:  
```
learning_rate_cos = tf.keras.experimental.CosineDecay(
    initial_learning_rate, decay_steps) 
```  
Где initial_learning_rate - начальная скорость обучения, decay_steps - количество итераций, по которым проходит затухание.

### Графики обучения:
- Оранжевый - темп 0.001 decay_steps = 7700 на валидации
- Красный - темп 0.01 decay_steps = 7700 на валидации
- Синий - темп 0.001 decay_steps = 66600 на валидации
- Голубой - темп 0.0001 decay_steps = 7700 на валидации
- Розовый - темп 0.0001 decay_steps = 22200 на валидации

*График точности на валидации*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_categorical_accuracy_cos.svg)

*График потерь на валидации*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/epoch_loss_cos.svg)

*График изменения темпа обучения при initial_learning_rate = 0.001*
- Оранжевый - decay_steps = 7700  
- Cиний - decay_steps = 66600 
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/learning_rate_cos_0.001.svg)

*График изменения темпа обучения при initial_learning_rate = 0.01 decay_steps = 7700*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/learning_rate_0.01.svg)

*График изменения темпа обучения при initial_learning_rate = 0.0001*
- Оранжевый - decay_steps = 7700  
- Cиний - decay_steps = 22200 
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/learning_rate_cos_0.0001.svg)

## 3. Реализовать и применить в обучении Косинусное затухание с перезапусками (Cosine Decay with Restarts), а также определить оптимальные параметры для данной политики

При использовании политики косинусного затухания c перезапусками мы взяли функцию:  
```
learning_rate_cos_res = (
  tf.keras.experimental.CosineDecayRestarts(
      initial_learning_rate,
      first_decay_steps, t_mul, m_mul))
```  
Где initial_learning_rate - начальная скорость обучения, first_decay_steps - количество итераций, по которым проходит затухание и после чего проиходит перезапуск; t_mul - используется для определения количества итераций в i-м периоде(first_decay_steps_i = first_decay_steps * t_mul), m_mul - используется для получения начальной скорости обучения i-го периода(initial_learning_rate_i = initial_learning_rate * m_mul).

### Графики обучения:
- Оранжевый - темп 0.001 при first_decay_steps=7700 t_mul=1.7 m_mul=0.7 на обучении 
- Синий - темп 0.001 при first_decay_steps=7700 t_mul=1.7 m_mul=0.7 на валидации
- Красный - темп 0.001 при first_decay_steps=7700 t_mul=1.3 m_mul=0.9 на обучении 
- Голубой - темп 0.001 при first_decay_steps=7700 t_mul=1.3 m_mul=0.9 на валидации

*График точности*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/categorical_accuracy_cos_dec.svg)

*График потерь*
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/loss_cos_res.svg)

*График изменения темпа обучения*
- Оранжевый - темп 0.001 при first_decay_steps=7700 t_mul=1.7 m_mul=0.7 
- Красный - темп 0.001 при first_decay_steps=7700 t_mul=1.3 m_mul=0.9 
![Alt-текст](https://github.com/the-GriS/CNN-food-101/blob/lab_3/diagrams/lab_3/learning_rate_cos_res.svg)

## Анализ результатов
Если судить по выборке обучения, то лучшим фиксированным темпом обучения является 0.001, т.к точность на обучении достигает 83.77 %, при потерях 0.5468. При этом стоит заметить, что при валидации лучшее значение получается при фиксированном темпе обучения 0.0001, которое равно 67.16% при потерях 1.222. Таким образом можно сделать вывод, что оптимальным темпом обучения является 0.0001.

При использовании политики косинусного затухания initial_learning_rate подбирался(0.01, 0.001, 0.0001), decay_steps - (7700, 66600). Судя по графикам при decay_steps = 7700 и при decay_steps = 22200 наше обучение в определенный момент прерывается. Это связанно с тем, что значение decay_steps меньше количества всех наших итераций и при достижении конца этих итераций learning_rate принимает значение 0, что можно проследить и в графике изменения темпа обучения. Оптимальными параметрами являются initial_learning_rate = 0,001 decay_steps = 7700 ,т. к. при таких значениях получается лучшее значение на валидации равное 66.54% при потерях 1.25.  

При использовании политики косинусного затухания c перезапусками мы приняли initial_learning_rate = 0.001, и в этом случае прослеживалась следующая тенденция: на валидации лучшее значение равное 67.66%  при потерях 1.212 было полученно на 18 эпохе когда first_decay_steps=7700 t_mul=1.7 m_mul=0.7(синий график), а на обучении лучшее значение равное 82.33% при потерях 0.6451 было полученно на 41 эпохе когда first_decay_steps=7700 t_mul=1.3 m_mul=0.9(красный график).
