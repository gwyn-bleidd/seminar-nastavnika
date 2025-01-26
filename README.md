# Алгоритм решения задачи маршрутизации транспортных средств (CVRP)

Алгоритм решает задачу маршрутизации транспортных средств (CVRP) с использованием комбинации следующих методов:

## Инициализация решения

Используется либо метод ближайшего соседа (`nearest_neighbor_init`), либо метод Кларка-Райта (`clarke_wright_init`) для создания начального решения. Эти методы обеспечивают разумное начальное распределение маршрутов с учетом ограничений по вместимости транспортных средств.

## Детерминированный отжиг

Основной процесс оптимизации основан на методе отжига, где температура постепенно снижается. На каждой итерации алгоритм с определенной вероятностью принимает ухудшающие решения, что позволяет избежать застревания в локальных минимумах.

В процессе отжига применяется **стохастическая фаза**, где случайным образом выбираются операторы для изменения решения:

- **Swap**: Обмен узлами между маршрутами.
- **Relocate**: Перемещение узла из одного маршрута в другой.
- **Swap in Route**: Обмен узлами внутри одного маршрута.
- **Cross Exchange**: Обмен последовательностями узлов между маршрутами.

## Локальный поиск

На определенных этапах (например, при низкой температуре или через фиксированное количество итераций) применяется локальный поиск с использованием алгоритма Лин-Кернигана (`lin_kernighan`). Этот метод включает в себя:

- **2-opt**: Оптимизация маршрута путем перестановки двух ребер.
- **3-opt**: Оптимизация маршрута путем перестановки трех ребер.
- **Or-opt**: Оптимизация маршрута путем перемещения последовательности узлов.

## Валидация решения

После каждой итерации проверяется корректность решения с учетом ограничений по вместимости транспортных средств и посещения всех узлов.

## Сравнение с оптимальным решением

Алгоритм сравнивает найденное решение с известным оптимальным (если оно предоставлено) и вычисляет:
- Процент отклонения от оптимума.
- Процент улучшения по сравнению с начальным решением.
