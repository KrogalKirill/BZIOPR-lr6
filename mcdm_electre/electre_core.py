"""electre_core.py

Модуль математических расчетов многокритериального анализа:
- Определение множества Парето-оптимальных альтернатив
- Расчет весов критериев методом ранга
- Нормирование данных и расчет матриц согласия/несогласия (ЭЛЕКТРА I)
- Построение ядра недоминируемых альтернатив
"""

from typing import List, Tuple
import numpy as np


def find_pareto_set(data: np.ndarray, is_max: List[bool]) -> np.ndarray:
    """Определяет маску недоминируемых по Парето альтернатив.

    :param data: Двумерный массив оценок (строки - альтернативы, столбцы - критерии)
    :param is_max: Список булевых флагов максимизации по каждому критерию
    :return: Булевый вектор-маска длины N, где True - Парето-оптимальная альтернатива
    """
    n, m = data.shape
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue

            better_or_eq = True
            strictly_better = False

            for k in range(m):
                vi = data[i, k]
                vj = data[j, k]
                if is_max[k]:
                    if vj < vi:
                        better_or_eq = False
                    if vj > vi:
                        strictly_better = True
                else:
                    if vj > vi:
                        better_or_eq = False
                    if vj < vi:
                        strictly_better = True

            # j доминирует над i, если j не хуже i по всем критериям и строго лучше хотя бы по одному
            if better_or_eq and strictly_better:
                dominated[i] = True
                break

    return ~dominated


def calc_weights_by_rank(ranks: np.ndarray) -> np.ndarray:
    """Вычисляет веса критериев методом ранга.

    :param ranks: Матрица оценок экспертов (строки - эксперты, столбцы - критерии)
    :return: Вектор нормализованных весов критериев
    """
    total_per_criterion = ranks.sum(axis=0)
    grand_total = total_per_criterion.sum()
    if grand_total == 0:
        return np.ones(ranks.shape[1]) / ranks.shape[1]
    return total_per_criterion / grand_total


def normalize_matrix(data: np.ndarray, is_max: List[bool]) -> np.ndarray:
    """Приводит матрицу к безразмерному виду в диапазоне [0; 1].

    Для максимизации: c_norm = c_i / max(c)
    Для минимизации: c_norm = min(c) / c_i
    """
    norm = np.zeros_like(data, dtype=float)
    n, m = data.shape

    for k in range(m):
        col = data[:, k]
        if is_max[k]:
            col_max = col.max()
            norm[:, k] = col / col_max if col_max != 0 else col
        else:
            col_min = col.min()
            # Предотвращение деления на ноль
            with np.errstate(divide='ignore'):
                norm[:, k] = np.where(col != 0, col_min / col, 1.0)
    return norm


def run_electre(
    data: np.ndarray,
    names: List[str],
    weights: np.ndarray,
    is_max: List[bool],
    c_star: float,
    d_star: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Выполняет расчет методом ЭЛЕКТРА I.

    :param data: Матрица оценок отобраных Парето-альтернатив
    :param names: Имена Парето-альтернатив
    :param weights: Вектор весов критериев
    :param is_max: Направление оптимизации критериев
    :param c_star: Порог согласия c*
    :param d_star: Порог несогласия d*
    :return: (матрица согласия, матрица несогласия, список альтернатив ядра)
    """
    n, m = data.shape
    norm = normalize_matrix(data, is_max)

    c_mat = np.zeros((n, n))
    d_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            concordance = 0.0
            max_discord = 0.0

            for k in range(m):
                # Если i не хуже j по критерию k
                if norm[i, k] >= norm[j, k]:
                    concordance += weights[k]
                else:
                    gap = norm[j, k] - norm[i, k]
                    if gap > max_discord:
                        max_discord = gap

            c_mat[i, j] = round(concordance, 4)
            d_mat[i, j] = round(max_discord, 4)

    # Определение отношений доминирования (i опережает j, если c_ij >= c* и d_ij <= d*)
    is_beaten = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and c_mat[i, j] >= c_star and d_mat[i, j] <= d_star:
                is_beaten[j] = True

    kernel = [names[i] for i in range(n) if not is_beaten[i]]
    return c_mat, d_mat, kernel