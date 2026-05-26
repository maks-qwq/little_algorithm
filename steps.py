import heapq
import numpy as np


def matrix_reduction(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Dokonuje redukcji macierzy kosztów dla Algorytmu Little'a.
    Odejmuje minima z wierszy i kolumn, a sumę odjętych wartości traktuje jako dolne ograniczenie kosztu.
    """
    inf = float("inf")
    m = matrix.copy()
    reduction_cost = 0.0

    # 1. Redukcja wierszy
    for i in range(m.shape[0]):
        row_min = np.min(m[i])

        if row_min == inf:
            continue

        if row_min > 0:
            reduction_cost += row_min
            m[i] -= row_min

    # 2. Redukcja kolumn
    for j in range(m.shape[1]):
        col_min = np.min(m[:, j])

        if col_min == inf:
            continue

        if col_min > 0:
            reduction_cost += col_min
            m[:, j] -= col_min

    return m, reduction_cost


def min_without(values: np.ndarray, forbidden_index: int) -> float:
    """
    Szuka minimum w tablicy, ale pomija jeden wskazany indeks.

    Jest to potrzebne przy liczeniu kary dla zera:
    - patrzę na minimum w wierszu bez aktualnej kolumny,
    - patrzę na minimum w kolumnie bez aktualnego wiersza.
    """
    inf = float("inf")
    best = inf

    for index, value in enumerate(values):
        if index == forbidden_index:
            continue

        if value < best:
            best = value

    if best == inf:
        return 0.0

    return best


def section_ij(matrix: np.ndarray) -> tuple[int | None, int | None, float]:
    """
    Wybiera zero o największej karze.

    Kara mówi, ile minimalnie stracę, jeżeli zabronię przejścia i -> j.
    W algorytmie Little'a wybieram zero z największą karą, bo ono najlepiej dzieli problem.
    """
    best_i = None
    best_j = None
    best_penalty = -1.0

    n = matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if np.isclose(matrix[i, j], 0.0):
                row_min = min_without(matrix[i, :], j)
                col_min = min_without(matrix[:, j], i)

                penalty = row_min + col_min

                if penalty > best_penalty:
                    best_i = i
                    best_j = j
                    best_penalty = penalty

    return best_i, best_j, best_penalty


def is_full_tour(path: list[tuple[int, int]], n: int) -> bool:
    """
    Sprawdza, czy wybrane krawędzie tworzą pełny cykl Hamiltona.

    Czyli dla n miast musi być dokładnie n krawędzi i każde miasto ma mieć:
    - dokładnie jedno wyjście,
    - dokładnie jedno wejście.
    """
    if len(path) != n:
        return False

    next_city = dict(path)

    if len(next_city) != n:
        return False

    if len(set(next_city.values())) != n:
        return False

    start = path[0][0]
    current = start

    for _ in range(n):
        if current not in next_city:
            return False

        current = next_city[current]

    return current == start


def creates_forbidden_subtour(path: list[tuple[int, int]], n: int) -> bool:
    """
    Sprawdza, czy dodanie krawędzi nie zrobiło małego cyklu.

    Mały cykl jest zły, bo w TSP potrzebuję jednego dużego cyklu przez wszystkie miasta.
    Przykład złej sytuacji: 1 -> 3 -> 1, gdy zostały jeszcze inne miasta.
    """
    next_city = dict(path)

    for start in next_city:
        current = start
        visited = set()

        while current in next_city:
            if current in visited:
                return not is_full_tour(path, n)

            visited.add(current)
            current = next_city[current]

    return False


def path_cost(original_matrix: np.ndarray, path: list[tuple[int, int]]) -> float:
    """
    Liczy prawdziwy koszt ścieżki na oryginalnej macierzy kosztów.
    """
    cost = 0.0

    for i, j in path:
        cost += original_matrix[i, j]

    return cost


def order_path_edges(path: list[tuple[int, int]], n: int, start: int = 0) -> list[tuple[int, int]]:
    """
    Układa krawędzie w normalnej kolejności trasy.

    Dzięki temu wynik będzie wyglądał elegancko, np.:
    1 -> 5 -> 3 -> 6 -> 4 -> 2 -> 1
    """
    next_city = dict(path)

    if start not in next_city:
        start = path[0][0]

    ordered_path = []
    current = start

    for _ in range(n):
        next_node = next_city[current]
        ordered_path.append((current, next_node))
        current = next_node

    return ordered_path


def split_problem(
    matrix: np.ndarray,
    lower_bound: float,
    path: list[tuple[int, int]],
    i: int,
    j: int
) -> list[tuple[float, np.ndarray, list[tuple[int, int]]]]:
    """
    Dzieli problem na dwa podproblemy:

    1. Biorę krawędź i -> j.
    2. Nie biorę krawędzi i -> j.

    Każdy podproblem dostaje nową macierz i nowe dolne ograniczenie.
    """
    inf = float("inf")
    n = matrix.shape[0]

    children = []

    # Przypadek 1: biorę krawędź i -> j

    included_path = path + [(i, j)]

    if not creates_forbidden_subtour(included_path, n):
        included_matrix = matrix.copy()

        # Skoro biorę i -> j, to:
        # - z miasta i nie mogę już wyjechać nigdzie indziej,
        # - do miasta j nie mogę już wjechać znikąd indziej.
        included_matrix[i, :] = inf
        included_matrix[:, j] = inf

        # Blokuję też natychmiastowy powrót j -> i.
        included_matrix[j, i] = inf

        included_matrix, reduction_cost = matrix_reduction(included_matrix)
        included_bound = lower_bound + reduction_cost

        children.append((included_bound, included_matrix, included_path))

    # Przypadek 2: nie biorę krawędzi i -> j


    excluded_matrix = matrix.copy()
    excluded_matrix[i, j] = inf

    excluded_matrix, reduction_cost = matrix_reduction(excluded_matrix)
    excluded_bound = lower_bound + reduction_cost

    children.append((excluded_bound, excluded_matrix, path.copy()))

    return children


def little_algorithm(cost_matrix: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """
    Główna funkcja Algorytmu Little'a dla problemu komiwojażera.

    Zwraca:
    - najlepszy koszt,
    - najlepszą trasę jako listę krawędzi.
    """
    original_matrix = np.asarray(cost_matrix, dtype=float)
    n = original_matrix.shape[0]

    start_matrix, start_bound = matrix_reduction(original_matrix)

    # Kolejka priorytetowa.
    # Zawsze najpierw rozwijam problem z najmniejszym dolnym ograniczeniem.
    queue = []
    counter = 0

    heapq.heappush(queue, (start_bound, counter, start_matrix, []))

    best_cost = float("inf")
    best_path = []

    while queue:
        lower_bound, _, matrix, path = heapq.heappop(queue)

        # Jeżeli dolne ograniczenie jest już gorsze niż najlepszy znany wynik,
        # to nie ma sensu dalej rozwijać tej gałęzi.
        if lower_bound >= best_cost:
            continue

        # Jeżeli mam już n krawędzi, to sprawdzam, czy powstała pełna trasa.
        if len(path) == n:
            if is_full_tour(path, n):
                current_cost = path_cost(original_matrix, path)

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = path

            continue

        # Wybieram zero z największą karą.
        i, j, _ = section_ij(matrix)

        if i is None or j is None:
            continue

        # Tworzę dwa podproblemy i wrzucam je do kolejki.
        children = split_problem(matrix, lower_bound, path, i, j)

        for child_bound, child_matrix, child_path in children:
            if child_bound < best_cost:
                counter += 1
                heapq.heappush(queue, (child_bound, counter, child_matrix, child_path))

    # Układam wynik od miasta 1, żeby wypisywanie było czytelniejsze.
    best_path = order_path_edges(best_path, n, start=0)

    #  indeksy z od 0 na od 1
    best_path_one_based = []

    for i, j in best_path:
        best_path_one_based.append((i + 1, j + 1))

    return best_cost, best_path_one_based
