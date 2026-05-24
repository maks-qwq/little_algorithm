import numpy as np

def matrix_reduction(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Dokonuje redukcji macierzy kosztów dla Algorytmu Little'a (TSP).
    Odejmuje minima z wierszy oraz kolumn i sumuje je, wyznaczając 
    dolne ograniczenie kosztu.

    Parametry:
    ----------
    matrix : np.ndarray
        Kwadratowa macierz kosztów (np. warstwy odległości między miastami).

    Zwraca:
    -------
    tuple[np.ndarray, float]
        Zredukowana macierz (nowa kopia) oraz całkowity koszt dokonanej redukcji.
    """
    inf = float('inf')
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

def section_ij():
    # TODO: Wyznaczenie odcinka <i\*j\*> o max optymistycznym koszcie wyłączenia (spośród wszystkich aij=O)
    pass


def split_problem():
    # TODO: Podział problemu na dwa podproblemy: z aij=1 i z aij=0
    # Wyznaczenie odcinka <i\*j\*> o max optymistycznym koszcie wyłączenia (spośród wszystkich aij=O)
    pass

def little_algorithm():
    # TODO: krok 4 i 5; gdzie jest też powrt do kroku 1
    pass

