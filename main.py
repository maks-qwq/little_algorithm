import steps 
import sys
import numpy as np

if __name__ == "__main__":
    inf = float('inf')
    
    # Definicja macierzy kosztów o rozmiarze 6x6 dla 6 miast (P/PP)
    # Wartości 'inf' na przekątnej blokują ruch z miasta do samego siebie
    matrix_6x6 = np.array([
        [inf,  25,  40,  31,  27,  15],
        [ 5, inf,  17,  30,  12,  20],
        [19,  22, inf,  11,  14,   8],
        [42,  13,  24, inf,  19,  33],
        [18,  10,   6,  25, inf,  21],
        [31,   7,  14,   9,  28, inf]
    ])

    print("=" * 60)
    print("URUCHOMIENIE ALGORYTMU LITTLE'A Z PODZIAŁEM NA PODPROBLEMY (P/PP)")
    print(f"Rozmiar instancji problemu: {matrix_6x6.shape[0]} x {matrix_6x6.shape[1]}")
    print("=" * 60)
    
    # Wywołanie algorytmu Little'a (wartość odcinająca v* oraz ostateczna ścieżka)
    #Wrzućcie niżej odpowiednią funkcję
    v_star, best_path = ###(matrix_6x6)
    
    # Prezentacja końcowych wyników
    print("\n" + "=" * 60)
    print("WYNIK KOŃCOWY:")
    print(f" -> Optymalna wartość odcinająca (v*): {v_star}")
    print(f" -> Wybrane odcinki rozwiązania TSP:   {best_path}")
    
    # Próba ułożenia krawędzi w logiczną, chronologiczną trasę
    try:
        path_dict = dict(best_path)
        start_node = best_path[0][0]
        ordered_path = [start_node]
        current = start_node
        
        for _ in range(len(best_path)):
            current = path_dict[current]
            ordered_path.append(current)
            
        print(f" -> Pełna trasa komiwojażera:          {' -> '.join(map(str, ordered_path))}")
    except KeyError:
        print(" -> Informacja: Zbiór krawędzi nie tworzy jednego zamkniętego cyklu.")
        
    print("=" * 60)