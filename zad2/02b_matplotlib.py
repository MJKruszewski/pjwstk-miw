import matplotlib.pyplot as plt
import math
import numpy as np

def main():
    #  Na razie tworzymy tylko wartości y
    y = [4, 5, 3, 7, 3]

    # Umieszczamy na wykresie wartości y, jeżeli nie podamy wartości
    # osi x zostaną one domyślnie uznane za 0,1,2,....
    plt.plot(y)
    # Wyświetlamy wykres, to polecenie zbiera wszystkie poprzednie polecenia
    # pyplota.
    # Od tej pory nowe polecenia będą tyczyć się kolejnego wykresu
    # Wykonanie programu zatrzyma się do chwili zamknięcia okna z wykresem
    plt.show()

    plt.plot(y)
    # Możemy dodać podpisy do osi
    plt.xlabel('Domyślne kroki osi x')
    plt.ylabel('Wartości na osi y')
    plt.show()

    # Tym razem ustawimy zarówno wartości na osi x jak i y
    x = list(range(1, 30))
    x = list(map(float, x))
    y = [math.sqrt(v) for v in x]
    plt.plot(x, y)
    plt.ylabel('Pierwiastki kwadratowe')
    plt.show()
    
    # Możliwe jest też umieszczenie kilku funkcji na jednym wykresie
    
    # Wejściem dla pyplota mogą być tablice numpy
    x = np.linspace(1, 20, 20)

    # Wydajemy trzy polecenia narysowania wykresu bez wywoływania
    # plt.show().
    plt.plot(x, x, label='liniowa')
    plt.plot(x, np.power(2,x), label='wykładnicza')
    plt.plot(x, np.log(x), label='logarytmiczna')
    
    plt.xlabel('Ilość danych')
    plt.ylabel('Ilość operacji')
    
    plt.title("Złożoność obliczeniowa")
    
    # Legenda wypisze znaczenie każdej linii na wykresie zgodnie z parametrem
    # label.
    plt.legend()
    
    # Musimy ustalić ograniczenie maksymalnej wartości na osi y, w innym przypadku
    # rozciągnie ją tak żeby zmieścić całą funkcję wykładniczą przez co dwie 
    # pozostałe będą wyglądały jak by były płaskie.
    plt.ylim(0, 30)
    
    plt.show()
    

if __name__ == '__main__':
    main()
