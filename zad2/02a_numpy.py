import numpy as np
import matplotlib.pylab as plt

def main():
    # W pythonie nie istnieją tablice, tworzyć można tylko listy
    l = [32,54,22,76,2]
    print(type(l))
    
    # W numpy można tworzyć tablice na podstawie list zawierających liczby
    a = np.array(l)
    print(type(a))
    
    # Należy zauważyć że numpy przyjmie tablicę zawierającą inne wartości
    # niż liczbowe i wykrzaczy się dopiero kiedy spróbujemy wykonać na niej
    # operacje arytmetyczne.
    try:
        b = np.array(['ala ma kota'])
        print('{} {}'.format(b, type(b)))
    except Exception as e:
        print(e)
        
    # Możemy sprawdzić typ oraz wymiary tablicy
    print('Typ a = {}'.format(a.dtype.name))
    print('Wymiary a = {}'.format(a.shape))
    print('Typ b = {}'.format(b.dtype.name))
    print('Wymiary b = {}'.format(b.shape))

    # Jak widać tablica B ma typ string, dlatego jeżeli nie chcemy
    # żeby do naszego kodu przedarły sie jakieś stringi najlepiej
    # jawnie podawać typ tablicy podczas jej tworzenia.
    c = np.array(l, dtype=np.float64)
    
    # Należy zwrócić uwagę na to że jeżeli jest to możliwe wartości są
    # konwertowane do wybranego typu podczas gdy bez jawnego jego podania
    # typ jest ustawiany na zgodny z wrarością pierwszego elementu listy.
    print('Typ c = {}'.format(c.dtype.name))
    print('Wymiary c = {}'.format(c.shape))

    # Jeżeli spróbujemy przekazać jakieś dane które nie mogą zostać przekonwertowane
    # zostanie podniesiony wyjątek w chwili tworzena tablicy.
    try:
        d = np.array(['ala ma kota'], dtype=np.float64)
        print('{} {}'.format(d, type(d)))
    except Exception as e:
        print(e)
        
    # Numpy służy przedewszystkim do pracy z macierzami jak np
    d = np.array([[1.5,2,3], [4,5,6]], np.float64)
    print('Typ d = {}'.format(d.dtype.name))
    print('Wymiary d = {}'.format(d.shape))

    # To polecenie stworzy wektor 6-cio elementowy z kolejnych liczb naturalnych
    # poczynając od zera.
    e = np.arange(6)
    print(e)
    
    # Następnie możemy go przekształcić na macierz 
    e = e.reshape(3, 2)
    print(e)
    f = e.reshape(2, 3)
    print(f)
    
    # Numpy zapewnia następujące operatory arytmetyczne
    print(d + f)  # Dodawanie i mnożenie
    print(d * f)  # Mnożenie element przez element
    print(e @ d)  # Mnożenie macierzy

    # Kilka specjalnych metod do generowania macierzy
    np.zeros((2, 3))  # Macierz złożona z samych zer
    np.linspace(1., 4., 6)  # Sześć liczb rzeczywistych równo rozłożonych między 1 a 4
    np.identity(3)  # Macierz jednostkowa 3x3

    # Możliwa jest też transpozycja macierzy
    g = np.transpose(f)
    g = f.T
    print('Macierz transponowana g= ')
    print(g)

    # W numpy można też stworzyć macierz odwrotną korzystając z bibliotek algebry
    # liniowej

    h = np.linalg.inv(f@np.transpose(f))
    print('Typ h = {}'.format(h.dtype.name))
    print('Wymiary h = {}'.format(h.shape))





if __name__ == '__main__':
    main()
