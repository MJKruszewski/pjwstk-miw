
# Tak wygląda definicja klasy w Pythonie.
# W odróżnieniu od Javy czy C++ nie deklaruje się oddzielnie pól klasy, tworzone
# są one w chwili inicjalizacji w konstruktorze.
# Możliwe jest dynamiczne dodawanie pól do już istniejących obiektów.
# Nie stosuje się tutaj też zazwyczaj getterów i setterów chyba że mają jakieś
# dodatkowe funkcjonalnosci, jak na przykład obsługa błędów.
class MyComplex:

    # To jest konstruktor. W pythonie nie ma możliwości przeładowania
    # konstruktora. Dodatkowo w karzdej metodzie klasy trzeba zadeklarować
    # self, odpowiednik this z C, jako jawny parametr
    def __init__(self, re, im):
        # Dobrze jest się w konstruktorze upewnić że wszystko jest takiego typu
        # jak chcemy.
        # Możliwe jest podanie dowolnych typów jako parametry i jeżeli obiekt
        # ma długi cykl życia to błąd może się pojawić bardzo daleko od swojego
        # źródła.
        self.re = float(re)
        self.im = float(im)

    # Przykład najzwyklejszej metody, sprawdza czy dana liczba ma zerową
    # część urojoną.
    def is_real_number(self):
        return self.im == 0  # Do pól trzeba odwoływać się przez self.nazwa_pola

    # Metody specjalne mają na końcu i na początku dwa podkreślniki.
    # Ta funkcja zostanie wywołana kiedy będziemy chcieli przekonwertować obiekt do napisu.
    def __str__(self):
        return '{} + i{}'.format(self.re, self.im)

    # Ta funkcja wywoływana jest też przy konwersji do napisu ale w innych kontekstach.
    # Dla większości podstawowych zastosowań starczy implementacja jak poniżej
    def __repr__(self):
        return self.__str__()

    # Funkcja obsługuje dodawanie
    def __add__(self, other):
        # Jeżeli druga liczba jest też zespolona dodajemy odpowiednio części
        # rzeczywiste i urojone
        if isinstance(other, MyComplex):
            return MyComplex(self.re + other.re, self.im + other.im)
        # Jeżeli jest to liczba rzeczywista do dodajemy ją do części rzeczywistej
        # naszej liczby.
        elif isinstance(other, (int, float)):
            return MyComplex(self.re + other, self.im)
        else:
            raise TypeError()


# Dobrze napisany skrypt w pythonie tak naprawdę nie powinien mieć swojego głównego kodu
# w "zerowym wcięciu" ponieważ powoduje to problemy przy importach.
# Dokładniej taki kod zostanie wywołany w chwili zaimportowania danego skryptu.
# Dlatego należy stworzyć funkcję main() a następnie uruchamiać ją tylko wtedy kiedy skrypt
# został wywołany bezpośrednio z terminala a nie poprzez improt.
def main():
    # Demonstracja naszej klasy
    a = MyComplex(1, 4)
    b = MyComplex(3, 3)
    print('a={}'.format(a))
    print('b={}'.format(b))
    print('a+b={}'.format(a + b))

    # Kilka ciekawych zmiennych specjalnych
    print(__name__) #  Nazwa modółu jeżeli import albo "__main__" jeżli odpalone z konsoli
    print(globals()) #  Słownik zawierający zmienne globalne (można go edytować)
    print(locals())  # Słownik zawierający zmienne lokalne (można go edytować)



# Ta konstrukcja sprawdza czy nasz skrypt został wywołany bezpośrednio z konsoli,
if __name__ == '__main__':
    main()
