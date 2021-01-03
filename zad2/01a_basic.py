# Punktem wejścia programu jest pierwsza komenda napisana
# w zerowym wcięciu.
print('Hello world')

# Jeżeli chcemy wypisać jakąś zmienną która nie jest napisem
# dla typów wbódowanych wystarczy napisać
print(12)
print(23.4)

# Do wypisania sformatowanego napisu tak jak w printf znanym 
# z C należy użyć metody format z klasy string
print('Ala ma {} koty'.format(3))
print('PI = {:2.3}'.format(3.14159265359))

# Dobry opis składni format znajduje się na stronie
# https://www.programiz.com/python-programming/methods/string/format

# W pythonie nie deklaruje się jawnie typów zmiennych jednak ma się 
# na niego wpływ
x = 'abc' # string
x = 'a' # nadal string, nie ma tu typu dla pojedynczego znaku
x = 12 # integer
x = 12.0 # float (tak naprawdę odpowiednik double z C)
# Jak widać na powyższym przykładzie zmienną można redefiniować
# nawet jako zmienną innego typu.

# Innymi ważnymi konstrukcjami w pythonie są listy i słowniki
l = [23, 43, 11, 8] # Lista cztero elementowa
pl = [] # Pusta lista
s = {'Ala':12, 'Jan': 46, 'Roman': 68} # Słownik
ps = {} # Pusty słownik

l.append(6) # Dodajemy element do listy
s['Jadwiga'] = 6 # Dodajemy element do słownika
l[2] = 0 # Zmieniamy wartość w liście
s['Ala'] = 18 # Zmieniamy wartość w słowniku

# Po listach można iterować za pomocą pętli for
for a in l:
    print(a)

# Jeżeli chcemy iterować po słowniku musimy 
# użyć metody która zwróci nam listę par 
# klucz:warotść
for key, value in s.items():
    print('{} : {}'.format(key, value))

# W pythonie nie ma pętli for takiej jak w C
# jeżeli chcemy iterować po kolejnych liczbach
# musimy użyć funckji range która wygeneruje nam 
# listę z liczbami
for i in range(10):
    print(i)

# Poprawny sposób iterowania po liście przy jednoczesnym
# dostępie do indeksu aktualnego elementu wygląda następująco
for index, value in enumerate(l):
    print('{} : {}'.format(index, value))

# Inną możliwą pętlą jest while
a = 3
while a>len(pl):
    pl.append(3)
# len(l) zwraca rozmiar listy l

# Podobnie jak w C w pythonie można przerywać pętle za pomocą 
# instrukcji breaK i continue

# Konstrukcja if else wygląda następująco
if a>3:
    print('a jest większe od 3')
elif a<0:
    print('a jest liczbą ujemną')
else:
    print('a jest w przedziale od 0 do 3')

# Jeżeli lista zawiera typy podstawowe lub dobrze zdefiniowane klasy
# możemy wykorzystać różne przydatne funkcje
print(min(l))
print(max(l))
print(sum(l))
print(sum(l) / len(l))

# Zwróćmy uwagę na to jakie wyniki daje dzielenie (// dzielenie bez reszty)
a = 1
b = 2
c = 1.0
d = 2.0
print('{} ({})'.format(a / b, type(a / b)))
print('{} ({})'.format(a / d, type(a / d)))
print('{} ({})'.format(c / b, type(c / b)))
print('{} ({})'.format(c / d, type(c / d)))
print('{} ({})'.format(a // b, type(a // b)))
print('{} ({})'.format(c // d, type(c // d)))
