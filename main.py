#!/usr/bin/python
# -*- coding: utf-8 -*-
# Code is commented in French because I've to used it at school.


from PIL import Image as PILImage, ImageOps
from pylab import imshow, show, title
from cmath import exp, pi
from typing import *

number = NewType('number', Union[int, complex, float])


def next_power_of_2(n: int) -> int:
    """
    Renvoie la plus petite puissance de 2 supérieure ou égale à n
    Exemple
        next_power_of_2(5) → 8
    :param n Nombre à majorer par une puissance de 2
    :return Puissance de 2
    """
    count = 0
    if n and not (n & (n - 1)):
        return n

    while n != 0:
        n >>= 1
        count += 1

    return 1 << count


def complete(pixels: List[number]) -> List[number]:
    """
    Complete une liste pour qu'elle soit de la taille 2^m x 2^n la plus petite possible
    Exemple
        complete([[1, 2, 3], [3, 4, 5], [6, 6, 8]]) → [[1, 2, 3, 0], [3, 4, 5, 0], [6, 6, 8, 0], [0, 0, 0, 0]]
    :param pixels Tableau de pixels à compléter
    :return Tableau complété
    """
    h, l = len(pixels), len(pixels[1])
    for i in range(h):
        pixels[i] += [0]*(next_power_of_2(l)-l)
    for i in range(next_power_of_2(h)-h):
        pixels.append([0]*next_power_of_2(l))
    return pixels


def cut_to_size(pixels: List[List[number]], height: int, width: int) -> List[List[number]]:
    """
    Renvoie le tableau de pixels coupé aux tailles height et width précisées.
    Exemple
        cut_to_size([[1, 2], [2, 3]], 1, 1) → [[1], [2]]
    :param pixels Tableau de pixels
    :param height Hauteur
    :param width Largeur
    :return Tableau découpé
    """
    return [x[:width] for x in pixels[:height]]


def omega(n: int, i: int) -> complex:
    """
    Calcule le omega dans la transformée de Fourier.
    C'est à dire que la fonction calcule la i-ème racine n-ième de l'unité
    ! En python le i mathématique est noté j
        -> Voir: https://bugs.python.org/issue10562
    Exemple:
        omega(1, 0) => (1+0j)
        omega(1, 1.5) => (-1-3.6739403974420594e-16j)
    :param n: Nombre de valeurs total
    :param i: Indice actuel
    :return: omega(n, i)
    """
    return exp((2.0 * pi * 1j * i) / n)


def real(v: List[List[number]]) -> List[List[float]]:
    """
    Renvoie la liste v avec la partie réelle de chacun de ses cellules.
    Exemple
        → real([[1+3j, 4+23j], [3+9j, 23-9j]]) → [[1, 4], [3, 23]]
    :param v Liste à convertir
    :return Liste réelle
    """
    new_list = []
    m, n = len(v), len(v[0])
    for i in range(m):
        d = []
        for j in range(n):
            d.append(v[i][j].real)  # Le nombre droit être un complex (pour avoir l'attribut real)
        new_list.append(d)
    return new_list


def transpose(li: List[List[number]]) -> List[List[number]]:
    """
    Renvoie la transposée sous forme de liste de listes.
    Explication détaillée avec une liste originale : [[1, 2], [3, 4]]
        → * l'étoile sert au "unpacking" de la liste, c'est-à-dire séparer chaque élément de la liste.
            → Exemple *[[1, 2], [3, 4]] → [1, 2], [3, 4]
        → zip() permet de faire la transposée en créant un tuple de tuple
            → Exemple zip([1, 2], [3, 4]) → ((1, 3), (2, 4)) sous forme d'objet (non visible sans conversion)
        → map(list, ...) permet de convertir chaque sous-tuple en liste
            → Exemple map(list, zip((1, 3), (2, 4))) → ([1, 3], [2, 4]) sous forme d'objet (non visible)
        → list() pour convertir le tout en liste et revenir sur un objet de la même forme que celui de début
            → Exemple list(map(list, zip((1, 3), (2, 4)))) → [[1, 3], [2, 4]]
    :param li Liste 2D à transposer (matrice)
    :return Liste transposée
    """
    return list(map(list, zip(*li)))


def multiply_list(s: number, v: List[number]):
    """
    Multiplie toute une liste v par un nombre s.
    → Reconstruit une nouvelle liste qui est retournée
    Exemple
        multiply_list(5, [1, 5, 4, 2]) → [5, 25, 20, 10].
    :param s Scalaire à appliquer à la liste v
    :param v Liste à multiplier
    :return Liste avec chaque élément multiplié par le scalaire s
    """
    new_l = []
    for item in v:
        new_l.append(item * s)
    return new_l


def add_list(v: List[number], v1: List[number]) -> List[number]:
    """
    Ajoute élément par élément deux listes.
    Exemple
        add_list([1, 2, 3], [3, 5, 3]) → [4, 7, 6]
    :param v Première liste
    :param v1 Deuxième liste
    :return Nouvelle liste où l'élément x est la somme des éléments se trouvant à la même position dans v et v1.
    """
    new_l = []
    for index, item in enumerate(v1):
        new_l.append(item + v[index])
    return new_l


def minus_list(v: List[number], v1: List[number]) -> List[number]:
    """
    Soustrait deux listes éléments par élément (v - v1)
    Exemple
        minus_list([4, 5, 6], [1, 2, 3]) → [3, 3, 3]
    :param v Liste v à laquelle on va soustraire une quantité
    :param v1 Deuxième liste
    :return Nouvelle liste où chaque élément d'index x est v[x] - v1[x]
    """
    new_l = []
    for index, item in enumerate(v1):
        new_l.append(v[index] - item)
    return new_l


def normalize_list2(v: List[List[number]]) -> List[List[number]]:
    """
    Normalise une liste v en 2D.
    La méthode multiplie l'ensemble des éléments de v par 1/n avec n le nombre de lignes de la liste.
    Exemple
        normalize_list2([[4, 6], [8, 10]]) → [[2.0, 3.0], [4.0, 5.0]]
    :param v Liste à normaliser
    :return Nouvelle liste normalisée
    """
    new_list = []
    for i in range(len(v)):
        new_list.append([])
        for j in v[i]:
            new_list[i].append(j*float(1/len(v)))
    return new_list


def dft(x: List[List[number]]) -> List[List[number]]:
    """
    Transformation de Fourier 1D.
    :param x Tableau 1D à transformer
    :return Tableau transformé
    """
    size = len(x)
    new_list = [[0]*size]*size
    for m in range(size):
        for n in range(size):
            new_list[m] = add_list(new_list[m], multiply_list(omega(size, n*m), x[n]))
    return new_list


def idft(x: List[List[number]]) -> List[List[number]]:
    """
    Transformation inverse de Fourier 1D.
    :param x Tableau 1D à transformer
    :return Tableau transformé
    """
    size = len(x)
    new_list = [[0]*size]*size
    for m in range(size):
        for n in range(size):
            new_list[m] = add_list(new_list[m], multiply_list(omega(size, -n*m), x[n]))
    return new_list


def fft(x: List[List[number]]) -> List[List[number]]:
    """
    Transformation de Fourier rapide 1D.
    :param x Tableau 1D à transformer
    :return Tableau transformé
    """
    n = len(x)
    if n == 1:
        return x
    pair, impair = fft(x[0::2]), fft(x[1::2])
    combined = [0] * n
    to = int(n / 2)
    for m in range(to):
        combined[m] = add_list(pair[m], multiply_list(omega(n, m), impair[m]))
        combined[int(m + n / 2)] = minus_list(pair[m], multiply_list(omega(n, m), impair[m]))
    return combined


def ifft(x: List[List[number]]) -> List[List[number]]:
    """
    Transformation inverse de Fourier rapide 1D.
    :param x Tableau 1D à transformer
    :return Tableau transformé
    """
    n = len(x)
    if n == 1:
        return x
    pair, impair = ifft(x[0::2]), ifft(x[1::2])
    combined = [0] * n
    to = int(n / 2)
    for m in range(to):
        combined[m] = add_list(pair[m], multiply_list(omega(n, -m), impair[m]))
        combined[int(m + n / 2)] = minus_list(pair[m], multiply_list(omega(n, -m), impair[m]))
    return combined


def dft2(f: List[List[number]]) -> List[List[number]]:
    """
    Transformée de Fourier 2D classique (complexité o(n^2))
    :param f Tableau 2D sur lequel on va appliquer la transformation
    :return Tableau transformé (avec complexes)
    """
    return transpose(dft(transpose(dft(f))))


def idft2(f: List[List[number]]) -> List[List[number]]:
    """
    Transformée inverse de Fourier 2D classique (complexité o(n^2))
    :param f Tableau 2D sur lequel on va appliquer la transformation inverse
    :return Tableau transformé (avec complexes)
    """
    return normalize_list2(transpose(idft(normalize_list2(transpose(idft(f))))))


def fft2(f: List[List[number]]) -> List[List[number]]:
    """
    Transformation de Fourier rapide 2D (complexité o(nlog(n)))
    :param f Tableau 2D sur lequel on va appliquer la transformation
    :return Tableau transformé (avec complexes)
    """
    return transpose(fft(transpose(fft(f))))


def ifft2(f: List[List[number]]) -> List[List[number]]:
    """
    Transformation inverse de Fourier rapide 2D (complexité o(nlog(n)))
    :param f Tableau 2D sur lequel on va appliquer la transformation inverse
    :return Tableau transformé (avec complexes)
    """
    return normalize_list2(transpose(ifft(normalize_list2(transpose(ifft(f))))))


def main() -> NoReturn:
    """
    Fonction principale du programme
    """
    im = PILImage.open("panda.jpg")
    gray_image = ImageOps.grayscale(im)
    pixels = list(gray_image.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

    # Affichage de l'image originale
    title("Image originale en niveau de gris")
    imshow(pixels, cmap='gray', vmin=0, vmax=255)
    show()

    pixels = complete(pixels)

    # Calcul de la transformée de Fourier et affichage
    array2 = fft2(pixels)
    title("Transformée de Fourier")
    imshow(real(array2), cmap='gray', vmin=0, vmax=255)
    show()

    # Calcul de la transformée inverse et affichage
    array2 = ifft2(array2)
    if (width, height) != (len(pixels[0]), len(pixels)):
        array2 = cut_to_size(array2, height, width)
    title("Transformée inverse de Fourier")
    imshow(real(array2), cmap='gray', vmin=0, vmax=255)
    show()


if __name__ == '__main__':
    main()
