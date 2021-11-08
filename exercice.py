#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, num=64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([(np.sqrt(c[0]**2+c[1]**2), np.arctan2(c[1],c[0])) for c in cartesian_coordinates ])


def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()

def creer_graphe():
    x=np.linspace(-1,1, num=250)
    y=x ** 2 * np.sin(1 / x ** 2) + x
    
    plt.scatter(x, y)
    plt.title("Graphe")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    

def Monte_Carlo(itt=5000):
    x_inside=[]
    y_inside=[]
    x_outside=[]
    y_outside=[]
    for i in range(itt):
        x=np.random.random()
        y=np.random.random()
        if np.sqrt(x**2+y**2) <= 1.0 :
            x_inside.append(x)
            y_inside.append(y)
        else:
            x_outside.append(x)
            y_outside.append(y)

    plt.scatter(x_inside,y_inside)
    plt.scatter(x_outside,y_outside)
    plt.title('Monte carlo')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    return float(len(x_inside)) / itt * 4


def Integrale():
    invexp = lambda x: np.exp(-x**2)
    resultat=quad(invexp, -np.inf, np.inf)
    
    x=np.arange(-4,4,0.1)
    y=[quad(invexp,0, i)[0] for i in x]
    
    plt.title('Integrale')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x,y)
    plt.show()
    return resultat
     
    
if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print('les valeurs sont : ', linear_values())
    coordonee=np.array([(0, 0), (10, 10), (2, -1)])
    print(f'les coordonnées cartésiennes {coordonee} en coordonnées polaires {coordinate_conversion(coordonee)}')
    print(find_closest_index(np.array([1, 3, 8, 10]), 9.5))
    creer_graphe()
    Monte_Carlo()
    print(Integrale())