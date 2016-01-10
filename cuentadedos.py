import cv2
import copy
import numpy as np


class cuentadedos:

    """Cuenta dedos con el esqueleto"""
    @staticmethod
    def cuentadedosesq(mascara):
        # Con la siguiente funcion se identifican todos los objetos que hay en una imagen binaria
        # En este caso se quieren identificar los objetos que hay en la imagen binaria que contiene el esqueleto de la mano.
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara)

        # De los valores de retorno de la funcion anterior se usa stats,
        # stars es una matriz con 5 columnas (left,top,width,height,area) y un numero de filas en funcion del numero de objetos
        # que hay en la imagen.

        sizearea = stats[:, 4] # vector con todas las areas de los objetos de la imagen:

        cont=0
        labelspeque=[]
        # Se pone como valores limites de area de los objetos 200<area<1000
        # dichos valores se usan para eliminar los segmentos del esqueleto muy pequenos y muy grandes.
        # Se han obtenido despues de un estudio de diferentes manos y posiciones de las mismas.
        areamin=200
        areamax=1000
        for i in sizearea:
            if i > areamin:
                if i < areamax:
                    cont=cont+1
                    #labelspeque  dice los objetos con un area dentro del rango establecido
                    labelspeque.append(cont)
        #cont devuelve el numero de dedos.
        return cont
