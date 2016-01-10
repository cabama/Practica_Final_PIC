import cv2
import copy
import numpy as np


class clasificador:

    """ Clasificacion de la mano """

    # Como argumento de entrada se una el area de la mano y los dedos contados por esqueleto.
    @staticmethod
    def clasify(area,dedosporesqueleto):

        # Primero se analiza el area de la no y despues el numero de dedos ibtenidos con el esqueleto par identificar
        # tres posiciones de la mano
        salida=""
        if area>17000:
            if dedosporesqueleto > 5:
                salida = 5

        if (area>14000 and area<17000):
            if( dedosporesqueleto == 3 or dedosporesqueleto == 4):
                salida = 3

        if area<14000:
            if dedosporesqueleto < 2:
                salida = 0


        return salida


