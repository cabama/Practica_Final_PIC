import cv2
import copy

class segmentacion:

    # Esta funcion devuelve una mascara en blanco y negro del umbralizado por color,
    # Argumento de entrada: Imagen RGB, Rango de colores minimo y rango de colores maximo,
    # Argumento de salida: imagen hsv, colorMin y color Max

    @staticmethod
    def umbralizacion_hsv(imagenRGB, colorMin, colorMax):
        hsv = cv2.cvtColor(imagenRGB, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, colorMin, colorMax)

    # Esta funcion obtiene la imagen en real con color a partir de una mascara
    @staticmethod
    def mask2imagen(imagen, mascara):
        imagen2 = copy.deepcopy(imagen)
        return cv2.bitwise_and(imagen2, imagen2, mask=mascara)

