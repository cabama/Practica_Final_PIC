import cv2
import copy
import numpy as np


class contornos:

    """Se retorna 1 todos los contornos obtenidos y 2 los contornos dibujados sobre la foto original"""
    @staticmethod
    def contornos(mascara):
        im2, contornos, hierarchy = cv2.findContours(mascara, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        return contornos

    # Esta funcion retorna todos los contornos obtenidos del objeto con mayor area de la imagen, es decir, la mano.
    @staticmethod
    def contorno_mayor(imagen_original, contornos):
        max_area = 0  # Area maxima del contorno

        # Recorremos los contornos en busca del que tiene el mayor area
        ci = 0
        for i in range(len(contornos)):
                cnt = contornos[i]
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    ci = i

        # Al conocer el mayor contorno lo consideramos la mano
        contorno_mano = contornos[ci]
        return contorno_mano, max_area

    """A partir del contorno de la imagen crea una imagen binaria"""
    @staticmethod
    def contorno2binarie(shape, contorno, color=(255, 255, 255)):
        # Creamos una imagen vacia del tamano de la imagen pasada
        manoBinaria = np.zeros(shape, np.uint8)
        cv2.drawContours(manoBinaria, [contorno], 0, (255, 255, 255), -1)
        return manoBinaria


    """Se retorna 1 el numero de dedos, 2 el mayor contorno (el de la mano), 3 este contorno mayor dibujado sobre la foto original"""
    @staticmethod
    def convexiHull(imagenOriginal, contorno_mano):
        # Copiamos la imagen_copia porque si no se pasa por referencia y se aplican los cambios a ella
        imagenCopia = copy.deepcopy(imagenOriginal)

        # Creamos un vector vacio del tamano de la imagen
        imagen_contorno_alone = np.zeros(imagenCopia.shape, np.uint8)

        # Aplicamos Hull y momentos al cotorno de la mano
        hull = cv2.convexHull(contorno_mano)
        moments = cv2.moments(contorno_mano)

        cx = 0
        cy = 0
        if moments['m00'] != 0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00

        centr = (cx, cy)
        cv2.circle(imagenCopia, centr, 5, [0, 0, 255], 2)
        cv2.drawContours(imagen_contorno_alone, [contorno_mano], 0, (0, 255, 0), 2)


        contorno_mano2 = cv2.approxPolyDP(contorno_mano,0.01*cv2.arcLength(contorno_mano,True),True)
        hull = cv2.convexHull(contorno_mano2, returnPoints=False)
        defects = cv2.convexityDefects(contorno_mano2, hull)
        mind=0
        maxd=0
        i = 0
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contorno_mano2[s][0])
            end = tuple(contorno_mano2[e][0])
            far = tuple(contorno_mano2[f][0])
            dist = cv2.pointPolygonTest(contorno_mano2, centr, True)
            cv2.line(imagenCopia, start, end, [0, 255, 0], 2)
            cv2.circle(imagenCopia, far, 5, [0, 0, 255], -1)



        return i, contorno_mano, imagen_contorno_alone, imagenCopia


    """Esqueleto de la mano"""
    @staticmethod
    def esqueletizar(imagenUmbralizada):

        size = np.size(imagenUmbralizada)

        # Creamos una variable esqueleto del tamano de la imagen
        binario_esqueleto = np.zeros(imagenUmbralizada.shape, np.uint8)

        # Creamos un elemento estructurante en forma de cruz
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        # Bucle dentro del cual se erosiona, se dilata, se substrae y se hace
        # una operacion disjunta.
        while not done:
            eroded = cv2.erode(imagenUmbralizada,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(imagenUmbralizada,temp)
            binario_esqueleto = cv2.bitwise_or(binario_esqueleto,temp)
            imagenUmbralizada = eroded.copy()

            zeros = size - cv2.countNonZero(imagenUmbralizada)
            if zeros == size:
                done = True

        binario_esqueleto = contornos.mejorar_esqueleto(binario_esqueleto)

        return binario_esqueleto

    @staticmethod
    def mejorar_esqueleto(binario_esqueleto):
        # Como al buscar los contornos del esqueleto se nos plantea el problema que son muy finos.
        # Vamos a dilatar otra vez el esqueleto para obtener mejor su contorno.

        # Creamos el kernel de la dilatacion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Dilatamos la imagen
        dilation = cv2.dilate(binario_esqueleto, kernel, iterations=1)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel2)
        dilation = cv2.dilate(opening, kernel3, iterations=1)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel4)

        return opening
