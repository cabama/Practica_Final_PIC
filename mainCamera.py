import sys
import cv2
import numpy as np

from segmentacion import segmentacion
from contornos import contornos
from cuentadedos import cuentadedos
from clasificador import clasificador

# Definimos una funcion para cerrar  la ventana abierta
cap = cv2.VideoCapture(0)

while (1):

    #Se captura el video
    ret, fotocv = cap.read()
    fotocv = cv2.blur(fotocv, (10, 10)) # Se aplica un filtro gausiano para mejorar la robustezdel programa
    height, width = fotocv.shape[:2]
    y1=(height/2)
    x1=(width/2)

    # Umbralizamos un rango de colores (rango de colores azul en este caso)

    color_minimo = np.array([100,150, 0], dtype="uint8")  # color minimo del filtrado
    color_maximo = np.array([140, 255, 255], dtype="uint8") # color maximo del filtrado
    mascara = segmentacion.umbralizacion_hsv(fotocv, color_minimo, color_maximo)  # Mascara de la imagen umbralizada, imagen binaria
    imagen_segmentada  = segmentacion.mask2imagen(fotocv, mascara)  # Imagen filtrada a partir de la mascara

    # Ahora se van a obtener los diferentes contornos que tiene la mascara (Imagen binaria).
    all_contornos = contornos.contornos(mascara)

    # Si tenemos algun contorno continuamos.
    # Si no tenemos contornos tampoco tenemos mano entonces no tenemos que emplear el algoritmo
    if len(all_contornos) > 0:

        # Obtenemos el mayor contorno de la mano
        contorno_mano, area = contornos.contorno_mayor(fotocv, all_contornos)
        # Obtenemos el numero de dedos, area ...
        num_dedos, _, foto_mano_puntos, foto_contorno_alone = contornos.convexiHull(fotocv, contorno_mano)
        print("area de la mano = " + str(area)) # Se saca por pantalla el area de la mano
        # Pasamos el contorno de la imagen a imagen binaria
        mano_binaria = contornos.contorno2binarie(fotocv.shape[:2], contorno_mano)
        # Esqueletizamos la imagen con el objetivo de poder obtener un segmento decada uno de los dedos.
        esqueleto = contornos.esqueletizar(mano_binaria)
        # Ahora se cuentan los dedos que hay en la mano en cada momento gracias al esqueleto
        dedosesq=cuentadedos.cuentadedosesq(esqueleto)
        # Se clasifica:
        salida=clasificador.clasify(area,dedosesq)

       # Ahora se sacan por patalla por un lado la imagen real y por otro lado
       # una imagen donde se puede visualizar mejor los difrentes gestos de la mano y pasos del programa

        print ("numerod dedos segun esqueleto = " + str(dedosesq)) #Saca por pantalla el numero de dedos

       # Aqui se forma la imagen con el esqueleto de la mano,los contornos de la mano en RGB
        # y un letrero indicando el numero de dedos en cada momento.
        imagen_esqueleto = np.zeros(fotocv.shape)
        esqueletorgb = cv2.cvtColor(esqueleto,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(esqueletorgb, [contorno_mano], 0, (255, 0, 0), 2)
        if salida == 5 :
            cv2.putText(esqueletorgb," Estas mostrando 5 dedos ", (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255,0,0),2)
        if salida == 3 :
            cv2.putText(esqueletorgb," Estas mostrando 3 dedos ", (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255,0,0),2)
        if salida == 0 :
            cv2.putText(esqueletorgb," Estas mostrando 0 dedos ", (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255,0,0),2)

    # Se saca por pantalla por un lado la imagen real,
    # en dicha imagen si pulsas la letra A aparece un rectangulo de color azul para centrar la mano.
    A=cv2.waitKey(5) & 0xFF
    if A == 97:
        cv2.rectangle(fotocv ,(x1-80,y1-130),(x1+80,y1+130),(255,0,0),3)

    cv2.imshow('imagen original', fotocv)
    #Por otro lado la imagen con el esqueleto y los contornos.
    cv2.imshow('esqueleto', esqueletorgb)


    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break

cv2.destroyAllWindows()

