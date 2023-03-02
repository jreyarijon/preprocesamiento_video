import os;
import math;
import numpy as np;
import cv2;
import skimage as ski;
from skimage import data, io, filters;
from skimage.feature import canny
from skimage.morphology import disk;
from IPython.display import display, Image;
import time;

# https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
# https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html


VIDEO = "preprocesamiento_video\DATASETS\pelota.mp4"

background = None
cap = cv2.VideoCapture(VIDEO);

while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median = filters.rank.median(gray, disk(10));
    if background is None:
        background = gray
        continue
        
    threshold = cv2.threshold(median, 125, 255, cv2.THRESH_BINARY)[1]

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

    # Filtra los contornos por su área y forma circular
    ball_contour = None
    for contour in contours:

        # Si el area del contorno es menor que 1700, la obviamos. 
        # Asi descartamos contornos que no son de interes.
        if cv2.contourArea(contour) < 1700:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Calculamos la diferencia entre el area del contorno y el area de un circulo. 
        # Si la diferencia es muy grande, el contorno no es muy circular.
        if abs(1 - (cv2.contourArea(contour) / (np.pi * radius**2))) > 0.35:
            continue

        ball_contour = contour
        break
    
    if ball_contour is not None:
        cv2.circle(frame, center, radius, (255,0,0), 2)
        # Dibuja el contorno de la pelota en el frame
        cv2.drawContours(frame, [ball_contour], -1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('median', median)
    cv2.imshow('gray', gray)
    cv2.imshow('Threshold', threshold)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()