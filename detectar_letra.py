import mediapipe as mp
import math
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

current_dir= Path(__file__).resolve().parent
model = tf.keras.models.load_model(current_dir/'modelo_distancias.h5')
labels = ['A', 'E', 'I', 'O', 'U']
pares_de_puntos = [(0, 4), (0, 8), (0, 12), (0, 16), (0, 20)]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detectar_letra(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)

    if resultado.multi_hand_landmarks:
        landmarks = resultado.multi_hand_landmarks[0].landmark
        distancias = [calcular_distancia(landmarks[i], landmarks[j]) for (i, j) in pares_de_puntos]
        entrada_modelo = np.array(distancias).reshape(1, -1)
        predicciones = model.predict(entrada_modelo)
        indice_predicho = np.argmax(predicciones)
        letra = labels[indice_predicho]
        return letra

    return None
