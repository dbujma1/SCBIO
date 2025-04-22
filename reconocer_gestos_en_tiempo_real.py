import mediapipe as mp
import cv2
import math
import numpy as np
import tensorflow as tf

# === Cargar modelo ===
model = tf.keras.models.load_model('modelo_distancias.h5')
labels = ['A', 'E', 'I', 'O', 'U']  # Deben coincidir con el orden del entrenamiento
pares_de_puntos = [(0, 4), (0, 8), (0, 12), (0, 16), (0, 20)]

# === Función para calcular distancias ===
def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# === Inicializar MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Webcam ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)

    if resultado.multi_hand_landmarks:
        landmarks = resultado.multi_hand_landmarks[0].landmark

        # Calcular distancias y hacer predicción
        distancias = [calcular_distancia(landmarks[i], landmarks[j]) for (i, j) in pares_de_puntos]
        entrada_modelo = np.array(distancias).reshape(1, -1)  # Convertir a forma (1,5)
        predicciones = model.predict(entrada_modelo)
        indice_predicho = np.argmax(predicciones)
        letra = labels[indice_predicho]

        # Dibujar mano y letra
        mp_drawing.draw_landmarks(frame, resultado.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f'Letra: {letra}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No se detecta mano', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow('Reconocimiento de gesto', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Liberar recursos ===
cap.release()
cv2.destroyAllWindows()
