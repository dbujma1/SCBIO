import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('modelo_entrenado.h5')  # poné la ruta real
clases = ['A', 'E', 'I', 'O', 'U']  # reemplazá con tus clases reales

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Captura desde webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip para que sea espejo
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convertir a RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar landmarks en la mano
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener el bounding box de la mano
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            x_max = int(max(x_coords) * w) + 20
            y_min = int(min(y_coords) * h) - 20
            y_max = int(max(y_coords) * h) + 20

            # Asegurar que no se salga de la imagen
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Recortar y preprocesar la imagen de la mano
            mano_crop = frame[y_min:y_max, x_min:x_max]
            mano_resize = cv2.resize(mano_crop, (64, 64))  # ajustá si tu modelo usa otro tamaño
            mano_norm = mano_resize.astype('float32') / 255.0
            mano_input = np.expand_dims(mano_norm, axis=0)

            # Predicción
            pred = model.predict(mano_input)
            idx = np.argmax(pred[0])
            letra = clases[idx]

            # Mostrar resultado
            cv2.putText(frame, f'Prediccion: {letra}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar ventana
    cv2.imshow("Comparador LSA", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()
