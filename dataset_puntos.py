import mediapipe as mp
import cv2
import os
import math
import csv

# === CONFIGURACIÓN ===
carpeta_dataset = 'dataset'  # Carpeta con subcarpetas A, E, I, O, U
archivo_salida = 'distancias_con_etiquetas.csv'
pares_de_puntos = [(0, 4), (0, 8), (0, 12), (0, 16), (0, 20)]  # Puedes modificar estos pares

# === FUNCIONES ===
def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# === MediaPipe Hands (modo imagen estática) ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # ¡MUY IMPORTANTE para imágenes!
    max_num_hands=1,
    min_detection_confidence=0.5  # Puedes probar subirlo o bajarlo si falla
)

# === CREAR CSV ===
with open(archivo_salida, mode='w', newline='') as archivo_csv:
    writer = csv.writer(archivo_csv)
    cabecera = [f'dist_{i}_{j}' for (i, j) in pares_de_puntos] + ['etiqueta']
    writer.writerow(cabecera)

    total_detectadas = 0
    total_imagenes = 0

    # Recorremos cada subcarpeta (A, E, I, O, U)
    for clase in os.listdir(carpeta_dataset):
        ruta_clase = os.path.join(carpeta_dataset, clase)
        if not os.path.isdir(ruta_clase):
            continue

        for nombre_imagen in os.listdir(ruta_clase):
            if not nombre_imagen.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            ruta_imagen = os.path.join(ruta_clase, nombre_imagen)
            imagen = cv2.imread(ruta_imagen)

            if imagen is None:
                print(f'⚠️ Imagen inválida: {ruta_imagen}')
                continue

            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            resultado = hands.process(imagen_rgb)
            total_imagenes += 1

            if resultado.multi_hand_landmarks:
                landmarks = resultado.multi_hand_landmarks[0].landmark
                distancias = [calcular_distancia(landmarks[i], landmarks[j]) for (i, j) in pares_de_puntos]
                distancias.append(clase)
                writer.writerow(distancias)
                total_detectadas += 1
            else:
                print(f'🚫 No se detectó mano en: {ruta_imagen}')

# === RESUMEN ===
print(f'\n✅ Total imágenes procesadas: {total_imagenes}')
print(f'✋ Manos detectadas correctamente: {total_detectadas}')
print(f'❌ Fallos de detección: {total_imagenes - total_detectadas}')
