import cv2
import mediapipe as mp

# Inicialización de la captura de video
dispositivoCaptura = cv2.VideoCapture(0)  # Puedes probar con 0, 1 o 2 si no funciona el 0

# Verificar si la cámara se abrió correctamente
if not dispositivoCaptura.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Inicialización de MediaPipe para la detección de manos
mpManos = mp.solutions.hands
manos = mpManos.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.9,
                      min_tracking_confidence=0.8)

# Inicialización de las herramientas para dibujar las conexiones de las manos
mpDibujar = mp.solutions.drawing_utils

while True:
    # Captura una imagen
    succes, img = dispositivoCaptura.read()
    
    # Comprobar si la captura de la imagen fue exitosa
    if not succes:
        print("Error al leer la imagen de la cámara.")
        break
    
    # Comprobar si la imagen está vacía
    if img is None or img.size == 0:
        print("La imagen está vacía.")
        break

    # Convertir la imagen a RGB
    try:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print(f"Error en cvtColor: {e}")
        break

    # Procesar la imagen para detectar las manos
    resultado = manos.process(imgRGB)

    # Si se detectan manos, dibujarlas
    if resultado.multi_hand_landmarks:
        for handLms in resultado.multi_hand_landmarks:
            mpDibujar.draw_landmarks(img, handLms, mpManos.HAND_CONNECTIONS)
    
    # Mostrar la imagen con las manos detectadas
    cv2.imshow("Imagen", img)

    # Esperar la tecla "Esc" para salir
    if cv2.waitKey(1) & 0xFF == 27:  # 27 es el código de la tecla ESC
        break

# Liberar la cámara y cerrar las ventanas
dispositivoCaptura.release()
cv2.destroyAllWindows()
