import pygame
import random
import cv2
import time
import os
import pygame_menu
from detectar_letra import detectar_letra
from pathlib import Path

current_dir= Path(__file__).resolve().parent
# Inicializar Pygame
pygame.init()
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Sordolingo")

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Letras posibles
letras = ['A', 'E', 'I', 'O', 'U']

# Estilos
font_large = pygame.font.SysFont(None, 120)
font_medium = pygame.font.SysFont(None, 80)
font_small = pygame.font.SysFont(None, 50)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Música
pygame.mixer.music.load(current_dir/'musica.mp3')
pygame.mixer.music.play(-1, 0.0)

# Imágenes
imagenes_path = current_dir/"Imagenes"

def mostrar_texto(texto, fuente, color, x, y, center=True):
    render = fuente.render(texto, True, color)
    rect = render.get_rect(center=(x, y)) if center else render.get_rect(topleft=(x, y))
    screen.blit(render, rect)

def flush_camera(cap, n=5):
    for _ in range(n):
        cap.read()

def juego_de_aprendizaje():
    for ronda in range(len(letras)):
        letra_objetivo = letras[ronda]
        imagen_path = os.path.join(imagenes_path, f"{letra_objetivo}.jpg")
        imagen_letra = pygame.image.load(imagen_path)
        imagen_letra = pygame.transform.scale(imagen_letra, (320, 240))

        screen.fill(WHITE)
        mostrar_texto(f"Letter {letra_objetivo}", font_large, BLACK, WIDTH // 2, 50)
        screen.blit(imagen_letra, (50, HEIGHT // 2 - 120))
        pygame.display.flip()

        acerto = False
        tiempo_inicial = time.time()
        while not acerto:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            letra = detectar_letra(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            frame_surface = pygame.transform.scale(frame_surface, (320, 240))

            screen.fill(WHITE)
            mostrar_texto(f"Letter {letra_objetivo}", font_large, BLACK, WIDTH // 2, 50)
            screen.blit(imagen_letra, (50, HEIGHT // 2 - 120))
            screen.blit(frame_surface, (WIDTH - 340, HEIGHT // 2 - 120))

            if letra == letra_objetivo and time.time() - tiempo_inicial >= 3:
                mostrar_texto("Correct!", font_medium, GREEN, WIDTH // 2, HEIGHT // 2 + 100)
                pygame.display.flip()
                time.sleep(1)
                acerto = True

            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    cap.release()
                    pygame.quit()
                    return
        time.sleep(2)

    screen.fill(WHITE)
    mostrar_texto("You have learnt all the vowels", font_large, BLACK, WIDTH // 2, HEIGHT // 2)
    pygame.display.flip()
    pygame.time.wait(5000)

def juego_contra_reloj():
    puntuacion = 0
    tiempo_por_ronda = 6
    rondas = 5

    for ronda in range(rondas):
        letra_objetivo = random.choice(letras)
        flush_camera(cap, n=15)
        cuenta_regresiva(3)

        inicio_ronda = time.time()
        consec = 0
        ultima = None
        acerto = False

        while time.time() - inicio_ronda < tiempo_por_ronda:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            letra = detectar_letra(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
            frame_surface = pygame.transform.scale(frame_surface, (320, 240))

            if letra == letra_objetivo:
                if letra == ultima:
                    consec += 1
                else:
                    consec = 1
                ultima = letra

                if consec >= 3:
                    puntuacion += 1
                    acerto = True
                    time.sleep(1)
                    break
            else:
                consec = 0
                ultima = None

            screen.fill(WHITE)
            mostrar_texto(f"Score: {puntuacion}", font_medium, BLACK, WIDTH // 2, 40)
            mostrar_texto(f"Round {ronda+1} of {rondas}", font_medium, BLACK, WIDTH // 2, 120)
            tiempo_restante = int(tiempo_por_ronda - (time.time() - inicio_ronda))
            mostrar_texto(f"Time: {tiempo_restante}s", font_medium, BLACK, WIDTH // 2, 200)

            color_letra = GREEN if letra == letra_objetivo else RED
            mostrar_texto(letra_objetivo, font_large, color_letra, WIDTH // 2, HEIGHT // 2 + 120)

            screen.blit(frame_surface, (WIDTH - 340, HEIGHT - 260))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    return

        if not acerto:
            time.sleep(1)

    screen.fill(WHITE)
    mostrar_texto("Finish!", font_large, BLACK, WIDTH // 2, 200)
    mostrar_texto(f"Final score: {puntuacion} of {rondas}", font_medium, BLACK, WIDTH // 2, 300)
    pygame.display.flip()
    pygame.time.wait(5000)

def cuenta_regresiva(segundos=3):
    for t in range(segundos, 0, -1):
        screen.fill(WHITE)
        mostrar_texto(str(t), font_large, BLACK, WIDTH // 2, HEIGHT // 2)
        pygame.display.flip()
        pygame.time.wait(1000)

def mostrar_menu():
    from pygame_menu import themes

    # Crear tema personalizado con selección morada
    tema_personalizado = themes.THEME_BLUE.copy()
    tema_personalizado.selection_color = (128, 0, 128)  # Morado
    tema_personalizado.widget_font_size = 40
    tema_personalizado.title_font_size = 60

    menu = pygame_menu.Menu('SORDOLINGO', WIDTH, HEIGHT, theme=tema_personalizado)

    menu.add.label("Learn sign language", font_size=40)
    menu.add.button('Play against the clock', juego_contra_reloj)
    menu.add.button('Learn the vowels', juego_de_aprendizaje)
    menu.add.button('Exit', pygame_menu.events.EXIT)

    menu.mainloop(screen)

mostrar_menu()

cap.release()
cv2.destroyAllWindows()
pygame.quit()
