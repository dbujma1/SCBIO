import os
from PIL import Image

# Ruta raíz
base_dir = 'dataset'
vocales = ['A', 'E', 'I', 'O', 'U']

for vocal in vocales:
    folder_path = os.path.join(base_dir, vocal)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)

            # Nombre base y extensión
            name, ext = os.path.splitext(filename)

            # Flip horizontal
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip.save(os.path.join(folder_path, f"{name}_flip{ext}"))

            # Rotar 20° a la izquierda (positivo)
            img_rot_left = img.rotate(20, expand=True)
            img_rot_left.save(os.path.join(folder_path, f"{name}_rot20{ext}"))

            # Rotar 20° a la derecha (negativo)
            img_rot_right = img.rotate(-20, expand=True)
            img_rot_right.save(os.path.join(folder_path, f"{name}_rot_20{ext}"))

