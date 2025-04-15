import os
from PIL import Image

# Ruta raíz
base_dir = 'dataset'
vocales = ['A', 'E', 'I', 'O', 'U']

# Tamaño deseado
img_size = (64, 64)

for vocal in vocales:
    folder_path = os.path.join(base_dir, vocal)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)

            # Nombre base y extensión
            name, ext = os.path.splitext(filename)

            # Redimensionar imagen original (y sobrescribirla)
            img_resized = img.resize(img_size)
            img_resized.save(img_path)

            # Flip horizontal
            img_flip = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip.save(os.path.join(folder_path, f"{name}_flip{ext}"))

            # Rotar 20° a la izquierda y redimensionar
            img_rot_left = img_resized.rotate(20, expand=True).resize(img_size)
            img_rot_left.save(os.path.join(folder_path, f"{name}_rot20{ext}"))

            # Rotar 20° a la derecha y redimensionar
            img_rot_right = img_resized.rotate(-20, expand=True).resize(img_size)
            img_rot_right.save(os.path.join(folder_path, f"{name}_rot_20{ext}"))
