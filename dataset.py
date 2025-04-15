import os
from PIL import Image, ImageEnhance

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
            img = Image.open(img_path).convert("RGB")

            # Nombre base y extensión
            name, ext = os.path.splitext(filename)

            # Redimensionar imagen original (y sobrescribirla)
            img_resized = img.resize(img_size)
            img_resized.save(img_path)

            # Flip horizontal
            img_flip = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip.save(os.path.join(folder_path, f"{name}_flip{ext}"))

            # Zoom in (10%)
            zoom_factor = 0.9  # Recortamos un 10% y reescalamos
            w, h = img_resized.size
            crop_box = (
                int(w * (1 - zoom_factor) / 2),
                int(h * (1 - zoom_factor) / 2),
                int(w * (1 + zoom_factor) / 2),
                int(h * (1 + zoom_factor) / 2)
            )
            img_zoom = img_resized.crop(crop_box).resize(img_size)
            img_zoom.save(os.path.join(folder_path, f"{name}_zoom{ext}"))

            # Reducir brillo en un 10%
            enhancer = ImageEnhance.Brightness(img_resized)
            img_darker = enhancer.enhance(0.9)  # 90% del brillo original
            img_darker.save(os.path.join(folder_path, f"{name}_dark{ext}"))
