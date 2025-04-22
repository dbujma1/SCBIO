import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Ruta del dataset y modelo guardado
dataset_dir = 'dataset'  # Cambia esto si tu dataset está en otra ubicación
model_path = 'model/vocal_model.h5'

# Tamaño de las imágenes y batch size
img_size = 50
batch_size = 32

# Cargar el modelo entrenado
model = tf.keras.models.load_model(model_path)

# Preparamos el ImageDataGenerator para validación
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Cargar datos de validación
val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation'  # Solo los datos de validación
)

# Evaluar el modelo en el conjunto de validación
val_loss, val_accuracy = model.evaluate(val_data)

print(f'Pérdida en validación: {val_loss}')
print(f'Precisión en validación: {val_accuracy}')
