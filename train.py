import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Configuración
img_size = 50  
batch_size = 32

# Preprocesamiento y validación con aumento de datos
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,  # Rango de rotación
    width_shift_range=0.2,  # Desplazamiento horizontal
    height_shift_range=0.2,  # Desplazamiento vertical
    shear_range=0.2,      # Cizalladura
    zoom_range=0.2,       # Zoom
    horizontal_flip=True,  # Volteo horizontal
    fill_mode='nearest'    # Cómo rellenar los píxeles vacíos
)

train_data = datagen.flow_from_directory(
    'dataset',
    target_size=(img_size, img_size),
    class_mode='categorical',
    batch_size=batch_size,
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset',
    target_size=(img_size, img_size),
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation'
)

# Modelo con Dropout
model = Sequential([
    Conv2D(32, (5,5), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (5,5), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (5,5), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),  # Dropout para reducir el overfitting
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping: detener si no mejora en 5 épocas
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento
model.fit(train_data, validation_data=val_data, epochs=70, callbacks=[early_stop])

# Guardar modelo
os.makedirs("model", exist_ok=True)
model.save("model/vocal_model.h5")