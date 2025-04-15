import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# Configuración
img_size = 64  
batch_size = 32

# Preprocesamiento y validación
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

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

# Modelo
model = Sequential([
    Conv2D(32, (5,5), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (5,5), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (5,5), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
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
