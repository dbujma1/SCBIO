import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Ruta del dataset de test externo y modelo entrenado
test_dataset_dir = 'test_dataset'  # Tu carpeta de test
model_path = 'model/vocal_model.h5'  # Modelo entrenado

# Tamaño de imágenes y batch
img_size = 50
batch_size = 32

# Cargar el modelo entrenado
model = tf.keras.models.load_model(model_path)

# Crear generador para el conjunto de test (sin particiones)
datagen = ImageDataGenerator(rescale=1./255)

# Cargar datos de test, redimensionando imágenes a 50x50
test_data = datagen.flow_from_directory(
    test_dataset_dir,
    target_size=(img_size, img_size),  # Redimensionar a 50x50
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False  # No barajar para análisis posterior
)

# Evaluar el modelo en el conjunto de test
test_loss, test_accuracy = model.evaluate(test_data)

# Mostrar resultados
print(f'📉 Pérdida en test: {test_loss}')
print(f'✅ Precisión en test: {test_accuracy}')

# Predecir las clases en el conjunto de test
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

# Obtener las etiquetas reales del conjunto de test
y_true = test_data.classes

# Crear la matriz de confusión
cm = confusion_matrix(y_true, y_pred_classes)

# Visualizar la matriz de confusión con un heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
plt.show()
