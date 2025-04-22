import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Cargar el archivo CSV con las distancias y las etiquetas
df = pd.read_csv('test_distancias_con_etiquetas.csv')

# Asegúrate de que el CSV tiene las columnas de distancias y etiquetas
print(df.head())  # Esto te ayudará a ver cómo están estructurados los datos

# Extraer las distancias (X) y las etiquetas (y) del dataframe
X_test = df[['dist_0_4', 'dist_0_8', 'dist_0_12', 'dist_0_16', 'dist_0_20']].values  # Las distancias
y_test = df['etiqueta'].values  # Las etiquetas

# Convertir las etiquetas a números (por ejemplo, 'A' -> 0, 'E' -> 1, etc.)
# Aquí asumimos que las etiquetas son letras y las convertimos a números de manera simple
labels = {'A': 0, 'E': 1, 'I': 2, 'O': 3, 'U': 4}  # Asegúrate de tener todas las letras que usas
y_test_num = np.array([labels[label] for label in y_test])

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_distancias.h5')

# Hacer predicciones con el modelo
predictions = model.predict(X_test)

# Las predicciones son probabilidades, por lo que necesitamos obtener la etiqueta con la probabilidad más alta
predicted_labels = np.argmax(predictions, axis=1)

# Calcular el porcentaje de aciertos
accuracy = accuracy_score(y_test_num, predicted_labels)

print(f'Porcentaje de aciertos: {accuracy * 100:.2f}%')
