import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# === Cargar el CSV ===
df = pd.read_csv("distancias_con_etiquetas.csv")

# === Separar características y etiquetas ===
X = df.drop(columns=["etiqueta"]).values
y = df["etiqueta"].values

# === Codificar etiquetas (A, E, I, O, U → 0-4) ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# === Dividir en train y test ===
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# === Definir el modelo ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # 5 clases (A, E, I, O, U)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Entrenar ===
model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

# === Evaluar ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Precisión en test: {accuracy:.4f}")

# === Guardar el modelo ===
model.save("modelo_distancias.h5")
