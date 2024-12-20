# Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

data = {
    'Transacao': [376.24, 489.56, 200.12, 600.45, 122.33, 550.23, 300.45, 430.90],
    'Valor': [141.00, 230.00, 120.00, 300.00, 150.00, 200.00, 100.00, 250.00],
    'ID_Cliente': [13758.91, 13759.00, 13760.23, 13761.12, 13762.54, 13763.33, 13764.23, 13765.12],
    'Suspeita': [0, 1, 0, 1, 0, 1, 0, 1]  # Variável binária
}

df = pd.DataFrame(data)

# Separando variável alvo e preditoras
X = df.drop(columns=['Suspeita'])  # Preditoras
y = df['Suspeita']  # Alvo

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizando as variáveis numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinando o modelo de Regressão Logística
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Avaliando o modelo
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Calculando e plotando a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, marker='o')
plt.title("Curva ROC")
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
plt.show()

print(f"AUC: {roc_auc_score(y_test, y_prob):.2f}")
