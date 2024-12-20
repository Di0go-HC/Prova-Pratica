import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'Transacao': [376.24, 489.56, 200.12, 600.45, 122.33, 550.23, 300.45, 430.90],
    'Valor': [141.00, 230.00, 120.00, 300.00, 150.00, 200.00, 100.00, 250.00],
    'ID_Cliente': [13758.91, 13759.00, 13760.23, 13761.12, 13762.54, 13763.33, 13764.23, 13765.12]
}

df = pd.DataFrame(data)

# Normalizando os dados para garantir que todas as variáveis tenham o mesmo peso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Definir o número de clusters (K) usando o método do cotovelo
# Vamos testar de 1 a 6 clusters
inertia = []
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k, random_state=23)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotando o gráfico de cotovelo
plt.plot(range(1, 7), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.show()

# A partir do gráfico de cotovelo, pode-se definir o número de clusters ideal
# Agora, vamos aplicar o K-Means com o número de clusters escolhido (suponhamos K=3)
kmeans = KMeans(n_clusters=3, random_state=23)
kmeans.fit(X_scaled)

# Adicionando os resultados de cluster no DataFrame
df['Cluster'] = kmeans.labels_

# Visualizando os clusters
print(df)

# Visualizando os clusters com um gráfico (apenas 2 variáveis para simplificação)
plt.scatter(df['Transacao'], df['Valor'], c=df['Cluster'], cmap='viridis')
plt.title('Clusterização de Transações')
plt.xlabel('Transação')
plt.ylabel('Valor')
plt.show()
