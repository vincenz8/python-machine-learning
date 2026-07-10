from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Definir dados de treino (features)
# Formato de cada item: [Quantidade de rodas, Peso em quilos]
x_treino = np.array([
    [4, 1200], # Carro 1
    [4, 1500], # Carro 2
    [2, 200],  # Moto 1
    [2, 250]   # Moto 2
])

# Classificar/agrupar os dados contidos em features (0 para Carro, 1 para Moto)
y_treino = np.array([0, 0, 1, 1]) # Carro 1 e Carro 2 pertencem ao grupo 0 (Carros), e Moto 1 e Moto 2 pertencem ao grupo 1 (Motos)

# Criar o modelo de classificação. Ele usará o KNN configurado para olhar o vizinho mais próximo (n_neighbors=1)
modelo = KNeighborsClassifier(n_neighbors=1)

# Treinar o modelo com dados existentes
modelo.fit(x_treino, y_treino)

# Definir um veículo sem relação direta com y_treino (sem grupo predeterminado)
novo_veiculo = np.array([[2, 180]])

# Prever o tipo (grupo) do veículo comparando as suas características (features) com os dados de treino (features e labels)
previsao = modelo.predict(novo_veiculo)

# Traduzir o resultado numérico (previsão do grupo ao qual o veículo novo pertence) para um tipo de veículo (Carro ou Moto)
if previsao[0] == 0:
  resultado = "Grupo 0: Carro"
else:
  resultado = "Grupo 1: Moto"

# Imprimir o resultado final
print(f"Resultado da previsão para um veículo de 2 rodas e 180kg: {resultado} ")
