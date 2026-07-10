import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Histórico de notas, frequência e resultado dos alunos
dados = {
    "nota1": [9,8,7,6,5,4,10,9,8,2,3,6,7,5,8],
    "nota2": [8,9,7,5,5,3,10,8,7,3,2,6,8,4,9],
    "frequencia": [95,90,85,80,70,60,100,92,88,55,50,75,82,65,91],
    "resultado": [
        "Aprovado",
        "Aprovado",
        "Aprovado",
        "Aprovado",
        "Reprovado",
        "Reprovado",
        "Aprovado",
        "Aprovado",
        "Aprovado",
        "Reprovado",
        "Reprovado",
        "Reprovado",
        "Aprovado",
        "Reprovado",
        "Aprovado",
    ]
}
# Verificar se a longitude dos arrays é igual
# print(f'longitude nota1: {len(dados['nota1'])}')
# print(f'longitude nota2: {len(dados['nota2'])}')
# print(f'longitude frequencia: {len(dados['frequencia'])}')
# print(f'longitude resultado: {len(dados['resultado'])}')

df = pd.DataFrame(dados)
df # Cada linha representa um aluno

# Usando train_test_split pra dividir as entradas das saídas correspondentes
X = df[['nota1', 'nota2', 'frequencia']]
y = df['resultado']
# Criando variáveis para conter os dados processados
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
    # Proporção alunos treino e teste:
    #   70% para treino
    #   30% para teste
)

# Criando o modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_treino, y_treino)

# IA analisou os dados dos alunos do grupo treino e descobriu padrões/regras
# Uma das regras descobertas pela IA durante o treino:
#   if frequencia > 80 and nota media (nota1 e nota2) > 7:
#      return 'Aprovado'

# Determinar o resultado dos alunos separados para teste (30% das entradas)
previsoes = modelo.predict(X_teste)

# Avaliado pela IA
acuracia = accuracy_score(y_teste, previsoes)
print(f'Acurácia dos resultados previstos: {acuracia*100:.2f}%')
# A acurácia equivale à quantidade de resultados que a IA acertou.
# Exemplo: 10 alunos, 9 acertos -> 90% de acurácia

# Comparar resultados reais com os de saída da IA (previsões)
resultado = pd.DataFrame({
    "Real": y_teste,
    "Previsto": previsoes
})
resultado

# Visualizando a árvore de decisões criada pela IA
plt.figure(figsize=(12,7))
tree.plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=modelo.classes_,
    filled=True
)
plt.show()
