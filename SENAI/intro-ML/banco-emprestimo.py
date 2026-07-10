# Cenário: Você trabalha em um banco. Crie um classificador para decidir se um cliente poderá receber um empréstimo ou não (0 = Negado, 1 = Aprovado).
# 1. Features: [Idade, Renda mensal em R$, Possui dívidas (0=Não, 1=Sim)].
# 2. Crie ao menos 6 exemplos de clientes para treinar o modelo (misture clientes bons e ruins).
# 3. Teste o modelo criando um cliente novo. Ele deve ter valores diferentes porém seguir a mesma estrutura de dados do array features.

import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 1.1. Dados do treino
historico_clientes = [
    [24, 2500, 0], # Cliente Robson. Jovem, renda média e sem dívidas -> Aprovado
    [27, 1200, 0], # Cliente João. Jovem, renda muito baixa e sem dívidas -> Negado
    [42, 12000, 1], # Cliente Pedro. Adulto, renda muito alta e com dívidas -> Negado
    [33, 8000, 1], # Cliente Cláudio. Adulto, renda alta e sem dívidas -> Aprovado
    [39, 2900, 1], # Cliente José. Adulto, renda média e com dívidas -> Negado
    [23, 1500, 0], # Cliente Vicente. Jovem, renda baixa e sem dívidas -> Aprovado
]

# 1.2. Classificação dos clientes: Aprovado (1) ou Negado (0)
decisoes_historico = np.array([1, 0, 0, 1, 0, 1])

# 2. Criar e treinar o modelo
modelo_banco = DecisionTreeClassifier(random_state=42)
modelo_banco.fit(historico_clientes, decisoes_historico)

# 3.1. Criar um cliente novo
clientes_novos = np.array([[39, 2900, 1], [24, 2900, 1]])
# 3.2. Prever se o cliente terá seu empréstimo aprovado ou negado
previsao = modelo_banco.predict(clientes_novos)

# Imprimir resultado
def imprimir_resultado(previsao, posicao):
  if previsao[posicao] == 1:
    print('Empréstimo aprovado.')
  else:
    print('Empréstimo negado.')

imprimir_resultado(previsao, 0) # Negado, idade muito alta (2 negados, 1 aprovado para idade > 30)
imprimir_resultado(previsao, 1) # Aprovado, idade dentro do límite da regra (2 aprovados, 1 negado para idade < 30)
