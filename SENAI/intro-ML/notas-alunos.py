import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Dados de treino: [Nota final: 0 a 10.0, Frequência em % (0 a 100)]
historico_alunos = np.array([
    [8.5, 95], # Aprovado
    [4.0, 90], # Reprovado
    [7.0, 80], # Aprovado
    [9.0, 50], # Reprovado
    [3.5, 40], # Reprovado
    [6.5, 78]  # Aprovado
])

# Respostas para a IA (1 = Aprovado, 0 = Reprovado)
aprovados_historico = np.array([1, 0, 1, 0, 0, 1])

# Criar e treinar modelo, adicionando o parâmetro random_state para manter a consistência do resultado
modelo_escola = DecisionTreeClassifier(random_state=42)
modelo_escola.fit(historico_alunos, aprovados_historico)

# Testar o modelo com dois alunos
alunos_novos = np.array([
    [7.5, 85], # Deverá ser aprovado
    [8.0, 55]  # Deverá ser reprovado
])

array_previsoes = modelo_escola.predict(alunos_novos)

def determinar_resultado(previsao, nome_aluno):
  if previsao == 1:
    print('O aluno ' + nome_aluno + ' foi APROVADO.')
  else:
    print('O aluno ' + nome_aluno + ' foi REPROVADO.')

determinar_resultado(array_previsoes[0], 'Joãozinho')
determinar_resultado(array_previsoes[1], 'Pedrinho')

print('')
# Imprimir de um jeito diferente (método da professora)
nomes = ['Joãozinho', 'Pedrinho']
for nome, previsao in zip(nomes, array_previsoes):
  situacao = 'REPROVADO(A)' if previsao == 0 else 'APROVADO(A)'
  print(f'Aluno {nome}: {situacao}')
