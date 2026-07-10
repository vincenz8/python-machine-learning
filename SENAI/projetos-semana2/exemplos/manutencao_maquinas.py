# a IA deverá aprender quando uma máquina precisa de manutenção.
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

dados = {
    # Características de cada máquina, que serão indicadores para a IA descobrir quais precisam de manutenção
    'temperatura': [60,65,70,72,80,85,90,75,68,95,88,62,78,82,66],
    'horas': [120,180,250,300,450,500,600,320,220,700,650,150,400,480,200],
    'vibracao': [2.0,2.2,2.5,2.8,3.6,4.0,4.5,3.0,2.4,5.0,4.2,2.1,3.4,3.8,2.3],
    # Indica quais máquinas precisam de manutenção
    'manutencao': [
        'não',
        'não',
        'não',
        'não',
        'sim',
        'sim',
        'sim',
        'não',
        'não',
        'sim',
        'sim',
        'não',
        'não',
        'sim',
        'não',
    ]
}
# Iniciar e atribuir um dataframe com o dicionário 'dados' à variável 'df',
# e mostrá-lo em forma de tabela no console (saída padrão).
df = pd.DataFrame(dados)
df

# Características das máquinas:
X = df[['temperatura', 'horas', 'vibracao']]
# Conclusão (precisava ou não de manutenção):
y = df['manutencao']

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    # 30% da mostra será usada pra testes e não fará parte do treinamento da IA.
    test_size=0.3,
    random_state=42
)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_treino, y_treino)
# Exemplo de regra que pode ser descoberta pela IA:
# Temperatura > 80 e Vibração > 3.5 -> Precisa de manutenção.

previsoes = modelo.predict(X_teste)

# Comparar resultados (esperados / previstos pela IA)
acuracia = accuracy_score(y_teste, previsoes)
print(f'Acurácia do modelo: {acuracia*100:.2f}%')
resultado = pd.DataFrame({
    'Real': y_teste,
    'Previsto': previsoes
})
resultado # A tabela não aparece caso o programa imprima mais coisas depois dela

# Testando uma nova máquina:
# Temp. 84C, Horas de func. 520, Vibração 4.1
nova_maquina = [[84,520,4.1]]
previsao2 = modelo.predict(nova_maquina)

print(f'A nova máquina precisa de manutenção?\nR: {previsao2[0]}') # A tabela comparativa não aparece

# Imprimir árvore de decisão
plt.figure(figsize=(14,8))
tree.plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=modelo.classes_,
    filled=False # Colorir ou não o diagrama
)
plt.show()
