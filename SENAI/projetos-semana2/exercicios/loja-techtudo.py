import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print ( " --- INICIANDO O SISTEMA DE IA DA TECHTUDO --- \ n" )
# 1. Os Dados "Sujos" que recebemos do banco de dados
dados_brutos = {
'Tempo_Site' : [ 15.5 , 5.0 , None , 45.2 , 12.0 , 5.0 , 60.5 , 2.1 , 15.5 , 30.0 ], # Tem um valor nulo aqui!
'Paginas_Vistas' : [ 5, 2, 1, 15 , 4, 2, 20 , 1, 5, 10 ],
'Dispositivo' : [ 'Mobile' , 'Mobile' , 'Desktop' , 'Desktop' , 'Mobile' ,
'Mobile' , 'Desktop' , 'Mobile' , 'Mobile' , 'Desktop' ],
'Carrinho_Abandonado' : [ 'Sim' , 'Não' , 'Não' , 'Sim' , 'Não' , 'Não' ,
'Sim' , 'Não' , 'Sim' , 'Sim' ],
'Comprou' : [ 1, 0, 0, 1, 0, 0, 1, 0, 1, 1]
}
# Criando nosso Dataset (DataFrame)
df = pd.DataFrame(dados_brutos)
print ( "Dataset Original com problemas:" )
print (df)
print ( " - " * 50 )

# ==========================================
# PARTE 1: PRÉ - PROCESSAMENTO (A FAXINA)
# ==========================================
# A) Removendo Duplicatas
# O sistema registrou alguns acessos duas vezes por erro. Remova as duplicatas!
df = df.drop_duplicates()

# B) Tratando Valores Nulos
# O cliente da linha 2 não teve o tempo registrado. Preencha o vazio com a MEDIANA da coluna.
mediana_tempo = df[ 'Tempo_Site' ].median()
df[ 'Tempo_Site' ] = df[ 'Tempo_Site' ]. fillna (mediana_tempo)

# C) Encoding (Transformando palavras em números)
# Modelos matemáticos não leem 'Mobile' ou 'Sim'. Transforme em 0 e 1.
# Para Dispositivo: 'Mobile' vira 0, 'Desktop' vira 1
df[ 'Dispositivo' ] = df[ 'Dispositivo' ].map({ 'Mobile' : __, 'Desktop' : __})
# Para Carrinho_Abandonado: 'Não' vira 0, 'Sim' vira 1
df[ 'Carrinho_Abandonado' ] = df[ 'Carrinho_Abandonado' ].map({ 'Não' : __,
'Sim' : __})
print ( " \ nDataset Limpo e Preparado para a IA:" )
print (df)
print ( " - " * 50 )

# ==========================================
# PARTE 2: TREINAMENTO DO MODELO
# ==========================================

# D) Separando Features e Label
# X = As características que o modelo vai usar para aprender
# y = A resposta que queremos prever
X = df[[ 'Tempo_Site' , 'Paginas_Vistas' , 'Dispositivo' ,
'Carrinho_Abandonado' ]]
y = df[ '____' ]

# E) Divisão Treino/Teste
# Separe 80% dos dados para o modelo estudar (treino) e 20% para a prova final (teste)
# Dica: use o test_size correto
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y,
test_size =____ , random_state =42 )

# F) Criando e Treinando a Árvore de Decisão
modelo = DecisionTreeClassifier( random_state =42 )
# Faça o modelo "aprender" com os dados de treino
modelo. ____ (X_treino, y_treino)

# ==========================================
# PARTE 3: AVALIAÇÃO RIGOROSA
# ==========================================

# G) Fazendo a "Prova Final"
# Peça para o modelo prever os resultados usando os dados de teste (que ele nunca viu)
previsoes = modelo. ____ (X_teste)

# H) Exibindo o Boletim do Modelo
print ( " \ n--- BOLETIM DE DESEMPENHO DA IA --- " )
print ( f "Acurácia: { accuracy_score(y_teste, previsoes) :.2f } " )
print ( f "Precisão: { precision_score(y_teste, previsoes,
zero_division =0) :.2f } " )
print ( f "Recall: { recall_score(y_teste, previsoes,
zero_division =0) :.2f } " )
print ( f "F1 - Score: { f1_score(y_teste, previsoes, zero_division =0) :.2f } " )
