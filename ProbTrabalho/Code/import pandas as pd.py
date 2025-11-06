import pandas as pd

# Supondo que o arquivo seja um CSV
brasileirao_partidas = pd.read_csv('ProbTrabalho\campeonato-brasileiro-full.csv')

#-------------------------------------------------
brasileirao_partidas.columns
#-------------------------------------------------
brasileirao_partidas['data'] = pd.to_datetime(brasileirao_partidas['data'], errors='coerce')

brasileirao_partidas = brasileirao_partidas[brasileirao_partidas['data'].dt.year >= 2015]

print(brasileirao_partidas)
#-------------------------------------------------
"Gostaria de ver quais são todas as formações táticas utilizadas para isso:"
formacoes_mandante = brasileirao_partidas['formacao_mandante'].unique()
formacoes_visitante = brasileirao_partidas['formacao_visitante'].unique()
print(formacoes_mandante)
print("-"*90)
print(formacoes_visitante)
#-------------------------------------------------
#Aqui junto as infos de mandante e visitante
formacoes_unicas = pd.unique(brasileirao_partidas[['formacao_mandante', 'formacao_visitante']].values.ravel())
print("Formações Únicas (Mandante e Visitante):", formacoes_unicas)
#-------------------------------------------------
print(brasileirao_partidas[['vencedor', 'mandante', 'visitante', 'formacao_mandante', 'formacao_visitante']].head())
#-------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

# Excluir empates (valores '-') na coluna 'vencedor'
vitorias = brasileirao_partidas[brasileirao_partidas['vencedor'] != '-'].copy()

# Determine the winning formation for each match that had a winner
def get_winning_formation(row):
    if row['vencedor'] == row['mandante']:
        return row['formacao_mandante']
    elif row['vencedor'] == row['visitante']:
        return row['formacao_visitante']
    return None

vitorias['winning_formation'] = vitorias.apply(get_winning_formation, axis=1)

# Count the number of wins by formation, excluding NaN values
vitorias_por_formacao = vitorias['winning_formation'].value_counts().dropna()

# Check the count of winning formations
print("Número de Vitórias por Formação:")
print(vitorias_por_formacao)


# Create the histogram
plt.figure(figsize=(12, 7))
vitorias_por_formacao.plot(kind='bar', color='skyblue')
plt.title('Número de Vitórias por Formação Tática', fontsize=16)
plt.xlabel('Formação Tática', fontsize=12)
plt.ylabel('Número de Vitórias', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
#-------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

# Excluir empates (valores '-') na coluna 'vencedor'
vitorias = brasileirao_partidas[brasileirao_partidas['vencedor'] != '-']

# Contar o número de vitórias por time
vitorias_por_time = vitorias['vencedor'].value_counts()

# Exibir as vitórias por time
print("Número de Vitórias por Time:")
print(vitorias_por_time)

# Criar o gráfico de barras
plt.figure(figsize=(12, 7))
vitorias_por_time.plot(kind='bar', color='skyblue')

# Título e rótulos
plt.title('Número de Vitórias por Time', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Número de Vitórias', fontsize=12)

# Ajustar o layout para não cortar os rótulos
plt.xticks(rotation=90, ha='right')

# Exibir o gráfico
plt.tight_layout()
plt.show()
#-------------------------------------------------

