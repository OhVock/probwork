import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv("ProbTrabalho\\campeonato-brasileiro-full.csv")

# Converter a coluna 'data' para datetime
df['data'] = pd.to_datetime(df['data'], errors='coerce')

# Filtrar os dados de 2015 em diante
df_filtrado = df[df['data'].dt.year >= 2015].copy()

# Contar o número de vitórias por formação vencedora do mandante
vitorias_por_formacao_mandante = df_filtrado[df_filtrado['vencedor'] == df_filtrado['mandante']]['formacao_mandante'].value_counts(dropna=True)

# Contar o número de vitórias por formação vencedora do visitante
vitorias_por_formacao_visitante = df_filtrado[df_filtrado['vencedor'] == df_filtrado['visitante']]['formacao_visitante'].value_counts(dropna=True)

# Somar as vitórias das duas colunas de formação (mandante e visitante)
vitorias_por_formacao_total = vitorias_por_formacao_mandante.add(vitorias_por_formacao_visitante, fill_value=0)

# Contar o número de vezes que cada formação foi utilizada pelo mandante
formacoes_mandante = df_filtrado['formacao_mandante'].value_counts(dropna=True)

# Contar o número de vezes que cada formação foi utilizada pelo visitante
formacoes_visitante = df_filtrado['formacao_visitante'].value_counts(dropna=True)

# Somar as contagens das duas colunas de formação (mandante e visitante)
formacoes_utilizadas_total = formacoes_mandante.add(formacoes_visitante, fill_value=0)

# Filtrar as formações que têm pelo menos 38 partidas jogadas
formacoes_utilizadas_total_filtradas = formacoes_utilizadas_total[formacoes_utilizadas_total >= 38]

# Calcular a porcentagem de vitórias para cada formação, apenas para as formações filtradas
porcentagem_vitorias = (vitorias_por_formacao_total.loc[formacoes_utilizadas_total_filtradas.index] / formacoes_utilizadas_total_filtradas) * 100

# Organizar as porcentagens de vitórias da maior para a menor
porcentagem_vitorias_sorted = porcentagem_vitorias.sort_values(ascending=False)

# Exibir a porcentagem de vitórias de cada formação, organizada
print(porcentagem_vitorias_sorted)
