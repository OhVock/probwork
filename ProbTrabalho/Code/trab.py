# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, expon, poisson
from scipy.optimize import minimize
import scipy.stats as stats

# Carregar o dataset
df = pd.read_csv("ProbTrabalho\campeonato-brasileiro-full.csv")

# Converter a coluna 'data' para datetime
df['data'] = pd.to_datetime(df['data'], errors='coerce')

# Filtrar para manter apenas os registros de 2015 em diante
df_filtrado = df[df['data'].dt.year >= 2015].copy()

# Calcular as vitórias por time
vitorias_por_time = (df_filtrado['vencedor']
                     .value_counts(dropna=True)
                     .drop(labels='-', errors='ignore')
                    )

# Recalcular as estatísticas descritivas da variável de vitórias por time
estatisticas_vitorias = vitorias_por_time.describe()

# Calcular moda (valor mais frequente) para vitórias por time
moda_vitorias = vitorias_por_time.mode()[0]

# Calcular o desvio padrão
desvio_padrao_vitorias = vitorias_por_time.std()

# Exibindo as estatísticas descritivas e a moda
print(estatisticas_vitorias)
print(f"Moda das Vitórias: {moda_vitorias}")
print(f"Desvio Padrão das Vitórias: {desvio_padrao_vitorias}")

# Converter os times para variáveis dummy (código numérico)
dummy_map = {time: i for i, time in enumerate(vitorias_por_time.index)}

# Criar uma série com os códigos dummy e associar ao número de vitórias
vitorias_dummy = vitorias_por_time.rename(index=dummy_map)

# Gerar a legenda para os códigos dummy (time -> código)
dummy_legend = {i: time for time, i in dummy_map.items()}

# Exibindo os primeiros códigos dummy com a legenda associada
print("Legenda Dummy:", dummy_legend)

# Plot 1 – Histograma das vitórias por time (com mais detalhes)
plt.figure(figsize=(12, 6))
plt.bar(vitorias_dummy.index, vitorias_dummy.values, color='steelblue')
plt.title("Vitórias por Time (2015–presente)")
plt.xlabel("Time (Ordenado por vitórias)")
plt.ylabel("Quantidade de Vitórias")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=90)  # Rotacionar os nomes dos times para melhor visualização
plt.show()

# Plot 2 – Boxplot das vitórias por time
plt.figure(figsize=(10, 5))
sns.boxplot(x=vitorias_dummy.values, color='skyblue')
plt.title("Boxplot das Vitórias por Time (2015–presente)")
plt.xlabel("Quantidade de Vitórias")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# Plotando o histograma com KDE
plt.figure(figsize=(14, 8))
sns.histplot(vitorias_dummy, kde=True, bins=30, color='steelblue', stat='density', linewidth=2)
plt.title("Histograma com Kernel Density Estimate (KDE) das Vitórias por Time (2015–presente)", fontsize=14)
plt.xlabel("Número de Vitórias", fontsize=12)
plt.ylabel("Densidade de Probabilidade", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Estimativa de parâmetros para a distribuição Normal
mu_normal, std_normal = norm.fit(vitorias_dummy)

# Estimativa de parâmetros para a distribuição Exponencial
lambda_expo = 1 / np.mean(vitorias_dummy)

# Estimativa de parâmetros para a distribuição Poisson
lambda_poisson = np.mean(vitorias_dummy)

# Plotando os dados e os ajustes das distribuições
plt.figure(figsize=(14, 8))

# Histograma dos dados observados
plt.hist(vitorias_dummy, bins=20, density=True, alpha=0.6, color='steelblue', label="Dados Observados")

# Ajuste da distribuição Normal
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p_normal = norm.pdf(x, mu_normal, std_normal)
plt.plot(x, p_normal, 'k', linewidth=2, label="Ajuste Normal")

# Ajuste da distribuição Exponencial
p_expo = expon.pdf(x, scale=1/lambda_expo)
plt.plot(x, p_expo, 'r', linewidth=2, label="Ajuste Exponencial")

# Ajuste da distribuição Poisson (como uma distribuição contínua para visualização)
p_poisson = poisson.pmf(np.floor(x), lambda_poisson)
plt.plot(np.floor(x), p_poisson, 'g', linewidth=2, label="Ajuste Poisson")

# Títulos e legenda
plt.title("Comparação das Distribuições Ajustadas aos Dados de Vitórias por Time (2015–presente)")
plt.xlabel("Número de Vitórias")
plt.ylabel("Densidade de Probabilidade")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Método de Momentos (MM) para as distribuições
mu_mm = np.mean(vitorias_dummy)
std_mm = np.std(vitorias_dummy)

lambda_mm = 1 / np.mean(vitorias_dummy)
lambda_poisson_mm = np.mean(vitorias_dummy)

print(f"Parâmetros estimados pelo Método de Momentos (MM):\n"
      f"Normal -> Média: {mu_mm}, Desvio Padrão: {std_mm}\n"
      f"Exponencial -> λ: {lambda_mm}\n"
      f"Poisson -> λ: {lambda_poisson_mm}")

# Função de verossimilhança para a distribuição Normal
def log_likelihood_normal(params, data):
    mu, std = params
    return -np.sum(np.log(norm.pdf(data, mu, std)))

# Função de verossimilhança para a distribuição Exponencial
def log_likelihood_expo(params, data):
    lambda_expo = params[0]
    return -np.sum(np.log(expon.pdf(data, scale=1/lambda_expo)))

# Função de verossimilhança para a distribuição Poisson
def log_likelihood_poisson(params, data):
    lambda_poisson = params[0]
    return -np.sum(np.log(poisson.pmf(data, lambda_poisson)))

# Estimativa de parâmetros pelo Método de Máxima Verossimilhança (MLE) para a Normal
params_normal_initial = [mu_mm, std_mm]
params_normal_mle = minimize(log_likelihood_normal, params_normal_initial, args=(vitorias_dummy,))
mu_mle, std_mle = params_normal_mle.x

# Estimativa de parâmetros pelo MLE para a Exponencial
params_expo_initial = [lambda_mm]
params_expo_mle = minimize(log_likelihood_expo, params_expo_initial, args=(vitorias_dummy,))
lambda_mle = params_expo_mle.x[0]

# Estimativa de parâmetros pelo MLE para a Poisson
params_poisson_initial = [lambda_poisson_mm]
params_poisson_mle = minimize(log_likelihood_poisson, params_poisson_initial, args=(vitorias_dummy,))
lambda_poisson_mle = params_poisson_mle.x[0]

# Exibir os parâmetros estimados pelo MLE
print(f"Parâmetros estimados pelo Método de Máxima Verossimilhança (MLE):\n"
      f"Normal -> Média: {mu_mle}, Desvio Padrão: {std_mle}\n"
      f"Exponencial -> λ: {lambda_mle}\n"
      f"Poisson -> λ: {lambda_poisson_mle}")

# Gerando QQ-Plots para as distribuições ajustadas
def qq_plot(data, dist, params, dist_name):
    plt.figure(figsize=(10, 6))
    stats.probplot(data, dist=dist, sparams=params, plot=plt)
    plt.title(f"QQ-Plot: Comparação dos Dados com a Distribuição {dist_name}", fontsize=14)
    plt.grid(True)
    plt.show()

# QQ-Plot para a distribuição Normal
qq_plot(vitorias_dummy, 'norm', (mu_mle, std_mle), 'Normal')

# QQ-Plot para a distribuição Exponencial
qq_plot(vitorias_dummy, 'expon', (1/lambda_mle,), 'Exponencial')

# QQ-Plot para a distribuição Poisson
qq_plot(vitorias_dummy, 'poisson', (lambda_poisson_mle,), 'Poisson')

def log_likelihood_normal(params, data):
    mu, std = params
    return -np.sum(np.log(norm.pdf(data, mu, std)))

def log_likelihood_expo(params, data):
    lambda_expo = params[0]
    return -np.sum(np.log(expon.pdf(data, scale=1/lambda_expo)))

def log_likelihood_poisson(params, data):
    lambda_poisson = params[0]
    return -np.sum(np.log(poisson.pmf(data, lambda_poisson)))

# Carregar o arquivo CSV (ajuste o caminho para o seu arquivo)
df = pd.read_csv("ProbTrabalho\campeonato-brasileiro-full.csv")

# Converter a coluna 'data' para datetime e filtrar os dados de 2015 em diante
df['data'] = pd.to_datetime(df['data'], errors='coerce')
df_filtrado = df[df['data'].dt.year >= 2015].copy()

# Calcular as vitórias por time
vitorias_por_time = (df_filtrado['vencedor']
                     .value_counts(dropna=True)
                     .drop(labels='-', errors='ignore')
                    )

# Estimar os parâmetros das distribuições Normal, Exponencial e Poisson
mu_normal, std_normal = norm.fit(vitorias_por_time)
lambda_expo = 1 / np.mean(vitorias_por_time)
lambda_poisson = np.mean(vitorias_por_time)

# Estimativas de parâmetros por MLE
params_normal_initial = [mu_normal, std_normal]
params_normal_mle = minimize(log_likelihood_normal, params_normal_initial, args=(vitorias_por_time,))
mu_mle, std_mle = params_normal_mle.x

params_expo_initial = [lambda_expo]
params_expo_mle = minimize(log_likelihood_expo, params_expo_initial, args=(vitorias_por_time,))
lambda_mle = params_expo_mle.x[0]

params_poisson_initial = [lambda_poisson]
params_poisson_mle = minimize(log_likelihood_poisson, params_poisson_initial, args=(vitorias_por_time,))
lambda_poisson_mle = params_poisson_mle.x[0]

# Cálculo dos resíduos padronizados para os ajustes
residuos_normal = (vitorias_por_time - mu_mle) / std_mle
residuos_expo = (vitorias_por_time - lambda_mle) / (1/lambda_mle)
residuos_poisson = (vitorias_por_time - lambda_poisson_mle) / lambda_poisson_mle

# Função para plotar histogramas e QQ-plots dos resíduos padronizados
def plot_residuos(residuos, dist_name, color='skyblue'):
    # Histogramas dos resíduos padronizados
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    sns.histplot(residuos, kde=True, color=color, bins=20, stat='density')
    plt.title(f"Histograma dos Resíduos Padronizados - {dist_name}")

    # QQ-Plot dos resíduos padronizados
    plt.subplot(2, 2, 2)
    stats.probplot(residuos, dist="norm", plot=plt)
    plt.title(f"QQ-Plot dos Resíduos Padronizados - {dist_name}")

    plt.tight_layout()
    plt.show()

# Plotando os resíduos padronizados para a distribuição Normal
plot_residuos(residuos_normal, "Normal", color='lightcoral')

# Plotando os resíduos padronizados para a distribuição Exponencial
plot_residuos(residuos_expo, "Exponencial", color='lightgreen')

# Plotando os resíduos padronizados para a distribuição Poisson
plot_residuos(residuos_poisson, "Poisson", color='lightblue')

# Importar as bibliotecas necessária
# Supondo que o dataframe 'df' já tenha sido carregado e os dados de vitórias já estejam disponíveis em 'vitorias_por_time'

# Calcular os quartis (Q1 e Q3) e o intervalo interquartil (IQR) para remover os outliers
Q1 = vitorias_por_time.quantile(0.25)
Q3 = vitorias_por_time.quantile(0.75)
IQR = Q3 - Q1

# Calcular os limites para os outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Identificar os outliers
outliers = vitorias_por_time[(vitorias_por_time < limite_inferior) | (vitorias_por_time > limite_superior)]

# Remover os outliers
vitorias_sem_outliers = vitorias_por_time[~vitorias_por_time.index.isin(outliers.index)]

# Gerar os gráficos comparativos com KDE, com e sem outliers
plt.figure(figsize=(14, 7))

# Gráfico com outliers
plt.subplot(1, 2, 1)
sns.histplot(vitorias_por_time, kde=True, color='steelblue', bins=30, stat='density', linewidth=2)
plt.title("Histograma com Kernel Density Estimate (KDE) - Com Outliers")
plt.xlabel("Número de Vitórias")
plt.ylabel("Densidade de Probabilidade")

# Gráfico sem outliers (considerando apenas os dados sem outliers)
plt.subplot(1, 2, 2)
sns.histplot(vitorias_sem_outliers, kde=True, color='lightgreen', bins=30, stat='density', linewidth=2)
plt.title("Histograma com Kernel Density Estimate (KDE) - Sem Outliers")
plt.xlabel("Número de Vitórias")
plt.ylabel("Densidade de Probabilidade")

plt.tight_layout()
plt.show()

Q1 = vitorias_por_time.quantile(0.25)
Q3 = vitorias_por_time.quantile(0.75)
IQR = Q3 - Q1

# Calcular os limites para os outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Identificar os outliers
outliers = vitorias_por_time[(vitorias_por_time < limite_inferior) | (vitorias_por_time > limite_superior)]

# Remover os outliers
vitorias_sem_outliers = vitorias_por_time[~vitorias_por_time.index.isin(outliers.index)]

# Estimativas de parâmetros para as distribuições Normal, Exponencial e Poisson
mu_normal, std_normal = norm.fit(vitorias_por_time)
lambda_expo = 1 / np.mean(vitorias_por_time)
lambda_poisson = np.mean(vitorias_por_time)

# Gerar os parâmetros de ajuste para cada distribuição
x = np.linspace(0, 200, 1000)

# Ajuste da distribuição normal para dados com outliers
p_normal = norm.pdf(x, mu_normal, std_normal)

# Ajuste da distribuição exponencial para dados com outliers
p_expo = expon.pdf(x, scale=1/lambda_expo)

# Ajuste da distribuição Poisson para dados com outliers
p_poisson = poisson.pmf(np.floor(x), lambda_poisson)

# Ajuste das distribuições para dados sem outliers
mu_normal_sem_outliers, std_normal_sem_outliers = norm.fit(vitorias_sem_outliers)
p_normal_sem_outliers = norm.pdf(x, mu_normal_sem_outliers, std_normal_sem_outliers)

lambda_expo_sem_outliers = 1 / np.mean(vitorias_sem_outliers)
p_expo_sem_outliers = expon.pdf(x, scale=1/lambda_expo_sem_outliers)

lambda_poisson_sem_outliers = np.mean(vitorias_sem_outliers)
p_poisson_sem_outliers = poisson.pmf(np.floor(x), lambda_poisson_sem_outliers)

# Gerar o gráfico comparativo com os dados com outliers e sem outliers
plt.figure(figsize=(14, 8))

# Gráfico com outliers
plt.subplot(1, 2, 1)
plt.hist(vitorias_por_time, bins=20, density=True, alpha=0.6, color='steelblue', label="Dados Observados")
plt.plot(x, p_normal, 'k', linewidth=2, label="Ajuste Normal")
plt.plot(x, p_expo, 'r', linewidth=2, label="Ajuste Exponencial")
plt.plot(np.floor(x), p_poisson, 'g', linewidth=2, label="Ajuste Poisson")
plt.title("Comparação das Distribuições Ajustadas aos Dados de Vitórias por Time (Com Outliers)", fontsize=14)
plt.xlabel("Número de Vitórias", fontsize=12)
plt.ylabel("Densidade de Probabilidade", fontsize=12)
plt.legend(loc="best")
plt.grid(True)

# Gráfico sem outliers
plt.subplot(1, 2, 2)
plt.hist(vitorias_sem_outliers, bins=20, density=True, alpha=0.6, color='lightgreen', label="Dados Observados")
plt.plot(x, p_normal_sem_outliers, 'k', linewidth=2, label="Ajuste Normal")
plt.plot(x, p_expo_sem_outliers, 'r', linewidth=2, label="Ajuste Exponencial")
plt.plot(np.floor(x), p_poisson_sem_outliers, 'g', linewidth=2, label="Ajuste Poisson")
plt.title("Comparação das Distribuições Ajustadas aos Dados de Vitórias por Time (Sem Outliers)", fontsize=14)
plt.xlabel("Número de Vitórias", fontsize=12)
plt.ylabel("Densidade de Probabilidade", fontsize=12)
plt.legend(loc="best")
plt.grid(True)

# Exibir os gráficos
plt.tight_layout()
plt.show()