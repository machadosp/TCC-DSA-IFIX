# Remover espaços extras dos nomes das colunas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np

# Carregar os dados
data_clean = pd.read_csv('C:\\Users\\wmach\\OneDrive\\�rea de Trabalho\\IFIX_MACRO.csv', delimiter=';', encoding='utf-8-sig')

# Remover espaços extras dos nomes das colunas
data_clean.columns = data_clean.columns.str.strip()

# Função para converter valores
def convert_to_float_manual(x):
    try:
        return float(str(x).replace(',', '.'))
    except ValueError:
        return None

# Converter todas as colunas, exceto a de data, para float
for col in data_clean.columns[1:]:
    data_clean[col] = data_clean[col].apply(convert_to_float_manual)

# Tratar valores ausentes
data_clean.dropna(inplace=True)

# Transformar a coluna de data em índice de data
data_clean['DATA'] = pd.to_datetime(data_clean['DATA'], format='%d/%m/%Y')
data_clean.set_index('DATA', inplace=True)

# Padronizar as variáveis explicativas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_clean.drop(columns=['IFIX']))

# Aplicar o LASSO com validação cruzada
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, data_clean['IFIX'])

# Coeficientes do LASSO
lasso_coef = pd.Series(lasso.coef_, index=data_clean.columns[1:])
selected_variables = lasso_coef[lasso_coef != 0].index

# Dados com as variáveis selecionadas
X_selected = data_clean[selected_variables]

# Dividir em conjuntos de treinamento e teste
train_size = int(len(data_clean) * 0.8)
train_data, test_data = X_selected[:train_size], X_selected[train_size:]
train_ifix, test_ifix = data_clean['IFIX'][:train_size], data_clean['IFIX'][train_size:]

# Ajustar o modelo ARIMA com as variáveis selecionadas
arima_order = (1, 1, 1)
arima_model = ARIMA(train_ifix, exog=train_data, order=arima_order)
arima_result = arima_model.fit()

# Prever no conjunto de teste
arima_forecast = arima_result.get_forecast(steps=len(test_ifix), exog=test_data)
arima_forecast_values = arima_forecast.predicted_mean

# Gráfico 1: Evolução do IFIX ao Longo do Tempo
plt.figure(figsize=(14, 7))
plt.plot(data_clean.index, data_clean['IFIX'], label='IFIX', color='blue')
plt.title('Evolução do IFIX ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('IFIX')
plt.legend()
plt.grid(True)
plt.savefig('evolucao_ifix.png')
plt.show()

# Previsões do modelo ARIMA
train_data['IFIX_PRED'] = arima_result.fittedvalues
test_data['IFIX_PRED'] = arima_forecast_values

# Remover o primeiro ponto do plot da linha IFIX Previsto pelo ARIMA
combined_data = pd.concat([train_data[['IFIX_PRED']], test_data[['IFIX_PRED']]], axis=0)
combined_data['IFIX'] = data_clean['IFIX']
combined_data['IFIX_PRED'] = combined_data['IFIX_PRED'].shift(-1)

# Gráfico 2: Comparação da Série Temporal do IFIX com o Resultado do Modelo ARIMA
plt.figure(figsize=(14, 7))
plt.plot(combined_data.index, combined_data['IFIX'], label='IFIX Real', color='blue')
plt.plot(combined_data.index, combined_data['IFIX_PRED'], label='IFIX Previsto pelo ARIMA', color='red')
plt.title('Comparação da Série Temporal do IFIX com o Resultado do Modelo ARIMA')
plt.xlabel('Data')
plt.ylabel('IFIX')
plt.legend()
plt.grid(True)
plt.savefig('comparacao_ifix_arima.png')
plt.show()

# Gráficos ACF e PACF para a série temporal do IFIX
fig, axes = plt.subplots(1, 2, figsize=(15, 5))


sm.graphics.tsa.plot_acf(combined_data['IFIX'], lags=39, ax=axes[0], title='Autocorrela��o')
sm.graphics.tsa.plot_pacf(combined_data['IFIX'], lags=19, ax=axes[1], title='Autocorrela��o Parcial')

formatter = ticker.FuncFormatter(lambda x, _: f'{x:.2f}'.replace('.', ','))

for ax in axes:
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
plt.savefig('acf_pacf_ifix.png')
plt.show()

# Calcular MSE para os conjuntos de treinamento e teste
mse_train = mean_squared_error(train_ifix, arima_result.fittedvalues)
mse_test = mean_squared_error(test_ifix, arima_forecast_values)

print("MSE no Conjunto de Treinamento:", mse_train)
print("MSE no Conjunto de Teste:", mse_test)
