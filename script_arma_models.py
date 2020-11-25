# EXPLORATOIRE
# MODELISATION DE SERIES TEMPORELLES: ARMA, ARIMA, SARIMA

#########################################################################################################

# Variables à définir
    # Import et traitement des données
path = 'C:/Users/Dwimo/Documents/05 DATA/DataOCR/Machine Learning/Series temporelles/Data/' # dossier contenant les données
file = 'AirPassengers.csv' # nom du fichier contenant les données
sep = ',' # séparateur de colonnes
index = 0 # nom ou numero de colonne
is_day_first = True

# Variables pour la stationnarisation de la série
shift_1 = 1 # valeur pour 1e différenciation
shift_2 = 12 # valeur pour 2e différenciation, None par défaut

# Affichage des graphiques
decomposition = False
acf_on_raw_data = False
acf_on_stationnary_data = False

# Paramètres du modèle
# ARIMA parameters
p = 1 # valeur du premier PACF lag significatif
d = 0 # 0 si absence de tendance ou tous les ACF lags (données brutes) proches de 0, sinon 1
q = 1 # valeur du premier ACF lag significatif
# SARIMA parameters
P = 1 # >= 1 si l'ACF est positive au lag S, sinon 0
D = 1 # 1 si la série présente une saisonnalité stable, sinon 0
Q = 1 # >= 1 si l'ACF est négative au lag S, sinon 0
S = 12 # ACF lag de la valeur absolue la plus élevée
# d+D <= 2, P+Q <=2, p+d+q+P+D+Q <= 6

#########################################################################################################

# import des librairies
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
raw_data = pd.read_csv(path+file, sep=sep, index_col=index, parse_dates=[index], dayfirst=is_day_first)

# Décomposition de la série (série, tendance, saisonnalité, résidu)
if decomposition :
    decomp_x = seasonal_decompose(raw_data, model='multiplicative')
    decomp_x.plot()
    plt.show()

# Préparation des données
data = np.log(raw_data) # log des données pour réduire le taux d'augmentation
split_point = round(len(raw_data) * 0.7)
train_data, test_data = data[0:split_point], data[split_point:]

data_shift = data.diff(periods=shift_1)

if shift_2:
    data_shift = data_shift.diff(periods=shift_2)

data_shift = data_shift.dropna().values[1:]

# Affichage des graphiques ACF et PACF
def display_acf_pacf(data_acf, data_pacf=0):
    """Affichage des graphes ACF et PACF"""

    # ACF
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.bar(range(len(data_acf)), data_acf, width = 0.1)
    plt.xlabel('lag')
    plt.ylabel('ACF')
    plt.axhline(y=0, color='black')
    plt.axhline(y=-1.96/np.sqrt(len(data)), color='b', linestyle='--', linewidth=0.8)
    plt.axhline(y=1.96/np.sqrt(len(data)), color='b', linestyle='--', linewidth=0.8)
    plt.ylim(-1,1)

    # PACF
    if type(data_pacf) == np.ndarray :
        plt.subplot(122)
        plt.bar(range(len(data_pacf)), data_pacf, width = 0.1)
        plt.xlabel('lag')
        plt.ylabel('PACF')
        plt.axhline(y=0, color='black')
        plt.axhline(y=-1.96/np.sqrt(len(data)), color='b', linestyle='--', linewidth=0.8)
        plt.axhline(y=1.96/np.sqrt(len(data)), color='b', linestyle='--', linewidth=0.8)
        plt.ylim(-1,1)

    plt.show()

if acf_on_raw_data:
    data_acf = acf(data) # acf sur données brutes
    data_pacf = pacf(data)
    display_acf_pacf(data_acf)

if acf_on_stationnary_data:
    data_shift_acf = acf(data_shift) # acf sur données stationnarisées
    data_shift_pacf = pacf(data_shift)
    display_acf_pacf(data_shift_acf, data_shift_pacf)

# Modèle
model = SARIMAX(train_data, order=(p,d,q), seasonal_order=(P,D,Q,S), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

# Forecasting
k = len(test_data)
forecast = model_fit.forecast(k)
forecast = np.exp(forecast)

rmse = np.sqrt(sum((forecast-test_data['Airpass'])**2)/len(test_data))

plt.figure(figsize=(10,5))
plt.plot(forecast,'r')
plt.plot(raw_data,'black')
plt.title('RMSE: %.2f' % rmse)
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()