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
model_vs_data = True

#########################################################################################################

# import des librairies
from numpy.core.arrayprint import DatetimeFormat
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
import itertools
import warnings
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
split_point = round(len(raw_data) * 0.7)
train_data, test_data = raw_data[0:split_point], raw_data[split_point:]

train_data = np.log(train_data) # log des données pour réduire le taux d'augmentation

data_shift = train_data.diff(periods=shift_1)

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

# Optimisation des paramètres du modèle
def grid_search_sarima_param(data, S = shift_2, print_params = False):
    """ Grid search for SARIMA optimal pdq and 
    seasonal PDQ parameters """

    S = S
    p = d = q = range(0,2)
    pdq = list(itertools.product(p,d,q))
    seasonal_PDQ = [(x[0], x[1], x[2], S) for x in pdq]

    warnings.filterwarnings("ignore") # specify to ignore warning messages

    min_rmse = 10000
    
    for param in pdq:
        for param_seasonal in seasonal_PDQ:
            try:
                model = SARIMAX(data,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)

                results = model.fit()

                k = len(test_data)
                forecast = results.forecast(k)
                forecast = np.exp(forecast)

                rmse = np.sqrt(sum((forecast-test_data['Airpass'])**2)/len(test_data))

                if rmse < min_rmse :
                    min_rmse = round(rmse,2)
                    optimal_aic = round(results.aic,2)
                    optimal_pdq = param
                    optimal_seasonal_pdq = param_seasonal

            except:
                continue

    if print_params:
        print('SARIMA{}x{} - AIC:{} - RMSE:{}'.format(optimal_pdq, optimal_seasonal_pdq, optimal_aic, min_rmse))

    return optimal_pdq, optimal_seasonal_pdq, optimal_aic, min_rmse

optimal_params = grid_search_sarima_param(train_data, print_params=True)

# Modèle
model = SARIMAX(train_data, order=optimal_params[0], seasonal_order=optimal_params[1], enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

# Forecasting
k = len(test_data)
forecast = model_fit.forecast(k)
forecast = np.exp(forecast)

def display_model_vs_raw(title='Model VS raw data'):
    rmse = optimal_params[3]
    aic = optimal_params[2]

    plt.figure(figsize=(10,5))
    plt.title(title, fontsize=12)
    plt.plot(forecast,'r')
    plt.plot(raw_data,'black')
    plt.text(max(raw_data.index), 130, 'RMSE : {} '.format(rmse), horizontalalignment='right', size=12)
    plt.text(max(raw_data.index), 90, 'AIC : {} '.format(aic), horizontalalignment='right', size=12)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylabel(raw_data.columns[0], fontsize=14)
    plt.show()

if model_vs_data :
    display_model_vs_raw()

