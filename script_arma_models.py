# EN COURS
# MODELISATION DE SERIES TEMPORELLES: ARMA, ARIMA, SARIMA

#########################################################################################################

# Variables à définir
    # Import et traitement des données
path = 'C:/Users/Dwimo/Documents/05 DATA/DataOCR/Machine Learning/Series temporelles/Data/' # dossier contenant les données
file = 'AirPassengers.csv' # nom du fichier contenant les données
sep = ',' # séparateur de colonnes
index = 0 # nom ou numero de colonne
is_day_first = True


#########################################################################################################

# import des librairies
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
raw_data = pd.read_csv(path+file, sep=sep, index_col=index, parse_dates=[index], dayfirst=is_day_first)

# Décomposition de la série (série, tendance, saisonnalité, résidu)
decomp_x = seasonal_decompose(raw_data, model='multiplicative')
decomp_x.plot()
plt.show()

# Préparation des données
data = np.log(raw_data) # log des données pour réduire le taux d'augmentation
data_shift = data.diff(periods=1).values[1:]

# ACF et PACF
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

data_acf = acf(data) # acf sur données brutes
data_pacf = pacf(data)
display_acf_pacf(data_acf)

data_shift_acf = acf(data_shift) # acf sur données différenciées
data_shift_pacf = pacf(data_shift)
display_acf_pacf(data_shift_acf, data_shift_pacf)

