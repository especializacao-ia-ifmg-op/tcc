import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time
import gc
import os

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 7

def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def denormalize(norm, _min, _max):
    return [(n * (_max-_min)) + _min for n in norm]

def get_search_dataset_multivariate(dataset):
    df1 = pd.read_csv(dataset, sep=";")
    variaveis = ['Rs', 'u2', 'Tmax', 'Tmin', 'RH', 'pr', 'ETo']
    num_lags = 4

    X = df1[variaveis[:-1]]
    y = df1['ETo']

    for var in variaveis:
        for lag in range(1, num_lags + 1):
            X[f'{var}_lag{lag}'] = df1[var].shift(lag)

    X = X.dropna()
    y = y.loc[X.index]
    
    n_split = 5
    tscv = TimeSeriesSplit(n_split)
    
    return X, y, tscv

def form_data(data, t, n_execucoes, n_previsoes):
    df = pd.DataFrame(data)
    df1 = df.T
    frames = [df1.iloc[:,0], df1.iloc[:,1], df1.iloc[:,2], df1.iloc[:,3], df1.iloc[:,4], df1.iloc[:,5], df1.iloc[:,6], df1.iloc[:,7], df1.iloc[:,8], df1.iloc[:,9], df1.iloc[:,10], df1.iloc[:,11],
          df1.iloc[:,12], df1.iloc[:,13], df1.iloc[:,14], df1.iloc[:,15], df1.iloc[:,16], df1.iloc[:,17],df1.iloc[:,18], df1.iloc[:,19], df1.iloc[:,20], df1.iloc[:,21], df1.iloc[:,22], 
          df1.iloc[:,23], df1.iloc[:,24], df1.iloc[:,25], df1.iloc[:,26], df1.iloc[:,27], df1.iloc[:,28], df1.iloc[:,29]]
    result = pd.concat(frames)
    r = pd.DataFrame(result) 
    r.insert(1, "Modelo", True)
    for i in range(n_execucoes * n_previsoes): # n_execucoes * n_previsoes
        r['Modelo'].iloc[i] = 'RF'+ t
    return r

def run_model(dataset_file_name, result_file_name, sufix, n_execucoes, n_previsoes):
    results = []
    
    X, y, tscv = get_search_dataset_multivariate(dataset_file_name)
      
    for i in range(n_execucoes):
        rmse_scores = []
        model = RandomForestRegressor(n_estimators=100)#, random_state=42)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)
            
        mean_rmse = np.mean(rmse_scores)
        results.append(mean_rmse)
        print(f"\n[{i + 1}-ésima execução] Média do RMSE em todos os splits: {mean_rmse:.6f}")

    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name,index=True)

    gc.collect()

# Running the model
n_var=7
n_lags=4
n_execucoes=30
n_previsoes=1

caminho_bases = 'bases'

for dataset in os.listdir(caminho_bases):
    if dataset.endswith('.csv'):
        caminho_dataset = os.path.join(caminho_bases, dataset)
        print(f'[multi_all_RF.py]: base = {caminho_dataset}\n')
        print(f'Running...')
        
        start = time.time()
        run_model(dataset_file_name=str(caminho_dataset), result_file_name='./resultados/m_all_results_RF_'+str(n_execucoes)+'_'+str(n_previsoes)+'_'+str(dataset), sufix=' (Rs, u2, Tmax, Tmin, RH, pr, ETo)', n_execucoes=n_execucoes, n_previsoes=n_previsoes)
        stop = time.time()

        print(f'...done! Execution time = {stop - start}.')        