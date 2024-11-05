import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time
import gc
import os

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 7

# Function definitions
def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def denormalize(norm, _min, _max):
    return [(n * (_max-_min)) + _min for n in norm]

def get_search_dataset_multivariate(dataset, n_var):
    df1 = pd.read_csv(dataset, sep=";")
    
    ints = df1.select_dtypes(include=['int64','int32','int16']).columns
    df1[ints] = df1[ints].apply(pd.to_numeric, downcast='integer')
    floats = df1.select_dtypes(include=['float']).columns
    df1[floats] = df1[floats].apply(pd.to_numeric, downcast='float')
    
    series = df1.iloc[:,1:n_var+1]
    norm_df = normalize(series)
    size = int(len(norm_df) * 0.80)
    train, test = norm_df[0:size], norm_df[size:len(norm_df)]
    
    return train, test

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
        r['Modelo'].iloc[i] = 'VAR'+ t
    return r

def run_model(dataset_file_name, result_file_name, sufix, n_var, n_execucoes, n_previsoes):        
    train, test = get_search_dataset_multivariate(dataset_file_name, n_var)    
    
    history = [x for x in train]
    predictions = []
    results = []
    
    for i in range(n_execucoes):
        order = 4
        for t in range(len(test)):
            model = VAR(train.values)
            model_fit = model.fit(maxlags=order)
            lag_order = model_fit.k_ar
            fc = model_fit.forecast(y=test[:t+4].values, steps=1)
            output = fc[0][4]
            yhat = output
            predictions.append(yhat)
            obs = test['ETo'].iloc[t]
            history.append(obs)
        
        rmse = np.sqrt(mean_squared_error(test['ETo'], predictions))
        predictions = []
        results.append(rmse)

    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name,index=True)

    del train
    del test
    del model
    del results
    gc.collect()

# Running the model
n_var=5
n_execucoes=30
n_previsoes=1

caminho_bases = 'bases/importance'

for dataset in os.listdir(caminho_bases):
    if dataset.endswith('.csv'):
        caminho_dataset = os.path.join(caminho_bases, dataset)
        print(f'[multi_some_VAR.py]: base = {caminho_dataset}\n')
        print(f'Running...')
        
        start = time.time()
        run_model(dataset_file_name=str(caminho_dataset), result_file_name='./resultados/m_some_results_VAR_'+str(n_execucoes)+'_'+str(n_previsoes)+'_'+str(dataset), sufix=' (Rs, u2, Tmax, RH, ETo)', n_var =n_var, n_execucoes=n_execucoes, n_previsoes=n_previsoes)
        stop = time.time()

        print(f'...done! Execution time = {stop - start}.')