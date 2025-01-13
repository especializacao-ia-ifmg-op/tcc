import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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
    
def to_tensor(data, features, target):
    X = torch.tensor(data[features].values, dtype=torch.float32)
    y = torch.tensor(data[target].values, dtype=torch.float32)
    return X, y    

class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, src):
        # src shape: (sequence_length, batch_size, input_dim)
        transformer_out = self.transformer(src, src)
        # Take the last output (for prediction)
        out = self.fc(transformer_out[-1, :, :])
        return out

def train_model(model, X_train, y_train, num_epochs, batch_size, optimizer, criterion):
    model.train()
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            output = model(batch_X.unsqueeze(1))  # reshape to (sequence_length, batch_size, input_dim)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(X_train)}')


def evaluate_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.unsqueeze(1)).squeeze()
        mse = criterion(predictions, y_test)
        return predictions, mse.item()

def add_lag_features(df, features, lag):
    for feature in features:
        for i in range(1, lag + 1):
            df[f'{feature}_lag_{i}'] = df[feature].shift(i)
    df.dropna(inplace=True)
    return df

def get_search_dataset_multivariate(dataset, lag=4, n_splits=5):
    df1 = pd.read_csv(dataset, sep=";")
    
    features = ['Rs', 'u2', 'Tmax', 'RH']
    target = 'ETo'

    df1 = add_lag_features(df1, features, lag)

    scaler = StandardScaler()
    scaled_features = [f'{feature}_lag_{i}' for feature in features for i in range(1, lag + 1)]
    df1[scaled_features] = scaler.fit_transform(df1[scaled_features])

    X = df1[scaled_features]
    y = df1[target]
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    input_dim = len(scaled_features)
    
    return X, y, tscv, input_dim

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
        r['Modelo'].iloc[i] = 'Transformer'+ t
    return r

def run_model(dataset_file_name, result_file_name, sufix, n_execucoes, n_previsoes):
    nhead = 2
    num_layers = 2
    hidden_dim = 64
    output_dim = 1
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32
    
    results = []
    
    X, y, tscv, input_dim = get_search_dataset_multivariate(dataset_file_name)
    
    for i in range(n_execucoes):
        rmse_scores = []
        model = TransformerModel(input_dim, nhead, num_layers, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            X_train, y_train = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32)
            X_test, y_test = torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)
            
            train_model(model, X_train, y_train, num_epochs, batch_size, optimizer, criterion)
            predictions, mse = evaluate_model(model, X_test, y_test, criterion)        
            rmse = np.sqrt(mse)
            rmse_scores.append(rmse)
            
        mean_rmse = np.mean(rmse_scores)
        results.append(mean_rmse)
        print(f"\n[{i + 1}-ésima execução] Média do RMSE em todos os splits: {mean_rmse:.6f}")

    results = form_data(results, sufix, n_execucoes, n_previsoes)
    results.to_csv(result_file_name,index=True)

    #del X_train_scaled
    del y_train
    #del X_test_scaled
    del y_test
    del model
    del results
    gc.collect()

# Running the model
n_var=5
n_execucoes=30
n_previsoes=1

bases_path = 'bases'

for dataset in os.listdir(bases_path):
    if dataset.endswith('.csv'):
        dataset_path = os.path.join(bases_path, dataset)
        print(f'[multi_some_Transformer.py]: base = {dataset_path}\n')
        print(f'Running...')
        
        start = time.time()
        run_model(dataset_file_name=str(dataset_path), result_file_name='./resultados/m_some_results_Transformer_'+str(n_execucoes)+'_'+str(n_previsoes)+'_'+str(dataset), sufix=' (Rs, u2, Tmax, RH, ETo)', n_execucoes=n_execucoes, n_previsoes=n_previsoes)
        stop = time.time()

        print(f'...done! Execution time = {stop - start}.')