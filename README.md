# Crypto Pump and Dump Detection Using Deep Learning Techniques
Cryptocurrency fraud detection, despite increased adoption, is inadequately explored,
particularly in pump and dump schemes. Existing studies on stock market
scams face limitations in applying to cryptocurrencies due to the absence of labeled
stock data and distinctive volatility. Current research primarily relies on statistical
or classical machine learning models. To address this gap, the proposal suggests
using two neural network architectures for pump and dump detection, showing
that deep learning outperforms existing methods. This underscores the necessity
for sophisticated approaches to combat fraud in the dynamic cryptocurrency
landscape.

## Code Structure
```
├── ...
├── data
│ ├── data.py (Dataset Handling Script)
│ ├── features_15S.csv.gz (Zipped 15 second Dataset)
│ ├── features_25S.csv.gz (Zipped 25 second Dataset)
│ ├── features_5S.csv.gz (Zipped 5 second Dataset)
|
|
|── models
| ├── anomaly_transformer.py (Anomaly Transformer Model Creation Script)
| ├── conv_lstm.py (Comvolution-LSTM Model Creation Script)
| ├── utils.py (Utility Functions Script)
|
|
|── PumpAndDumpDetection.ipynb (Pump And Dump Detection Notebook)
```

## Instructions to Run the Code
1) Run the first cell of 'PumpAndDumpDetection' Notebook. It will create the '/content/data/' and '/content/models/' folders.
2) Upload the datasets and the Data Handling Script in the '/content/data/' path.
3) Upload the model creation scripts and the utility script in the '/content/models/' path.
4) Run all the cells in the notebook.

