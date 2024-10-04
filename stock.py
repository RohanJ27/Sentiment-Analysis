
import inline as inline
import matplotlib
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import metrics


ticker= input("What stock do you want to analyze?: ")
start=input("From what date?: ")
end=input("To what date?: ")
stock_data = yf.download(ticker, start, end)
print(stock_data.head())

stock_data['Adj Close'].plot()
plt.ylabel("Adjusted Close Prices")
plt.show()

stock_data['Adj Close'].plot()
plt.ylabel("Adjusted Close Prices")
plt.show()