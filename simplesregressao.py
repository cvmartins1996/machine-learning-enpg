# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:48:24 2017

@author: Caio Vinicius, Lyncon Rodrigo, Felipe Vieira, Yuri Cesar, Phillipe Wagner
Roberto Henrique, Jack Leal
"""

# Simple Linear Regression

# Importar as libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar os datasets

dataset = pd.read_csv('cursoxanosexp.csv')
# variavel independente 
X = dataset.iloc[:, :-1].values
# virgula para pegar todas as colunas
# -1 exceto a ultima coluna

# variavel dependente 
y = dataset.iloc[:, 1].values
# dados preparados

# Dividir o dataset em Training set e Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Prepara Simples Regressão Linear para o Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prever os resultados do Test set
y_pred = regressor.predict(X_test)

# Visualizar os resultados do Training set

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Curso vs Experiencia na Area (Training set)')
plt.xlabel('Meses de Curso')
plt.ylabel('Meses de Experiencia')
plt.show()

# Visualizar os resultados do Test set

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'purple')
plt.title('Curso vs Experiencia na Area(Test set)')
plt.xlabel('Meses de Curso')
plt.ylabel('Meses de Experiencia')
plt.show()
