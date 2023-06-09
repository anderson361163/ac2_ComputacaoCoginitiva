# -*- coding: utf-8 -*-
"""Ac2_Cognitiva.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yzpcJEcEPr7-YZZxXcJIDTl3XbYXu3N_

# AC2 - RA.: 1902142 e RA.: 1901940
"""

#Importações

import numpy as np
import pandas as pd

#Importar o Arquivo
dataset = pd.read_csv('/content/drive/MyDrive/IMPACTA/Comp Cognitiva/Case_cobranca.csv') 

# Ponto de Parada 1  -  Olhar variável dataset no explorador de variáveis
dataset.columns

"""Criando o Alvo"""

#Criando alvo do dataset
dataset['ALVO']   = [0 if np.isnan(x) or x > 90 else 1 for x in dataset['TEMP_RECUPERACAO']]

"""Criando as colunas de pré-processamento"""

# Ponto de Parada 2  -  Olhar nova coluna 'ALVO' criada na variável dataset

## Tratamento de nulos e normalização --- Variáveis de entrada numéricas
#dataset['PRE_IDADE']        = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] # Trata mínimo
dataset['PRE_IDADE']        = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] # Trata mínimo
dataset['PRE_IDADE']        = [1. if x > 76 else (x-18)/(76-18) for x in dataset['PRE_IDADE']] # Trata máximo por percentil 99 e coloca na fórmula
##--- Dummies - transformação de atributos categóricos em numéricos e tratamanto de nulos ---------------
dataset['PRE_NOVO']         = [1 if x=='NOVO'                      else 0 for x in dataset['TIPO_CLIENTE']]   
dataset['INVESTIDOR']       = [1 if x=='INVESTIDOR'                else 0 for x in dataset['TIPO_CLIENTE']]   
dataset['PRE_TOMADOR_VAZIO']= [1 if x=='TOMADOR' or str(x)=='nan'  else 0 for x in dataset['TIPO_CLIENTE']]  
dataset['PRE_CDC']          = [1 if x=='CDC'                       else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_PESSOAL']      = [1 if x=='PESSOAL'                   else 0 for x in dataset['TIPO_EMPRESTIMO']]
#dataset['CD_SEXO'].fillna('M', inplace=True)
dataset['PRE_SEXO_M']       = [1 if x=='M'                         else 0 for x in dataset['CD_SEXO']]
dataset['PRE_SEXO_F']       = [1 if x=='F'                         else 0 for x in dataset['CD_SEXO']]
dataset['PRE_SEXO_I']        = [1 if x == np.NaN else 0 for x in dataset['CD_SEXO']]


dataset['PRE_SEXO_TESTE']        = [0 if not(x=='F' or x=='M') else x for x in dataset['CD_SEXO']]
dataset['PRE_SEXO_TESTE']

"""criando as colunas que serão utilizadas para analise"""

# Ponto de Parada 3  -  Olhar as novas colunas criadas na variável dataset

##------------------------------------------------------------
## Separando em dados de treinamento e teste
##------------------------------------------------------------
y = dataset['ALVO']              # Carrega alvo ou dataset.iloc[:,7].values
X = dataset.iloc[:, 8:15].values # Carrega colunas 8, 9, 10, 11, 12, 13 e 14 (a 15 não existe até este momento)
X

"""separando os dados para treinamento e analise"""

#
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 142)

"""Relizando o treinamento em diferentes algoritmos"""

#---------------------------------------------------------------------------
## Ajustando modelos - Aprendizado supervisionado  
#---------------------------------------------------------------------------
# Árvore de decisão com dados de treinamento
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtree.fit(X_train, y_train)

# Regressão linear com dados de treinamento
from sklearn.linear_model import LinearRegression
LinearReg = LinearRegression()
LinearReg.fit(X_train, y_train)

# Regressão logística com dados de treinamento
from sklearn.linear_model import LogisticRegression
LogisticReg = LogisticRegression()
LogisticReg.fit(X_train, y_train)

#Rede Neural com dados de treinamento
from sklearn.neural_network import MLPClassifier 
RNA = MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True,
       epsilon=1e-08, hidden_layer_sizes=(10,20), learning_rate='constant',
       learning_rate_init=0.001, max_iter=2000000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.3, verbose=False,
       warm_start=False)
RNA.fit(X_train, y_train)

"""Relizando o treinamento"""

# Ponto de Parada 5  - Olhar o treinamento dos modelos - Onde estão os modelos?

#---------------------------------------------------------------------------
## Previsão usando todos os conjuntos (treinamento e teste)
#---------------------------------------------------------------------------
# Árvore de Decisão
y_pred_train_DT = dtree.predict(X_train)
y_pred_test_DT  = dtree.predict(X_test)
y_pred_train_DT_C  = dtree.predict_proba(X_train)
y_pred_test_DT_C  = dtree.predict_proba(X_test)
# Regressão Linear
y_pred_train_RL = LinearReg.predict(X_train)
y_pred_test_RL  = LinearReg.predict(X_test)
# Regressão Logística
y_pred_train_RLog = LogisticReg.predict_proba(X_train)
y_pred_test_RLog  = LogisticReg.predict_proba(X_test)
# Redes Neurais
y_pred_train_RNA = RNA.predict(X_train)
y_pred_test_RNA  = RNA.predict(X_test)
y_pred_train_RNA_P = RNA.predict_proba(X_train)
y_pred_test_RNA_P  = RNA.predict_proba(X_test)

#----------------------------------------------------------------------
## Montando um Data Frame (Matriz) com os resultados
#----------------------------------------------------------------------
# Conjunto de treinamento
df_train = pd.DataFrame(y_pred_train_DT, columns=['CLASSIF_DT'])
#df_train['CLASSIF_RL'] = [1 if x > 0.6 else 0 for x in y_pred_train_RL]
#df_train['CLASSIF_RLog'] = [1 if x > 0.6 else 0 for x in y_pred_train_RLog[:,1]]
df_train['CLASSIF_RNA'] = y_pred_train_RNA
#df_train['REGRESSION_DT'] = [x for x in y_pred_train_DT_C[:,1]] 
#df_train['REGRESSION_RL'] = y_pred_train_RL
#df_train['REGRESSION_RLog'] = [x for x in y_pred_train_RLog[:,1]]
df_train['REGRESSION_RNA'] = [x for x in y_pred_train_RNA_P[:,1]]
df_train['TARGET'] = [x for x in y_train]
df_train['TRN_TST'] = 'TRAIN'
# Conjunto de test
df_test = pd.DataFrame(y_pred_test_DT, columns=['CLASSIF_DT'])
#df_test['CLASSIF_RL'] = [1 if x > 0.6 else 0 for x in y_pred_test_RL]
#df_test['CLASSIF_RLog'] = [1 if x > 0.6 else 0 for x in y_pred_test_RLog[:,1]]
df_test['CLASSIF_RNA'] = y_pred_test_RNA
#df_test['REGRESSION_DT'] = [x for x in y_pred_test_DT_C[:,1]]  
#df_test['REGRESSION_RL'] = y_pred_test_RL
#df_test['REGRESSION_RLog'] = [x for x in y_pred_test_RLog[:,1]]
df_test['REGRESSION_RNA'] = [x for x in y_pred_test_RNA_P[:,1]]
df_test['TARGET'] = [x for x in y_test]
df_test['TRN_TST'] = 'TEST' 
# Juntando Conjunto de Teste e Treinamento
df_total = pd.concat([df_test, df_train])

"""Verificando quem tem o menor MSE"""

##-----------------------------------------------------------------
## Cálculo dos erros da classificação e MSE (Mean Squared Error) Rede Neural
##-----------------------------------------------------------------
Erro_RNA_Classificacao = np.mean(np.absolute(df_test['CLASSIF_RNA'] - df_test['TARGET']))
Erro_RNA_MSE = np.mean((df_test['REGRESSION_RNA'] - df_test['TARGET']) ** 2) 
print()
print('---------------------------------------------')
print('Rede Neural - Erro de Classificação:', round(Erro_RNA_Classificacao, 4))
print('Rede Neural - MSE:',round(Erro_RNA_MSE, 4))
print('----------------------------------------------')
##-----------------------------------------------------------------
## Cálculo dos erros da classificação e MSE (Mean Squared Error) Regressão Logistica
##-----------------------------------------------------------------
#Erro_RLog_Classificacao = np.mean(np.absolute(df_test['CLASSIF_RLog'] - df_test['TARGET']))
#Erro_RLog_MSE = np.mean((df_test['REGRESSION_RLog'] - df_test['TARGET']) ** 2) 
#print()
#print('---------------------------------------------')
#print('Regressão Logistica - Erro de Classificação:',round(Erro_RLog_Classificacao, 4))
#print('Regressão Logistica - MSE:',round(Erro_RLog_MSE, 4))
#print('----------------------------------------------')
##-----------------------------------------------------------------
## Cálculo dos erros da classificação e MSE (Mean Squared Error) Regressão Linear
##-----------------------------------------------------------------
#Erro_RL_Classificacao = np.mean(np.absolute(df_test['CLASSIF_RL'] - df_test['TARGET']))
#Erro_RL_MSE = np.mean((df_test['REGRESSION_RL'] - df_test['TARGET']) ** 2) 
#print()
#print('---------------------------------------------')
#print('Regressão Linear - Erro de Classificação:', round(Erro_RL_Classificacao, 4))
#print('Regressão Linear - MSE:',round(Erro_RL_MSE, 4))
#print('----------------------------------------------')

df_total

# Verificando a acuracia dos algoritmos

#Improtando as bibliotecas necessarias
from sklearn.metrics import confusion_matrix

#print('Matrix de confusão da Regressão Linear')
#print('-----------------------------------')
#print( confusion_matrix(df_test['TARGET'], df_test['CLASSIF_RL']))
#print()
#print()
#print('Matrix de confusão da Regressão Logística')
#print('-----------------------------------')
#print(confusion_matrix(df_test['TARGET'], df_test['CLASSIF_RLog']))
#print()
#print()
#print('Matrix de confusão da Árvore de Decisão')
#print('-----------------------------------')
#print(confusion_matrix(df_test['TARGET'], df_test['CLASSIF_DT']))
print()
print()
print('Matrix de confusão da Rede Neural')
print('-----------------------------------')
print(confusion_matrix(df_test['TARGET'], df_test['CLASSIF_RNA']))

df_total

"""Exportando a analise"""

## Exportando os dados para avaliação dos resultados
#df_total.to_csv('/content/drive/MyDrive/IMPACTA/Comp Cognitiva/resultado_comparacao.csv')

df_total.to_excel('/content/drive/MyDrive/IMPACTA/Comp Cognitiva/resultado_comparacao.xlsx')

"""
Exportando o algoritmo"""

import pickle
File_RNA_Model = open('/content/drive/MyDrive/IMPACTA/Comp Cognitiva/RNA_Model.p', 'wb')
pickle.dump(RNA, File_RNA_Model)
File_RNA_Model.close()