## Autor: Prof. Roberto Angelo
## Objetivo 1: Rever conceitos de pré-processamento
## Objetivo 2: Rever conceitos de regressão e classificação
## Objetivo 3: Revisar diversas técnicas
## Objetivo 4: Métricas de Comparação
## Objetivo 5: Comparar algoritmos 
##------------------------------------------------------------------------

# Bibliotecas padrão utilizadas
import numpy as np
import pandas as pd

## Carregando os dados
dataset = pd.read_csv('Case_cobranca.csv') 

# Ponto de Parada 1  -  Olhar variável dataset no explorador de variáveis
''' Recorte esta linha e cole nos pontos de paradas

#------------------------------------------------------------------------------------------
# Pré-processamento das variáveis
#------------------------------------------------------------------------------------------
## Tratamento de nulos no alvo --- Tempo de Atraso - transformação para alvo binário (>90 dias) 
dataset['ALVO']   = [0 if np.isnan(x) or x > 90 else 1 for x in dataset['TEMP_RECUPERACAO']]

# Ponto de Parada 2  -  Olhar nova coluna 'ALVO' criada na variável dataset


## Tratamento de nulos e normalização --- Variáveis de entrada numéricas
dataset['PRE_IDADE']        = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] # Trata mínimo
dataset['PRE_IDADE']        = [1. if x > 76 else (x-18)/(76-18) for x in dataset['PRE_IDADE']] # Trata máximo por percentil 99 e coloca na fórmula
dataset['PRE_QTDE_DIVIDAS'] = [0.  if np.isnan(x) else x/16. for x in dataset['QTD_DIVIDAS']] # retirada de outlier com percentil 99 e normalização     
##--- Dummies - transformação de atributos categóricos em numéricos e tratamanto de nulos ---------------
dataset['PRE_NOVO']         = [1 if x=='NOVO'                      else 0 for x in dataset['TIPO_CLIENTE']]    
dataset['PRE_TOMADOR_VAZIO']= [1 if x=='TOMADOR' or str(x)=='nan'  else 0 for x in dataset['TIPO_CLIENTE']]                        
dataset['PRE_CDC']          = [1 if x=='CDC'                       else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_PESSOAL']      = [1 if x=='PESSOAL'                   else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_SEXO_M']       = [1 if x=='M'                         else 0 for x in dataset['CD_SEXO']]

# Ponto de Parada 3  -  Olhar as novas colunas criadas na variável dataset


##------------------------------------------------------------
## Separando em dados de treinamento e teste
##------------------------------------------------------------
y = dataset['ALVO']              # Carrega alvo ou dataset.iloc[:,7].values
X = dataset.iloc[:, 8:15].values # Carrega colunas 8, 9, 10, 11, 12, 13 e 14 (a 15 não existe até este momento)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 25)

# Ponto de Parada 4  -  Olhar novas variáveis vetorias criadas X, y, X_test ...


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


# Ponto de Parada 6 - Olhar os modelos sendo rodados para os conjutos trn e tst
# OLhar y_pred_train_DT, y_pred_test_DT ... y_pred_test_RNA_P


#----------------------------------------------------------------------
## Montando um Data Frame (Matriz) com os resultados
#----------------------------------------------------------------------
# Conjunto de treinamento
df_train = pd.DataFrame(y_pred_train_DT, columns=['CLASSIF_DT'])
df_train['CLASSIF_RL'] = [1 if x > 0.6 else 0 for x in y_pred_train_RL]
df_train['CLASSIF_RLog'] = [1 if x > 0.6 else 0 for x in y_pred_train_RLog[:,1]]
df_train['CLASSIF_RNA'] = y_pred_train_RNA
df_train['REGRESSION_DT'] = [x for x in y_pred_train_DT_C[:,1]] 
df_train['REGRESSION_RL'] = y_pred_train_RL
df_train['REGRESSION_RLog'] = [x for x in y_pred_train_RLog[:,1]]
df_train['REGRESSION_RNA'] = [x for x in y_pred_train_RNA_P[:,1]]
df_train['TARGET'] = [x for x in y_train]
df_train['TRN_TST'] = 'TRAIN'
# Conjunto de test
df_test = pd.DataFrame(y_pred_test_DT, columns=['CLASSIF_DT'])
df_test['CLASSIF_RL'] = [1 if x > 0.6 else 0 for x in y_pred_test_RL]
df_test['CLASSIF_RLog'] = [1 if x > 0.6 else 0 for x in y_pred_test_RLog[:,1]]
df_test['CLASSIF_RNA'] = y_pred_test_RNA
df_test['REGRESSION_DT'] = [x for x in y_pred_test_DT_C[:,1]]  
df_test['REGRESSION_RL'] = y_pred_test_RL
df_test['REGRESSION_RLog'] = [x for x in y_pred_test_RLog[:,1]]
df_test['REGRESSION_RNA'] = [x for x in y_pred_test_RNA_P[:,1]]
df_test['TARGET'] = [x for x in y_test]
df_test['TRN_TST'] = 'TEST' 
# Juntando Conjunto de Teste e Treinamento
df_total = pd.concat([df_test, df_train])

## Exportando os dados para avaliação dos resultados
df_total.to_csv('resultado_comparacao.csv')


# Ponto de Parada 7 - Abrir resultado_comparacao.csv no Excel e calcular os erros classificação e MSE
# Gerar matriz de confusão no Excel  de todos os classificadores


## Matriz de Confusão Regressão Linear
from sklearn.metrics import confusion_matrix
print('Matrix de confusão da Regressão Linear')
print('-----------------------------------')
print( confusion_matrix(df_test['TARGET'], df_test['CLASSIF_RL']))
print()
print()
print('Matrix de confusão da Regressão Logística')
print('-----------------------------------')
print(confusion_matrix(df_test['TARGET'], df_test['CLASSIF_RLog']))
print()
print()
print('Matrix de confusão da Árvore de Decisão')
print('-----------------------------------')
print(confusion_matrix(df_test['TARGET'], df_test['CLASSIF_DT']))
print()
print()
print('Matrix de confusão da Rede Neural')
print('-----------------------------------')
print(confusion_matrix(df_test['TARGET'], df_test['CLASSIF_RNA']))

# Ponto de Parada 8 - Olhar as matrizes geradas aqui e...
# Comparar com as matrizes de confusão do Excel

##-----------------------------------------------------------------
## Cálculo dos erros da classificação e MSE (Mean Squared Error) RNA
##-----------------------------------------------------------------
Erro_RNA_Classificacao = np.mean(np.absolute(df_test['CLASSIF_RNA'] - df_test['TARGET']))
Erro_RNA_MSE = np.mean((df_test['REGRESSION_RNA'] - df_test['TARGET']) ** 2) 
print()
print('---------------------------------------------')
print('Rede Neural - Erro de Classificação:',Erro_RNA_Classificacao)
print('Rede Neural - MSE:',Erro_RNA_MSE)
print('----------------------------------------------')


# Ponto de Parada 9  -  Calcular os erros para os demais classficadores

### Regressão Linear
print('Regressão Linear')
print('Fórmula: ' + str(LinearReg.intercept_)) 
print('+ ' +  str(dataset.columns[8]) + '*' + str(LinearReg.coef_[0]) )
print('+ ' +  str(dataset.columns[9]) + '*' + str(LinearReg.coef_[1]) )                        
print('+ ' +  str(dataset.columns[10]) + '*' + str(LinearReg.coef_[2]))                        
print('+ ' +  str(dataset.columns[11]) + '*' + str(LinearReg.coef_[3]))
print('+ ' +  str(dataset.columns[12]) + '*' + str(LinearReg.coef_[4]))                        
print('+ ' +  str(dataset.columns[13]) + '*' + str(LinearReg.coef_[5])) 
print('+ ' +  str(dataset.columns[14]) + '*' + str(LinearReg.coef_[6]))                      
print('')

### Regressão Logística
print('Regressão Logística')
print('Fórmula: 1/(1 + exp(-( ' + str(LogisticReg.intercept_[0])) 
print('+ ' +  str(dataset.columns[8]) + '*' + str(LogisticReg.coef_[0,0]) )
print('+ ' +  str(dataset.columns[9]) + '*' + str(LogisticReg.coef_[0,1]) )                        
print('+ ' +  str(dataset.columns[10]) + '*' + str(LogisticReg.coef_[0,2]))                        
print('+ ' +  str(dataset.columns[11]) + '*' + str(LogisticReg.coef_[0,3]))
print('+ ' +  str(dataset.columns[12]) + '*' + str(LogisticReg.coef_[0,4]))                        
print('+ ' +  str(dataset.columns[13]) + '*' + str(LogisticReg.coef_[0,5])) 
print('+ ' +  str(dataset.columns[14]) + '*' + str(LogisticReg.coef_[0,6]))                      
print(')')


# Ponto de Parada 10  -  Observar os modelos gerados de RLin e RLog

# Fim de Paradas
'''

'''
# Salvar modelo:
import pickle
File_RNA_Model = open('RNA_Model.p', 'wb')
pickle.dump(RNA, File_RNA_Model)
File_RNA_Model.close()


# Carregar modelo:
import pickle
File_RNA_Model = open('RNA_Model.p', 'rb')
RNA = pickle.load(File_RNA_Model)
'''



