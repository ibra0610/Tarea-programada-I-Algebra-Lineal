from ACP import ACP
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as la 
import pandas as pd 
import seaborn as sns  

datos = pd.read_csv('EjemploEstudiantes.csv', delimiter = ';', decimal= ",", index_col=0) 


acp = ACP(datos, n_componentes = 5) 
acp.plot_plano_principal() 
acp.plot_circulo()


  

    #centrar y reducir la tabla de datos X 
    #calcular la Matriz de correlaciones 
    #Calcular los vectores y valores propios de la matriz 
    #Ordenar de mayor a menor estos valores propios 
    #calcular la matriz de componentes principales 
    #Calcular la matriz de calidades de los individuos 
    #Calcular la matriz de coordenadas de las variables 
    #Calcular la matriz de calidades de las variables (cosenos cuadrados) 
    #calcular el vemctor de inercias de los ejes 
    #Programar metodos que grafiquen el plano principal, el circulo de correlaciones. 