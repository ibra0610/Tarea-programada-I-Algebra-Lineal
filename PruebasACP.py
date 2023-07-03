#Tarea Programada Algebra Lineal 
#Javier Cruz C02517 Alexander Wang Wu C28559 Sebastián Arce Flores C10577 David Meléndez Aguilar C04726

from ACP import ACP
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as la 
import pandas as pd 
import seaborn as sns  

datos = pd.read_csv('EjemploEstudiantes.csv', delimiter = ';', decimal= ",", index_col=0)  
print(datos) 
colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] #arreglo de colores para que en los graficos aparezcan de diferente color
acp = ACP(datos, n_componentes = 5) 
acp.plot_plano_principal(colores=colores) 
acp.plot_circulo()



    
    
    
    
   
    
   
    
    #Programar metodos que grafiquen el plano principal, el circulo de correlaciones. 