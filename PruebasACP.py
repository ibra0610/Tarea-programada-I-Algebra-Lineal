from ACP import ACP
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as la 
import pandas as pd 
import seaborn as sns  

datos = pd.read_csv('EjemploEstudiantes.csv', delimiter = ';', decimal= ",", index_col=0) 


pca = ACP(datos, n_componentes = 2) 


