#Tarea Programada Algebra Lineal 
#Javier Cruz C02517 Alexander Wang Wu C28559 Sebastián Arce Flores C10577

import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as la 
import pandas as pd 
import seaborn as sns 

#Cargar los datos del archivo EjemploEdtudiantes.csv

datos = pd.read_csv('EjemploEstudiantes.csv', delimiter=';', decimal= ",", index_col= 0) 
 

print(datos)