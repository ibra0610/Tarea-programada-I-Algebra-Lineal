#Tarea Programada Algebra Lineal 
#Javier Cruz C02517 Alexander Wang Wu C28559 Sebastián Arce Flores C10577

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as la 
import pandas as pd 
import seaborn as sns 


class ACP: 

    def __init__(self, datos, n_componentes):
        self.__datos = datos 
        self.__n_componentes = n_componentes 
    
    