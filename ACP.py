#Tarea Programada Algebra Lineal 
#Javier Cruz C02517 Alexander Wang Wu C28559 Sebastián Arce Flores C10577 David Meléndez Aguilar C04726

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as la 
import pandas as pd 
import seaborn as sns 


class ACP: 

    def __init__(self, datos, n_componentes):
        self.__datos = datos 
        self.__n_componentes = n_componentes 
    
    @property 
    def datos(self): 
        return self.__datos 
    
    @datos.setter
    def datos(self, datos): 
        self.__datos = datos 

    @property
    def n_componentes(self): 
        return self.__n_componentes 
    
    @n_componentes.setter 
    def n_componentes(self, n_componentes): 
        self.__n_componentes = n_componentes 
    
    def centrar_datos(self):  #centrar y reducir la tabla de datos X 
        datos_centrados = self.__datos - np.mean(self.__datos, axis = 0) 
        return datos_centrados 
    
    def matriz_correlaciones(self): #calcular la Matriz de correlaciones 
        datos_centrados = self.centrar_datos() 
        matriz_correlaciones = np.corrcoef(datos_centrados.T) 
        return matriz_correlaciones 
    
    def eig_ordenados(self): #Calcular los vectores y valores propios de la matriz y ordenar de mayor a menor
        matriz_correlaciones = self.matriz_correlaciones() 
        valores_propios, vectores_propios = np.linalg.eig(matriz_correlaciones)  
        indices_ordenados = np.argsort(valores_propios)[:: -1]
        valores_propios_ordenados = valores_propios[indices_ordenados] 
        vectores_propios_ordenados = vectores_propios[:, indices_ordenados] 
        return valores_propios_ordenados, vectores_propios_ordenados 
    
    def componentes_principales(self):  #calcular la matriz de componentes principales 
        _, vectores_propios_ordenados = self.eig_ordenados() 
        datos_centrados = self.centrar_datos() 
        componentes = np.dot(datos_centrados, vectores_propios_ordenados) 
        return componentes 
    
    def cosenos_individuos(self):  #Calcular la matriz de calidades de los individuos 
        componentes = self.componentes_principales()
        cosenos_cuadrados = np.square(componentes) / np.sum(np.square(componentes), axis = 1, keepdims = True) 
        return cosenos_cuadrados 
    
    def correlacion_variables(self): #Calcular la matriz de coordenadas de las variables 
        matriz_correlaciones = self.matriz_correlaciones() 
        _, vectores_propios_ordenados = self.eig_ordenados() 
        correlaciones = np.dot(matriz_correlaciones, vectores_propios_ordenados) 
        return correlaciones 
    
    def cosenos_variables(self):  #Calcular la matriz de calidades de las variables (cosenos cuadrados) 
        _, vectores_propios_ordenados = self.eig_ordenados() 
        cosenos_cuadrados = np.square(vectores_propios_ordenados) 
        return cosenos_cuadrados 
    
    def varianza_explicada(self): #calcular el vector de inercias de los ejes 
        valores_propios_ordenados, _ = self.eig_ordenados() 
        varianza_explicada = np.cumsum(valores_propios_ordenados) / np.sum(valores_propios_ordenados) 
        return varianza_explicada 
    
    #Abajo se encuentran los dos metodos que se encargan de graficar los datos 

    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal', colores = None): 
        nombres_estudiantes = self.__datos.index.tolist() #se encarga de extraer solamente los nombres de los estudiantes de los datos para poder graficarlos
        componentes = self.componentes_principales() 
        plt.figure(figsize=(8, 8)) 
        plt.scatter(componentes[:, ejes[0]], componentes[:, ejes[1]], c=colores) #Formato de la grafica
        if ind_labels: 
            for i, (x,y) in enumerate(zip(componentes[:, ejes[0]], componentes[:, ejes[1]])): 
                plt.text(x, y, nombres_estudiantes[i]) #se encarga de imprimir los nombres de los estudiantes en la grafica, estando en un ciclo hace que los nombres aparezcan correctamente
        
        plt.xlabel('Componente Principal {}'.format(ejes[0] + 1)) #Determina el nombre de la etiqueta en el eje x
        plt.ylabel('Componente Principal {}'.format(ejes[1] + 1)) #Determina el nombre de la etiqueta en el eje y
        plt.title(titulo) #Determina el titulo de la grafica
        plt.grid(True) 
        plt.show() #muestra la grafica 
        
    def plot_circulo(self, ejes=[0, 1], var_labels= True, titulo = 'Circulo de Correlaciones'): 
        nombres_materias = self.__datos.columns.tolist() #se encarga de extraer solamente los nombres de las materias de los datos para poder graficarlos
        correlacion_var = self.correlacion_variables()
        fig, ax = plt.subplots()
        circulo = plt.Circle((0, 0), radius=1.75, edgecolor='k', facecolor='none') #crea el circulo de la grafica
        plt.gca().add_patch(circulo)
        for i, (x, y) in enumerate(zip(correlacion_var[:, ejes[0]], correlacion_var[:, ejes[1]])): 
            ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, fc='b', ec='b') #Diseno de las flechas del grafico, 'b' significa que son azules
            if var_labels: 
                ax.text(x, y, nombres_materias[i], fontsize=8) #se encarga de imprimir los nombres de las materias en la grafica, estando en un ciclo hace que los nombres aparezcan correctamente
        plt.xlabel(f'Componente {ejes[0]+1}') #Determina el nombre de la etiqueta en el eje x
        plt.ylabel(f'Componente {ejes[1]+1}') #Determina el nombre de la etiqueta en el eje y
        plt.title(titulo) #Determina el titulo de la grafica
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        plt.grid()
        plt.show() #muestra la grafica 