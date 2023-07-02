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
    
    def centrar_datos(self): 
        datos_centrados = self.__datos - np.mean(self.__datos, axis = 0) 
        return datos_centrados 
    
    def matriz_correlaciones(self): 
        datos_centrados = self.centrar_datos() 
        matriz_correlaciones = np.corrcoef(datos_centrados.T) 
        return matriz_correlaciones 
    
    def eig_ordenados(self): 
        matriz_correlaciones = self.matriz_correlaciones() 
        valores_propios, vectores_propios = np.linalg.eig(matriz_correlaciones)  
        indices_ordenados = np.argsort(valores_propios)[:: -1]
        valores_propios_ordenados = valores_propios[indices_ordenados] 
        vectores_propios_ordenados = vectores_propios[:, indices_ordenados] 
        return valores_propios_ordenados, vectores_propios_ordenados 
    
    def componentes_principales(self): 
        _, vectores_propios_ordenados = self.eig_ordenados() 
        datos_centrados = self.centrar_datos() 
        componentes = np.dot(datos_centrados, vectores_propios_ordenados) 
        return componentes 
    
    def cosenos_individuos(self): 
        componentes = self.componentes_principales()
        cosenos_cuadrados = np.square(componentes) / np.sum(np.square(componentes), axis = 1, keepdims = True) 
        return cosenos_cuadrados 
    
    def correlacion_variables(self): 
        matriz_correlaciones = self.matriz_correlaciones() 
        _, vectores_propios_ordenados = self.eig_ordenados() 
        correlaciones = np.dot(matriz_correlaciones, vectores_propios_ordenados) 
        return correlaciones 
    
    def cosenos_variables(self): 
        _, vectores_propios_ordenados = self.eig_ordenados() 
        cosenos_cuadrados = np.square(vectores_propios_ordenados) 
        return cosenos_cuadrados 
    
    def varianza_explicada(self): 
        valores_propios_ordenados, _ = self.eig_ordenados() 
        varianza_explicada = np.cumsum(valores_propios_ordenados) / np.sum(valores_propios_ordenados) 
        return varianza_explicada 
    
    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal', colores = None): 
        nombres_estudiantes = self.__datos.index.tolist()
        componentes = self.componentes_principales() 
        plt.figure(figsize=(8, 8)) 
        plt.scatter(componentes[:, ejes[0]], componentes[:, ejes[1]], c=colores) 
        if ind_labels: 
            for i, (x,y) in enumerate(zip(componentes[:, ejes[0]], componentes[:, ejes[1]])): 
                plt.text(x, y, nombres_estudiantes[i]) 
        
        plt.xlabel('Componente Principal {}'.format(ejes[0] + 1)) 
        plt.ylabel('Componente Principal {}'.format(ejes[1] + 1))
        plt.title(titulo) 
        plt.grid(True) 
        plt.show() 
        
    def plot_circulo(self, ejes=[0, 1], var_labels= True, titulo = 'Circulo de Correlaciones'): 
        nombres_materias = self.__datos.columns.tolist()
        correlacion_var = self.correlacion_variables()
        fig, ax = plt.subplots()
        circulo = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='none')
        plt.gca().add_patch(circulo)
        for i, (x, y) in enumerate(zip(correlacion_var[:, ejes[0]], correlacion_var[:, ejes[1]])): 
            ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, fc='b', ec='b') #Diseno de las flechas del grafico, 'b' significa que son azules
            if var_labels: 
                ax.text(x, y, nombres_materias[i], fontsize=8) 
        plt.xlabel(f'Componente {ejes[0]+1}')
        plt.ylabel(f'Componente {ejes[1]+1}')
        plt.title(titulo)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        plt.grid()
        plt.show()