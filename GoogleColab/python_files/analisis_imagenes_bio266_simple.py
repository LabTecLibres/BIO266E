# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:14:57 2020

@author: Prosimios
"""

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import scipy.signal as scis

def transecto_peaks(imagen, transecto, base, umbral):
    plt.figure()
    plt.imshow(imagen)
    plt.axhline(transecto , color='r', ls ='--', label = 'transecto')

    x = [i for i in range(0,imagen.shape[1])]
    plt.figure()
    plt.plot(x,imagen[transecto, :, 0],'r',label = 'canal rojo')
    plt.plot(x,imagen[transecto, :, 1],'g',label = 'canal verde')
    plt.plot(x,imagen[transecto, :, 2],'b',label = 'canal azul')
    plt.axhline(base , color='k', ls ='--', label = 'base')

    sleep(2)

    plt.ylabel('valor del pixel')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    sleep(2)

    print('\nEn base a esta figura.\n'\
          'Escoge un color o suma de colores de la siguiente lista para hacer los análisis:\n')

    colores = ['Rojo','Verde','Rojo+Verde']
    plot_col = ['r','g','y']

    for pair in enumerate(colores):
        print(pair)
    sleep(2)
    eleccion = int(input('\nIngresa el valor: '))
    sleep(2)

    if eleccion == 2:
        valores = imagen[transecto, :, 0].astype('float64')  + imagen[transecto, :, 1].astype('float64')
    else:
        valores = imagen[transecto, :, eleccion]

    peaks = scis.find_peaks_cwt(valores, np.arange(1,int(len(valores)/10)))   
    x_peaks = list()
    peak_values = list()
    for peak in peaks:
        valor = valores[peak]
        if valor > base:
            x_peaks.append(peak)
            peak_values.append(valor)

    plt.figure()
    plt.plot(x,valores,color = plot_col[eleccion],label = 'canal '+colores[eleccion])
    plt.plot(x_peaks, peak_values, "xk")
    plt.axhline(base , color='k', ls ='--', label = 'base')
    plt.axhline(umbral , color='m', ls ='--', label = 'umbral')
    plt.ylabel('valor de la señal')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    print('\nLos valores de la señal son:')
    for i in range(0,len(x_peaks)):
        #print(str(x_peaks[i]), ' : ', str(peak_values[i]))
        print(i+1, ' : ', str(peak_values[i]))