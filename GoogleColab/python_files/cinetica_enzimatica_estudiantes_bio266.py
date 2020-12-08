# -------------------------
"""
Funciones que se utilizaran en el curso BIO266E - Laboratorio de Bioquímica II - Genética Molecular
Código bajo licencia GNU GPL 3.0
"""
# --------------------------

# Observación: Falta incluir los docstring de cada funcion

import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
import csv
import scipy.stats
import scipy.optimize as opt
import ipywidgets as widgets
import IPython
from IPython.display import HTML

def fit_micment_ph(datos_exp_x, pka_e1, pka_e2, substrate_conc, initial_enzyme):

    ka_e1 = 10**-pka_e1
    ka_e2 = 10**-pka_e2

    ph_dependant_km_numerator = np.multiply(np.add(np.add(np.divide(datos_exp_x, ka_e1), np.divide(ka_e2, datos_exp_x)), 1), 450)
    ph_dependant_km_denominator = np.add(np.add(np.divide(datos_exp_x, 10**-6.0), np.divide(10**-9.8, datos_exp_x)), 1)
    ph_dependant_km = np.divide(ph_dependant_km_numerator, ph_dependant_km_denominator)

    ph_dependant_vmax_numerator = 8606*(initial_enzyme)
    ph_dependant_vmax_denominator = np.add(np.add(np.divide(datos_exp_x, 10**-6.0), np.divide(10**-9.8, datos_exp_x)), 1)
    ph_dependant_vmax = np.divide(ph_dependant_vmax_numerator, ph_dependant_vmax_denominator)

    MicMent_ph = np.multiply(np.divide(np.multiply(ph_dependant_vmax, substrate_conc), np.add(ph_dependant_km, substrate_conc)), 60)

    return MicMent_ph

def fit_micment_temp(datos_exp_x, substrate_conc, k_m, initial_enzyme):

    # For observed rate
    k_cat_observed = np.multiply(np.divide(np.multiply(1.380E-23, datos_exp_x), 6.626E-34), np.exp(-(np.divide(50000, np.multiply(8.314, datos_exp_x)))))

    # For observed inactivation rate
    k_inactivation_observed = np.multiply(np.divide(np.multiply(1.380E-23, datos_exp_x), 6.626E-34), np.exp(-(np.divide(94000, np.multiply(8.314, datos_exp_x)))))

    # Equilibrium temperature
    k_temp_equilibrium = np.exp(np.multiply(np.divide(150000, 8.314), np.subtract(np.divide(1, 313), np.divide(1, datos_exp_x))))
    inactivation_parameter = np.exp(np.divide(np.multiply(np.multiply(k_temp_equilibrium, 1), k_inactivation_observed), np.add(1, k_temp_equilibrium)))

    # Calculation of temperature-dependant Vmax
    temperature_dependant_vmax =  np.divide(np.multiply(np.multiply(k_cat_observed, initial_enzyme), inactivation_parameter), np.add(1, k_temp_equilibrium))

    MicMent_temp = np.multiply(np.divide(np.multiply(temperature_dependant_vmax, substrate_conc), np.add(k_m, substrate_conc)), 60)

    return MicMent_temp

def fit_micment_comp_inhibitor(datos_exp_x, inhibitor_concentration, k_ic, initial_enzyme):

    substrate_concentration = datos_exp_x

    # Apparent parameters
    v_max_comp_inhibition = 8606*(initial_enzyme)
    km_comp_inhibition = np.multiply(np.add(np.divide(inhibitor_concentration, k_ic), 1), 450)

    MicMent_comp_inhib = np.multiply(np.divide(np.multiply(v_max_comp_inhibition, substrate_concentration), np.add(km_comp_inhibition, substrate_concentration)), 60)

    return MicMent_comp_inhib

def fit_micment_mixed_inhibitor(datos_exp_x, inhibitor_concentration, k_iu, k_ic, initial_enzyme):

    substrate_concentration = datos_exp_x
    mixed_inhibition_v_max = np.divide(np.multiply(8606, initial_enzyme), (np.add(1, np.divide(inhibitor_concentration, k_iu))))
    mixed_inhibition_km = np.divide(np.multiply(np.add(1, np.divide(inhibitor_concentration, k_ic)), 450), np.add(1, np.divide(inhibitor_concentration, k_iu)))

    MicMent_mixed_inhib = np.multiply(np.divide(np.multiply(mixed_inhibition_v_max, substrate_concentration), np.add(mixed_inhibition_km, substrate_concentration)), 60)

    return MicMent_mixed_inhib

def fit_micment_uncomp_inhibitor(datos_exp_x, inhibitor_concentration, k_iu, initial_enzyme):

    substrate_concentration = datos_exp_x
    uncomp_inhibition_v_max = np.divide(np.multiply(8606, initial_enzyme), np.add(np.divide(inhibitor_concentration, k_iu), 1))
    uncomp_inhibition_km = np.divide(450, np.add(np.divide(inhibitor_concentration, k_iu), 1))

    MicMent_uncomp_inhib = np.multiply(np.divide(np.multiply(uncomp_inhibition_v_max, substrate_concentration), np.add(uncomp_inhibition_km, substrate_concentration)), 60)

    return MicMent_uncomp_inhib

def fit_micment_dil(datos_exp_x, v_lim, k_m):

    MicMent = np.multiply(np.divide(np.multiply(v_lim, datos_exp_x), np.add(k_m, datos_exp_x)), 60)
    
    return MicMent

def fit_lwburk(datos_exp_x, pendiente, intercepto):
    
    lineweaver_burk = np.add(np.multiply(datos_exp_x, pendiente), intercepto)

    return lineweaver_burk

def fit_progress_curve(datos_exp_x, a, b):
    
    MicMent_progress = np.add(np.multiply(datos_exp_x, a), b)

    return MicMent_progress

def generarParametrosIniciales(func_to_fit, datos_exp_x, datos_exp_y):

    def sumaDeErrorCuadrado(parametros):
    
        warnings.filterwarnings("ignore")
        value_array = func_to_fit(datos_exp_x, *parametros)
        
        return np.sum((datos_exp_y - value_array) ** 2.0)

    rangoParametros = []

    for i in range(1, func_to_fit.__code__.co_argcount):
        limite_inferior = float(input("Por favor ingrese el límite inferior de búsqueda para {} : \n".format(func_to_fit.__code__.co_varnames[i])))
        limite_superior = float(input("Por favor ingrese el límite superior de búsqueda para {} : \n".format(func_to_fit.__code__.co_varnames[i])))
        rangoParametros.append([limite_inferior, limite_superior])
        
    param = [[], []]

    for elem in rangoParametros:

        param[0].append(elem[0])
        param[1].append(elem[1])
        
    resultados = opt.differential_evolution(sumaDeErrorCuadrado, bounds=rangoParametros, seed=6)

    return resultados.x, param

def ajustar_modelo(func_to_fit, datos_exp_x, datos_exp_y):
    
    if func_to_fit.__name__ == "fit_micment_dil":
        
        resultados_modelo = {}

        parametrosEvoDif, limite_parametros = generarParametrosIniciales(func_to_fit, datos_exp_x, datos_exp_y)

        parametrosAjustados, pcov = opt.curve_fit(func_to_fit, datos_exp_x, datos_exp_y, p0=parametrosEvoDif, bounds=limite_parametros)

        print("Parámetros Ajustados:")
        for i in range(1, func_to_fit.__code__.co_argcount):
            print("{} : {}".format(func_to_fit.__code__.co_varnames[i], parametrosAjustados[i-1]))
            
        print("\nDebido a que la funcion ajusta la V_max en uM/seg, se convertirá a uM/min")
        print("La V_max ajustada (en uM/min) es:\n", parametrosAjustados[0]*60)

        prediccionesModelo = func_to_fit(datos_exp_x, *parametrosAjustados)
        errorAbs = prediccionesModelo - datos_exp_y
        difDatosModelo = datos_exp_y - prediccionesModelo

        errorCuadrado = np.square(errorAbs)
        mediasErrorCuadrado = np.mean(errorCuadrado)
        raizMediasErrorCuadrado = np.sqrt(mediasErrorCuadrado)

        r_cuadrado_modelo = 1.0 - (np.var(errorAbs)/np.var(datos_exp_y))

        print("R cuadrado : {}".format(r_cuadrado_modelo))

        resultados_modelo["prediccionesModelo"] = prediccionesModelo
        resultados_modelo["difDatosModelo"] = difDatosModelo

        return resultados_modelo

    if func_to_fit.__name__ == "fit_micment_comp_inhibitor":
        
        resultados_modelo = {}

        parametrosEvoDif, limite_parametros = generarParametrosIniciales(func_to_fit, datos_exp_x, datos_exp_y)

        parametrosAjustados, pcov = opt.curve_fit(func_to_fit, datos_exp_x, datos_exp_y, p0=parametrosEvoDif, bounds=limite_parametros)

        print("Parámetros Ajustados:")
        for i in range(1, func_to_fit.__code__.co_argcount):
            print("{} : {}".format(func_to_fit.__code__.co_varnames[i], parametrosAjustados[i-1]))
            
        print("\nA partir de los parámetros ajustados se calcula la V_max y Km:\n")
        print("V_max:", round(8606*parametrosAjustados[2], 4))
        print("Km:", round(((parametrosAjustados[0]/parametrosAjustados[1])+1)*450, 4))
        print("\n")

        prediccionesModelo = func_to_fit(datos_exp_x, *parametrosAjustados)
        errorAbs = prediccionesModelo - datos_exp_y
        difDatosModelo = datos_exp_y - prediccionesModelo

        errorCuadrado = np.square(errorAbs)
        mediasErrorCuadrado = np.mean(errorCuadrado)
        raizMediasErrorCuadrado = np.sqrt(mediasErrorCuadrado)

        r_cuadrado_modelo = 1.0 - (np.var(errorAbs)/np.var(datos_exp_y))

        print("R cuadrado : {}".format(r_cuadrado_modelo))

        resultados_modelo["prediccionesModelo"] = prediccionesModelo
        resultados_modelo["difDatosModelo"] = difDatosModelo

        return resultados_modelo
    
    if func_to_fit.__name__ == "fit_micment_mixed_inhibitor":
        
        resultados_modelo = {}

        parametrosEvoDif, limite_parametros = generarParametrosIniciales(func_to_fit, datos_exp_x, datos_exp_y)

        parametrosAjustados, pcov = opt.curve_fit(func_to_fit, datos_exp_x, datos_exp_y, p0=parametrosEvoDif, bounds=limite_parametros)

        print("Parámetros Ajustados:")
        for i in range(1, func_to_fit.__code__.co_argcount):
            print("{} : {}".format(func_to_fit.__code__.co_varnames[i], parametrosAjustados[i-1]))
            
        print("\nA partir de los parámetros ajustados se calcula la V_max y Km:\n")
        print("V_max:", round((8606*parametrosAjustados[3])/((parametrosAjustados[0]/parametrosAjustados[1])+1), 4))
        print("Km:", round((((parametrosAjustados[0]/parametrosAjustados[2])+1)*450)/(1+(parametrosAjustados[0]/parametrosAjustados[1])), 4))
        print("\n")

        prediccionesModelo = func_to_fit(datos_exp_x, *parametrosAjustados)
        errorAbs = prediccionesModelo - datos_exp_y
        difDatosModelo = datos_exp_y - prediccionesModelo

        errorCuadrado = np.square(errorAbs)
        mediasErrorCuadrado = np.mean(errorCuadrado)
        raizMediasErrorCuadrado = np.sqrt(mediasErrorCuadrado)

        r_cuadrado_modelo = 1.0 - (np.var(errorAbs)/np.var(datos_exp_y))

        print("R cuadrado : {}".format(r_cuadrado_modelo))

        resultados_modelo["prediccionesModelo"] = prediccionesModelo
        resultados_modelo["difDatosModelo"] = difDatosModelo

        return resultados_modelo
        
    if func_to_fit.__name__ == "fit_micment_uncomp_inhibitor":
        
        resultados_modelo = {}

        parametrosEvoDif, limite_parametros = generarParametrosIniciales(func_to_fit, datos_exp_x, datos_exp_y)

        parametrosAjustados, pcov = opt.curve_fit(func_to_fit, datos_exp_x, datos_exp_y, p0=parametrosEvoDif, bounds=limite_parametros)

        print("Parámetros Ajustados:")
        for i in range(1, func_to_fit.__code__.co_argcount):
            print("{} : {}".format(func_to_fit.__code__.co_varnames[i], parametrosAjustados[i-1]))
            
        print("\nA partir de los parámetros ajustados se calcula la V_max y Km:\n")
        print("V_max:", round((8606*parametrosAjustados[2])/((parametrosAjustados[0]/parametrosAjustados[1])+1), 4))
        print("Km:", round(450/((parametrosAjustados[0]/parametrosAjustados[1])+1), 4))
        print("\n")

        prediccionesModelo = func_to_fit(datos_exp_x, *parametrosAjustados)
        errorAbs = prediccionesModelo - datos_exp_y
        difDatosModelo = datos_exp_y - prediccionesModelo

        errorCuadrado = np.square(errorAbs)
        mediasErrorCuadrado = np.mean(errorCuadrado)
        raizMediasErrorCuadrado = np.sqrt(mediasErrorCuadrado)

        r_cuadrado_modelo = 1.0 - (np.var(errorAbs)/np.var(datos_exp_y))

        print("R cuadrado : {}".format(r_cuadrado_modelo))

        resultados_modelo["prediccionesModelo"] = prediccionesModelo
        resultados_modelo["difDatosModelo"] = difDatosModelo

        return resultados_modelo
        
    else: 
    
        resultados_modelo = {}

        parametrosEvoDif, limite_parametros = generarParametrosIniciales(func_to_fit, datos_exp_x, datos_exp_y)

        parametrosAjustados, pcov = opt.curve_fit(func_to_fit, datos_exp_x, datos_exp_y, p0=parametrosEvoDif, bounds=limite_parametros)

        print("Parámetros Ajustados:")
        for i in range(1, func_to_fit.__code__.co_argcount):
            print("{} : {}".format(func_to_fit.__code__.co_varnames[i], parametrosAjustados[i-1]))

        prediccionesModelo = func_to_fit(datos_exp_x, *parametrosAjustados)
        errorAbs = prediccionesModelo - datos_exp_y
        difDatosModelo = datos_exp_y - prediccionesModelo

        errorCuadrado = np.square(errorAbs)
        mediasErrorCuadrado = np.mean(errorCuadrado)
        raizMediasErrorCuadrado = np.sqrt(mediasErrorCuadrado)

        r_cuadrado_modelo = 1.0 - (np.var(errorAbs)/np.var(datos_exp_y))

        print("R cuadrado : {}".format(r_cuadrado_modelo))

        resultados_modelo["prediccionesModelo"] = prediccionesModelo
        resultados_modelo["difDatosModelo"] = difDatosModelo

        return resultados_modelo

def typeset():
  """
  MathJax initialization for the current cell.
  
  This installs and configures MathJax for the current output.
  """
  display(HTML('''
      <script src="https://www.gstatic.com/external_hosted/mathjax/latest/MathJax.js?config=TeX-AMS_HTML-full,Safe&delayStartupUntil=configured"></script>
      <script>
        (() => {
          const mathjax = window.MathJax;
          mathjax.Hub.Config({
          'tex2jax': {
            'inlineMath': [['$', '$'], ['\\(', '\\)']],
            'displayMath': [['$$', '$$'], ['\\[', '\\]']],
            'processEscapes': true,
            'processEnvironments': true,
            'skipTags': ['script', 'noscript', 'style', 'textarea', 'code'],
            'displayAlign': 'center',
          },
          'HTML-CSS': {
            'styles': {'.MathJax_Display': {'margin': 0}},
            'linebreaks': {'automatic': true},
            // Disable to prevent OTF font loading, which aren't part of our
            // distribution.
            'imageFont': null,
          },
          'messageStyle': 'none'
        });
        mathjax.Hub.Configured();
      })();
      </script>
      '''))

def interactive_ph_graph(ph_range, ph_data_func):
    
    # Para renderizar LaTeX
    # ce.typeset()
    
    # Se obtienen los datos originales
    ph_range = ph_range
    x_data = np.power(10, np.multiply(ph_range, -1))
    y_data = ph_data_func(x_data, 5.3, 8.8, 500, 4.3E-3/250)
    
    def update(a, b, c, d, e, f):
        
        plt.figure(dpi=150)
        plt.title("Variación de la velocidad de reaccion frente a cambios en pKa")
        plt.xlabel("pH")
        plt.ylabel("Tasa de Reaccion (uM/min)")
        
        # Update pH parameters
        ka_e1 = 10**-a
        ka_e2 = 10**-b
        ka_ea1 = 10**-c
        ka_ea2 = 10**-d
        substrate_conc = e
        initial_enzyme = f
        
        ph_dependant_km_numerator = np.multiply(np.add(np.add(np.divide(x_data, ka_e1), np.divide(ka_e2, x_data)), 1), 450)
        ph_dependant_km_denominator = np.add(np.add(np.divide(x_data, ka_ea1), np.divide(ka_ea2, x_data)), 1)
        ph_dependant_km = np.divide(ph_dependant_km_numerator, ph_dependant_km_denominator)

        ph_dependant_vmax_numerator = 8606*(initial_enzyme)
        ph_dependant_vmax_denominator = np.add(np.add(np.divide(x_data, ka_ea1), np.divide(ka_ea2, x_data)), 1)
        ph_dependant_vmax = np.divide(ph_dependant_vmax_numerator, ph_dependant_vmax_denominator)

        MicMent_ph = np.multiply(np.divide(np.multiply(ph_dependant_vmax, substrate_conc), np.add(ph_dependant_km, substrate_conc)), 60)

        plt.scatter(ph_range, y_data, c="r", label="pKa_E 1 original: {}\npKa_E 2 original: {}\npKa_ES 1 original: {}\npKa_ES 2 original: {}\nConcentración de sustrato: {}\nConcentración de enzima: {}".format(5.3, 8.8, 6.0, 9.8, 500, 4.3E-3/250), s=8.0)
        plt.scatter(ph_range, MicMent_ph, c="b", label="pKa_E 1 modificado: {}\npKa_E 2 modificado: {}\npKa_ES 1 modificado: {}\npKa_ES 2 modificado: {}\nConcentración de sustrato modificada: {}\nConcentración de enzima modificada: {}".format(a, b, c, d, e, f), s=8.0)
        plt.plot(ph_range, y_data, c="r", lw=2.0, alpha=0.5)
        plt.plot(ph_range, MicMent_ph, c="b", lw=2.0, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        
    pka1_slider = widgets.FloatSlider(value=5.3, min=1.0, max=13.0, step=0.1, continuous_update=True, description="pKa enzima 1", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    pka2_slider = widgets.FloatSlider(value=8.8, min=1.0, max=13.0, step=0.1, continuous_update=True, description="pKa enzima 2", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    pka1_es_slider = widgets.FloatSlider(value=6.0, min=1.0, max=13.0, step=0.1, continuous_update=True, description="pKa enzima-sustrato 1", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    pka2_es_slider = widgets.FloatSlider(value=9.8, min=1.0, max=13.0, step=0.1, continuous_update=True, description="pKa enzima-sustrato 2", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    substrate_slider = widgets.FloatSlider(value=500, min=500.0, max=10000.0, step=100, continuous_update=True, description="Concentracion de sustrato", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    enzyme_dropdown = widgets.Dropdown(options=[round(num, 8) for num in list(np.concatenate((np.linspace((4.3E-3/10000), (4.3E-3/1000), 50), np.linspace((4.3E-3/1000), 4.3E-3, 50))))], value=4.3E-3/1000, description='Concentracion de enzima:', disabled=False, layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})

    output_graph = widgets.interactive_output(update, {'a':pka1_slider, 'b':pka2_slider, 'c':pka1_es_slider, 'd':pka2_es_slider, 'e':substrate_slider, 'f':enzyme_dropdown})
    header = widgets.HTML(value="<H1><center>Gráfico Interactivo pH - Enzimas</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    footer = widgets.HTML(value="<H1><center>A continuación se realizarán algunas actividades relacionadas con el efecto del pH en las enzimas</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    equation = widgets.HTMLMath(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    sliders = widgets.VBox([pka1_slider, pka2_slider, pka1_es_slider, pka2_es_slider, substrate_slider, enzyme_dropdown])

    final_output = widgets.AppLayout(header=header,
            left_sidebar=sliders,
            center=output_graph,
            footer=footer,
            right_sidebar=None,
            align_items='center',
            pane_widths=[3, 3, 1],
            pane_heights=[1, 5, '60px'])
    
    return final_output

def interactive_temp_graph(temp_range, temp_data_func):
    
    # Para renderizar LaTeX
    # ce.typeset()
    
    # Se obtienen los datos originales
    x_data = temp_range
    y_data = temp_data_func(x_data, 500, 450, 4.3E-3/250)
    
    def update(a, b, c, d, e, f):
        
        plt.figure(dpi=150)
        plt.title("Variación de la velocidad de reaccion frente a cambios en la temperatura")
        plt.xlabel("Temperatura (K)")
        plt.ylabel("Tasa de Reaccion (uM/min)")
        
        # Update temp parameters
        # For observed rate
        k_cat_observed = np.multiply(np.divide(np.multiply(1.380E-23, x_data), 6.626E-34), np.exp(-(np.divide(a, np.multiply(8.314, x_data)))))

        # For observed inactivation rate
        k_inactivation_observed = np.multiply(np.divide(np.multiply(1.380E-23, x_data), 6.626E-34), np.exp(-(np.divide(b, np.multiply(8.314, x_data)))))

        # Equilibrium temperature
        k_temp_equilibrium = np.exp(np.multiply(np.divide(c, 8.314), np.subtract(np.divide(1, d), np.divide(1, x_data))))
        inactivation_parameter = np.exp(np.divide(np.multiply(np.multiply(k_temp_equilibrium, 1), k_inactivation_observed), np.add(1, k_temp_equilibrium)))

        # Calculation of temperature-dependant Vmax
        temperature_dependant_vmax =  np.divide(np.multiply(np.multiply(k_cat_observed, f), inactivation_parameter), np.add(1, k_temp_equilibrium))

        MicMent_temp = np.multiply(np.divide(np.multiply(temperature_dependant_vmax, e), np.add(450, e)), 60)

        plt.scatter(x_data, y_data, c="r", label="$\Delta G_a$: {}\n$\Delta G_i$: {}\n$\Delta H_e$: {}\nTemperatura Equilibrio: {}\nConcentracion de sustrato: {}\nConcentracion de enzima: {}".format(50000, 94000, 150000, 313, 500, 4.3E-3/250), s=8.0)
        plt.scatter(x_data, MicMent_temp, c="b", label="$\Delta G_a$ modificado: {}\n$\Delta G_i$ modificado: {}\n$\Delta H_e$ modificado: {}\n$T_e$ modificada: {}\nConcentracion de sustrato modificada: {}\nConcentracion de enzima modificada: {}".format(a, b, c, d, e, f), s=8.0)
        plt.plot(x_data, y_data, c="r", lw=2.0, alpha=0.5)
        plt.plot(x_data, MicMent_temp, c="b", lw=2.0, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        
    delta_g_act_slider = widgets.FloatSlider(value=50000.0, min=25000.0, max=75000.0, step=1000.0, continuous_update=True, description="dG de activacion", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    delta_g_inact_slider = widgets.FloatSlider(value=94000.0, min=50000.0, max=150000.0, step=1000.0, continuous_update=True, description="dG de inactivacion", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    delta_h_eq_slider = widgets.FloatSlider(value=150000.0, min=125000.0, max=400000.0, step=1000.0, continuous_update=True, description="dH de equilibrio", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    temp_eq_slider = widgets.FloatSlider(value=313.0, min=283.0, max=383.0, step=1, continuous_update=True, description="T de equilibrio", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    substrate_slider = widgets.FloatSlider(value=500.0, min=500.0, max=10000.0, step=100, continuous_update=True, description="Concentracion de sustrato", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    enzyme_dropdown = widgets.Dropdown(options=[round(num, 8) for num in list(np.concatenate((np.linspace((4.3E-3/10000), (4.3E-3/1000), 50), np.linspace((4.3E-3/1000), 4.3E-3, 50))))], value=4.3E-3/1000, description='Concentracion de enzima:', disabled=False, layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})

    output_graph = widgets.interactive_output(update, {'a':delta_g_act_slider, 'b':delta_g_inact_slider, 'c':delta_h_eq_slider, 'd':temp_eq_slider, 'e':substrate_slider, 'f':enzyme_dropdown})
    header = widgets.HTML(value="<H1><center>Gráfico Interactivo Temperatura - Enzimas</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    footer = widgets.HTML(value="<H1><center>A continuación se realizarán algunas actividades relacionadas con el efecto de la temperatura en las enzimas</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    equation = widgets.HTMLMath(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    sliders = widgets.VBox([delta_g_act_slider, delta_g_inact_slider, delta_h_eq_slider, temp_eq_slider, substrate_slider, enzyme_dropdown])

    final_output = widgets.AppLayout(header=header,
            left_sidebar=sliders,
            center=output_graph,
            footer=footer,
            right_sidebar=None,
            align_items='center',
            pane_widths=[3, 3, 1],
            pane_heights=[1, 5, '60px'])
    
    return final_output

def interactive_progress_graph(timespan, progress_data_func):
    
    # Para renderizar LaTeX
    # ce.typeset()
    
    # Se obtienen los datos originales
    x_data = timespan
    y_data = progress_data_func(x_data, 22.7, 0)
    
    def update(a):
        
        plt.figure(dpi=150)
        plt.title("Variación de la concentración de producto por unidad de tiempo")
        plt.xlabel("Tiempo (min)")
        plt.ylabel("Producto (uM)")
        
        # Update temp parameters
        MicMent_progress = np.add(np.multiply(x_data, a), 0)

        plt.scatter(x_data, y_data, c="r", label="a: {}".format(500), s=8.0)
        plt.scatter(x_data, MicMent_progress, c="b", label="a modificada: {}".format(a), s=8.0)
        plt.plot(x_data, y_data, c="r", lw=2.0, alpha=0.5)
        plt.plot(x_data, MicMent_progress, c="b", lw=2.0, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        
    a_slider = widgets.FloatSlider(value=22.7, min=0, max=100.0, step=0.5, continuous_update=True, description="a", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})

    output_graph = widgets.interactive_output(update, {'a':a_slider})
    header = widgets.HTML(value="<H1><center>Gráfico Interactivo Curva Progreso - Enzimas</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    footer = widgets.HTML(value="<H1><center>A continuación se realizarán algunas actividades relacionadas con las curvas de progreso y su utilidad en la cinética enzimática</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    equation = widgets.HTMLMath(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    sliders = widgets.VBox([a_slider])

    final_output = widgets.AppLayout(header=header,
            left_sidebar=sliders,
            center=output_graph,
            footer=footer,
            right_sidebar=None,
            align_items='center',
            pane_widths=[3, 3, 1],
            pane_heights=[1, 5, '60px'])
    
    return final_output

def interactive_comp_inhibitor_graph1(substrate_range, comp_inhib_data_func):
    
    # Para renderizar LaTeX
    # ce.typeset()
    
    # Se obtienen los datos originales
    x_data = substrate_range
    y_data = comp_inhib_data_func(x_data, 10000, 2500, 4.3E-3/250)
    
    def update(a, b, c):
        
        plt.figure(dpi=150)

        # Apparent parameters
        v_max_comp_inhibition = 8606*(c)
        km_comp_inhibition = np.multiply(np.add(np.divide(a, b), 1), 450)
        normal_data = np.multiply(np.divide(np.multiply(8606*c, x_data), np.add(450, x_data)), 60)

        MicMent_comp_inhib = np.multiply(np.divide(np.multiply(v_max_comp_inhibition, x_data), np.add(km_comp_inhibition, x_data)), 60)
        plt.scatter(x_data, normal_data, c="r", label="Sin inhibidor\n competitivo".format(10000, 2500, 4.3E-3/250), s=2.0)
        plt.scatter(x_data, MicMent_comp_inhib, c="b", label="[Inhibidor]:\n{}\nK_ic:\n{}\n[Enzima]:\n{}".format(a, b, c), s=2.0)
        plt.plot(x_data, normal_data, c="r", lw=2.0, alpha=0.5)
        plt.plot(x_data, MicMent_comp_inhib, c="b", lw=2.0, alpha=0.5)
        plt.xlabel("Concentration de sustrato (uM)")
        plt.ylabel("Tasa de Reaccion\n (uM/min)")
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

        plt.tight_layout()
        
    inhibitor_slider = widgets.FloatSlider(value=10000, min=0, max=50000, step=500, continuous_update=True, description="Concentración de \nInhibidor (uM):", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    kic_slider = widgets.FloatSlider(value=2500, min=500, max=20000, step=500, continuous_update=True, description="K_ic:", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    enzyme_dropdown = widgets.Dropdown(options=[round(num, 8) for num in list(np.hstack((np.linspace((4.3E-3/10000), (4.3E-3/1500), 50), 4.3E-3/250, np.linspace((4.3E-3/1000), 4.3E-3, 50))))], value=4.3E-3/250, description='Concentracion de \nenzima (uM):', disabled=False, layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})

    output_graph = widgets.interactive_output(update, {'a':inhibitor_slider, 'b':kic_slider, 'c':enzyme_dropdown})
    header = widgets.HTML(value="<H1><center>Gráfico Interactivo de Inhibidores Competitivos</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    footer = widgets.HTML(value="<H1><center>A continuación se realizarán algunas actividades relacionadas al efecto que tienen los inhibidores competitivos sobre la tasa de reacción enzimática</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    equation = widgets.HTMLMath(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    sliders = widgets.VBox([inhibitor_slider, kic_slider, enzyme_dropdown])

    final_output = widgets.AppLayout(header=header,
            left_sidebar=sliders,
            center=output_graph,
            footer=footer,
            right_sidebar=None,
            align_items='center',
            pane_widths=[1, 1, 0],
            pane_heights=[1, 7, '100px'])

    return final_output

def interactive_mixed_inhibitor_graph1(substrate_range, mixed_inhib_data_func):
    
    # Para renderizar LaTeX
    # ce.typeset()
    
    # Se obtienen los datos originales
    x_data = substrate_range
    y_data = mixed_inhib_data_func(x_data, 10000, 17000, 17000, 4.3E-3/250)
    
    def update(a, b, c, d):
        
        plt.figure(dpi=150)

        # Apparent parameters
        mixed_inhibition_v_max = np.divide(np.multiply(8606, d), (np.add(1, np.divide(a, c))))
        mixed_inhibition_km = np.divide(np.multiply(np.add(1, np.divide(a, b)), 450), np.add(1, np.divide(a, c)))

        MicMent_mixed_inhib = np.multiply(np.divide(np.multiply(mixed_inhibition_v_max, x_data), np.add(mixed_inhibition_km, x_data)), 60)
        normal_data = np.multiply(np.divide(np.multiply(8606*d, x_data), np.add(450, x_data)), 60)
        
        plt.scatter(x_data, normal_data, c="r", label="Sin inhibidor\n competitivo".format(10000, 2500, 4.3E-3/250), s=2.0)
        plt.scatter(x_data, MicMent_mixed_inhib, c="b", label="[Inhibidor]:\n{}\nK_ic:\n{}\nK_iu:\n{}\n[Enzima]:\n{}".format(a, b, c, d), s=2.0)
        plt.plot(x_data, normal_data, c="r", lw=2.0, alpha=0.5)
        plt.plot(x_data, MicMent_mixed_inhib, c="b", lw=2.0, alpha=0.5)
        plt.xlabel("Concentration de sustrato (uM)")
        plt.ylabel("Tasa de Reaccion\n (uM/min)")
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

        plt.tight_layout()
        
    inhibitor_slider = widgets.FloatSlider(value=8500, min=0, max=50000, step=500, continuous_update=True, description="Concentración de Inhibidor (uM):", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    kic_slider = widgets.FloatSlider(value=17000, min=500, max=20000, step=500, continuous_update=True, description="K_ic:", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    kiu_slider = widgets.FloatSlider(value=17000, min=500, max=20000, step=500, continuous_update=True, description="K_iu:", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    enzyme_dropdown = widgets.Dropdown(options=[round(num, 8) for num in list(np.hstack((np.linspace((4.3E-3/10000), (4.3E-3/1500), 50), 4.3E-3/250, np.linspace((4.3E-3/1000), 4.3E-3, 50))))], value=4.3E-3/250, description='Concentracion de enzima (uM):', disabled=False, layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})

    output_graph = widgets.interactive_output(update, {'a':inhibitor_slider, 'b':kic_slider, 'c':kiu_slider, 'd':enzyme_dropdown})
    header = widgets.HTML(value="<H1><center>Gráfico Interactivo de Inhibidores Mixtos</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    footer = widgets.HTML(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    equation = widgets.HTMLMath(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    sliders = widgets.VBox([inhibitor_slider, kic_slider, kiu_slider, enzyme_dropdown])

    final_output = widgets.AppLayout(header=header,
            left_sidebar=sliders,
            center=output_graph,
            footer=None,
            right_sidebar=None,
            align_items='center',
            pane_widths=[1, 1, 0],
            pane_heights=[1, 7, '100px'])

    return final_output

def interactive_uncomp_inhibitor_graph1(substrate_range, uncomp_inhib_data_func):
    
    # Para renderizar LaTeX
    # ce.typeset()
    
    # Se obtienen los datos originales
    x_data = substrate_range
    y_data = uncomp_inhib_data_func(x_data, 10000, 13000, 4.3E-3/250)
    
    def update(a, b, c):
        
        plt.figure(dpi=150)

        # Apparent parameters
        uncomp_inhibition_v_max = np.divide(np.multiply(8606, c), np.add(np.divide(a, b), 1))
        uncomp_inhibition_km = np.divide(450, np.add(np.divide(a, b), 1))

        MicMent_uncomp_inhib = np.multiply(np.divide(np.multiply(uncomp_inhibition_v_max, x_data), np.add(uncomp_inhibition_km, x_data)), 60)
        normal_data = np.multiply(np.divide(np.multiply(8606*c, x_data), np.add(450, x_data)), 60)

        plt.scatter(x_data[:250], normal_data[:250], c="r", label="Sin inhibidor\nno-competitivo".format(10000, 7500, 4.3E-3/250), s=2.0)
        plt.scatter(x_data[:250], MicMent_uncomp_inhib[:250], c="b", label="[Inhibidor]:\n{}\nK_iu:\n{}\n[Enzima]:\n{}".format(a, b, c), s=2.0)
        plt.plot(x_data[:250], normal_data[:250], c="r", lw=2.0, alpha=0.5)
        plt.plot(x_data[:250], MicMent_uncomp_inhib[:250], c="b", lw=2.0, alpha=0.5)
        plt.xlabel("Concentration de sustrato (uM)")
        plt.ylabel("Tasa de Reaccion\n (uM/min)")
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

        plt.tight_layout()
        
    inhibitor_slider = widgets.FloatSlider(value=4500, min=0, max=50000, step=500, continuous_update=True, description="Concentración de Inhibidor (uM):", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    kiu_slider = widgets.FloatSlider(value=13000, min=500, max=20000, step=500, continuous_update=True, description="K_iu:", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    enzyme_dropdown = widgets.Dropdown(options=[round(num, 8) for num in list(np.hstack((np.linspace((4.3E-3/10000), (4.3E-3/1500), 50), 4.3E-3/250, np.linspace((4.3E-3/1000), 4.3E-3, 50))))], value=4.3E-3/250, description='Concentracion de enzima (uM):', disabled=False, layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})

    output_graph = widgets.interactive_output(update, {'a':inhibitor_slider, 'b':kiu_slider, 'c':enzyme_dropdown})
    header = widgets.HTML(value="<H1><center>Gráfico Interactivo de Inhibidores Acompetitivos</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    footer = widgets.HTML(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    equation = widgets.HTMLMath(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    sliders = widgets.VBox([inhibitor_slider, kiu_slider, enzyme_dropdown])

    final_output = widgets.AppLayout(header=header,
            left_sidebar=sliders,
            center=output_graph,
            footer=None,
            right_sidebar=None,
            align_items='center',
            pane_widths=[1, 1, 0],
            pane_heights=[1, 7, '100px'])

    return final_output

def find_kcat():
    
    initial_enzyme = float(input("Por favor ingrese su concentración inicial de enzima: \n"))
    fitted_v_max = float(input("Por favor ingrese su Vmax que obtuvo en el ajuste: \n"))
    
    k_cat = round(fitted_v_max/initial_enzyme, 4)
    
    print("La constante catalítica (k_cat) calculada es: \n", k_cat)
    
    return None

def find_x_intercept():
    
    a = float(input("Por favor ingrese Km/Vmax calculado anteriormente: \n"))
    b = float(input("Por favor ingrese 1/Vmax calculado anteriormente: \n"))
    
    intercepto_x = -(1/((1/b)*(a)))
    
    print("El intercepto en el eje X (-1/Km) es: \n", intercepto_x)
    
def interactive_comp_inhibitor_graph2(substrate_range, comp_inhib_data_func):
    
    # Para renderizar LaTeX
    # ce.typeset()
    
    # Se obtienen los datos originales
    x_data = substrate_range
    y_data = comp_inhib_data_func(x_data, 10000, 2500, 4.3E-3/250)
    
    def update(a, b, c):
        
        plt.figure(dpi=150)

        # Apparent parameters
        v_max_comp_inhibition = 8606*(c)
        km_comp_inhibition = np.multiply(np.add(np.divide(a, b), 1), 450)
        normal_data = np.multiply(np.divide(np.multiply(8606*c, x_data), np.add(450, x_data)), 60)

        MicMent_comp_inhib = np.multiply(np.divide(np.multiply(v_max_comp_inhibition, x_data), np.add(km_comp_inhibition, x_data)), 60)

        plt.scatter(np.divide(1, x_data[:]), np.divide(1, normal_data[:]), c="r", s=8.0)
        plt.scatter(np.divide(1, x_data[:]), np.divide(1, MicMent_comp_inhib[:]), c="b", s=8.0)
        plt.plot(np.divide(1, x_data[:]), np.divide(1, normal_data[:]), c="r", lw=2.0, alpha=0.5)
        plt.plot(np.divide(1, x_data[:]), np.divide(1, MicMent_comp_inhib[:]), c="b", lw=2.0, alpha=0.5)
        plt.xlabel("$\\frac{1}{[S]}$")
        plt.ylabel("$\\frac{1}{V_{app}}$")

        plt.tight_layout()
        
    inhibitor_slider = widgets.FloatSlider(value=10000, min=0, max=50000, step=500, continuous_update=True, description="Concentración de \nInhibidor (uM):", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    kic_slider = widgets.FloatSlider(value=2500, min=500, max=20000, step=500, continuous_update=True, description="K_ic:", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    enzyme_dropdown = widgets.Dropdown(options=[round(num, 8) for num in list(np.hstack((np.linspace((4.3E-3/10000), (4.3E-3/1500), 50), 4.3E-3/250, np.linspace((4.3E-3/1000), 4.3E-3, 50))))], value=4.3E-3/250, description='Concentracion de \nenzima (uM):', disabled=False, layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})

    output_graph = widgets.interactive_output(update, {'a':inhibitor_slider, 'b':kic_slider, 'c':enzyme_dropdown})
    header = widgets.HTML(value="<H1><center>Gráfico Interactivo de Inhibidores Competitivos</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    footer = widgets.HTML(value="<H1><center>A continuación se realizarán algunas actividades relacionadas al efecto que tienen los inhibidores competitivos sobre la tasa de reacción enzimática</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    equation = widgets.HTMLMath(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    sliders = widgets.VBox([inhibitor_slider, kic_slider, enzyme_dropdown])

    final_output = widgets.AppLayout(header=header,
            left_sidebar=sliders,
            center=output_graph,
            footer=footer,
            right_sidebar=None,
            align_items='center',
            pane_widths=[1, 1, 0],
            pane_heights=[1, 7, '100px'])

    return final_output

def interactive_mixed_inhibitor_graph2(substrate_range, mixed_inhib_data_func):
    
    # Para renderizar LaTeX
    # ce.typeset()
    
    # Se obtienen los datos originales
    x_data = substrate_range
    y_data = mixed_inhib_data_func(x_data, 10000, 17000, 17000, 4.3E-3/250)
    
    def update(a, b, c, d):
        
        plt.figure(dpi=150)

        # Apparent parameters
        mixed_inhibition_v_max = np.divide(np.multiply(8606, d), (np.add(1, np.divide(a, c))))
        mixed_inhibition_km = np.divide(np.multiply(np.add(1, np.divide(a, b)), 450), np.add(1, np.divide(a, c)))

        MicMent_mixed_inhib = np.multiply(np.divide(np.multiply(mixed_inhibition_v_max, x_data), np.add(mixed_inhibition_km, x_data)), 60)
        normal_data = np.multiply(np.divide(np.multiply(8606*d, x_data), np.add(450, x_data)), 60)

        plt.scatter(np.divide(1, x_data[:]), np.divide(1, normal_data[:]), c="r", s=8.0)
        plt.scatter(np.divide(1, x_data[:]), np.divide(1, MicMent_mixed_inhib[:]), c="b", s=8.0)
        plt.plot(np.divide(1, x_data[:]), np.divide(1, normal_data[:]), c="r", lw=2.0, alpha=0.5)
        plt.plot(np.divide(1, x_data[:]), np.divide(1, MicMent_mixed_inhib[:]), c="b", lw=2.0, alpha=0.5)
        plt.xlabel("$\\frac{1}{[S]}$")
        plt.ylabel("$\\frac{1}{V_{app}}$")

        plt.tight_layout()
        
    inhibitor_slider = widgets.FloatSlider(value=8500, min=0, max=50000, step=500, continuous_update=True, description="Concentración de Inhibidor (uM):", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    kic_slider = widgets.FloatSlider(value=17000, min=500, max=20000, step=500, continuous_update=True, description="K_ic:", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    kiu_slider = widgets.FloatSlider(value=17000, min=500, max=20000, step=500, continuous_update=True, description="K_iu:", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    enzyme_dropdown = widgets.Dropdown(options=[round(num, 8) for num in list(np.hstack((np.linspace((4.3E-3/10000), (4.3E-3/1500), 50), 4.3E-3/250, np.linspace((4.3E-3/1000), 4.3E-3, 50))))], value=4.3E-3/250, description='Concentracion de enzima (uM):', disabled=False, layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})

    output_graph = widgets.interactive_output(update, {'a':inhibitor_slider, 'b':kic_slider, 'c':kiu_slider, 'd':enzyme_dropdown})
    header = widgets.HTML(value="<H1><center>Gráfico Interactivo de Inhibidores Mixtos</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    footer = widgets.HTML(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    equation = widgets.HTMLMath(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    sliders = widgets.VBox([inhibitor_slider, kic_slider, kiu_slider, enzyme_dropdown])

    final_output = widgets.AppLayout(header=header,
            left_sidebar=sliders,
            center=output_graph,
            footer=None,
            right_sidebar=None,
            align_items='center',
            pane_widths=[1, 1, 0],
            pane_heights=[1, 7, '100px'])

    return final_output

def interactive_uncomp_inhibitor_graph2(substrate_range, uncomp_inhib_data_func):
    
    # Para renderizar LaTeX
    # ce.typeset()
    
    # Se obtienen los datos originales
    x_data = substrate_range
    y_data = uncomp_inhib_data_func(x_data, 10000, 13000, 4.3E-3/250)
    
    def update(a, b, c):
        
        plt.figure(dpi=150)

        # Apparent parameters
        uncomp_inhibition_v_max = np.divide(np.multiply(8606, c), np.add(np.divide(a, b), 1))
        uncomp_inhibition_km = np.divide(450, np.add(np.divide(a, b), 1))

        MicMent_uncomp_inhib = np.multiply(np.divide(np.multiply(uncomp_inhibition_v_max, x_data), np.add(uncomp_inhibition_km, x_data)), 60)
        normal_data = np.multiply(np.divide(np.multiply(8606*c, x_data), np.add(450, x_data)), 60)

        plt.scatter(np.divide(1, x_data[:]), np.divide(1, normal_data[:]), c="r", s=8.0)
        plt.scatter(np.divide(1, x_data[:]), np.divide(1, MicMent_uncomp_inhib[:]), c="b", s=8.0)
        plt.plot(np.divide(1, x_data[:]), np.divide(1, normal_data[:]), c="r", lw=2.0, alpha=0.5)
        plt.plot(np.divide(1, x_data[:]), np.divide(1, MicMent_uncomp_inhib[:]), c="b", lw=2.0, alpha=0.5)
        plt.xlabel("$\\frac{1}{[S]}$")
        plt.ylabel("$\\frac{1}{V_{app}}$")

        plt.tight_layout()
        
    inhibitor_slider = widgets.FloatSlider(value=4500, min=0, max=50000, step=500, continuous_update=True, description="Concentración de Inhibidor (uM):", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    kiu_slider = widgets.FloatSlider(value=13000, min=500, max=20000, step=500, continuous_update=True, description="K_iu:", layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})
    enzyme_dropdown = widgets.Dropdown(options=[round(num, 8) for num in list(np.hstack((np.linspace((4.3E-3/10000), (4.3E-3/1500), 50), 4.3E-3/250, np.linspace((4.3E-3/1000), 4.3E-3, 50))))], value=4.3E-3/250, description='Concentracion de enzima (uM):', disabled=False, layout=widgets.Layout(width='auto', height='auto'), style={'description_width': 'initial'})

    output_graph = widgets.interactive_output(update, {'a':inhibitor_slider, 'b':kiu_slider, 'c':enzyme_dropdown})
    header = widgets.HTML(value="<H1><center>Gráfico Interactivo de Inhibidores Acompetitivos</center></H1>", layout=widgets.Layout(width='auto', grid_area='header'))
    footer = widgets.HTML(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    equation = widgets.HTMLMath(value="", layout=widgets.Layout(width='auto', grid_area='header'))
    sliders = widgets.VBox([inhibitor_slider, kiu_slider, enzyme_dropdown])

    final_output = widgets.AppLayout(header=header,
            left_sidebar=sliders,
            center=output_graph,
            footer=None,
            right_sidebar=None,
            align_items='center',
            pane_widths=[1, 1, 0],
            pane_heights=[1, 7, '100px'])

    return final_output