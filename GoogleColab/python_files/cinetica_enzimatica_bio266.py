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

def ph_data(planck_constant=6.626E-34, r_constant=8.314, boltz_const=1.380E-23, delta_g_act=50000, temperature=295, experimental_km=450, pka_e1=2.2, pka_e2=9.1, pka_ea1=6.0, pka_ea2=9.8, initial_enzyme=4.3E-3/250):

    # General pH parameters to be used
    ka_e1 = 10**(-pka_e1)
    ka_e2 = 10**(-pka_e2)
    ka_ea1 = 10**(-pka_ea1)
    ka_ea2 = 10**(-pka_ea2)
    ph_range = [round(num, 2) for num in np.linspace(1, 14, 100)]
    h_range = [10**(-num) for num in ph_range]

    # For pH-dependant Km
    ph_dependant_km_numerator = np.multiply(np.add(np.add(np.divide(h_range, ka_e1), np.divide(ka_e2, h_range)), 1), experimental_km)
    ph_dependant_km_denominator = np.add(np.add(np.divide(h_range, ka_ea1), np.divide(ka_ea2, h_range)), 1)

    ph_dependant_km = np.divide(ph_dependant_km_numerator, ph_dependant_km_denominator)

    # For pH-dependant Vmax
    ph_independant_kcat = np.exp(-(delta_g_act/(r_constant*temperature)))*((boltz_const*temperature)/(planck_constant))

    ph_dependant_vmax_numerator = ph_independant_kcat*initial_enzyme
    ph_dependant_vmax_denominator = np.add(np.add(np.divide(h_range, ka_ea1), np.divide(ka_ea2, h_range)), 1)

    ph_dependant_vmax = np.divide(ph_dependant_vmax_numerator, ph_dependant_vmax_denominator)

    # Substrate Range
    substrate_range = np.hstack((np.linspace(100, 1000, 5, dtype=int),np.linspace(2000, 8000, 6, dtype=int)))

    results_dict = {'ph_range':ph_range}

    for substrate_conc in substrate_range:

        data = np.multiply(np.divide(np.multiply(ph_dependant_vmax, substrate_conc), np.add(ph_dependant_km, substrate_conc)), 60) # Multiply by 60 to get micromoles/min

        gaussian_data = [np.random.normal(num, 0.25) for num in data]
        nonneg_gauss_data = [0 if i < 0 else i for i in gaussian_data]

        results_dict["ph_substrate_{}_um".format(substrate_conc)] = nonneg_gauss_data

    return results_dict

def temp_data(planck_constant=6.626E-34, r_constant=8.314, boltz_const=1.380E-23, delta_g_act=50000, delta_g_inact=94000, lower_temperature=283, upper_temperature=373, eq_temp=313, delta_h_eq=150000, experimental_km=450, initial_enzyme=4.3E-3/250):

    # General temperature parameters to be used
    temperature_range = np.linspace(lower_temperature, upper_temperature, upper_temperature-lower_temperature, dtype=int)      # Kelvin
    substrate_range = np.hstack((np.linspace(100, 1000, 5, dtype=int),np.linspace(2000, 8000, 6, dtype=int)))                     # In micromoles

    # For observed rate
    k_cat_observed = np.multiply(np.divide(np.multiply(boltz_const, temperature_range), planck_constant), np.exp(-(np.divide(delta_g_act, np.multiply(r_constant, temperature_range)))))

    # For observed inactivation rate
    k_inactivation_observed = np.multiply(np.divide(np.multiply(boltz_const, temperature_range), planck_constant), np.exp(-(np.divide(delta_g_inact, np.multiply(r_constant, temperature_range)))))

    # Equilibrium temperature
    k_temp_equilibrium = np.exp(np.multiply(np.divide(delta_h_eq, r_constant), np.subtract(np.divide(1, eq_temp), np.divide(1, temperature_range))))
    inactivation_parameter = np.exp(np.divide(np.multiply(np.multiply(k_temp_equilibrium, 1), k_inactivation_observed), np.add(1, k_temp_equilibrium)))

    # Calculation of temperature-dependant Vmax
    temperature_dependant_vmax =  np.divide(np.multiply(np.multiply(k_cat_observed, initial_enzyme), inactivation_parameter), np.add(1, k_temp_equilibrium))

    results_dict = {'temp_range':np.add(temperature_range, -273)}

    for substrate_conc in substrate_range:

        temperature_dependant_rate_numerator = np.multiply(temperature_dependant_vmax, substrate_conc)
        temperature_dependant_rate_denominator = np.add(experimental_km, substrate_conc)

        data = np.multiply(np.divide(temperature_dependant_rate_numerator, temperature_dependant_rate_denominator), 60) # To obtain micromoles/min

        gaussian_data = [np.random.normal(num, 0.25) for num in data]
        nonneg_gauss_data = [0 if i < 0 else i for i in gaussian_data]

        results_dict["temp_substrate_{}_um".format(substrate_conc)] = nonneg_gauss_data

    return results_dict

def comp_inhibitor_data(planck_constant=6.626E-34, r_constant=8.314, boltz_const=1.380E-23, delta_g_act=50000, temperature=295, k_ic=25000, lower_inhibitor_conc=20000, upper_inhibitor_conc=300000, lower_substrate_conc=0, upper_substrate_conc=10000, experimental_km=450, initial_enzyme=4.3E-3/250):

    # General competitive inhibitor parameters to be used
    inhibitor_concentration = np.linspace(lower_inhibitor_conc, upper_inhibitor_conc, 5) # If we start from 0 it'll be the same as without inhibitor
    substrate_concentration = np.linspace(lower_substrate_conc, upper_substrate_conc, (upper_substrate_conc-lower_substrate_conc)//2)

    # Apparent parameters
    k_cat_comp_inhibition = boltz_const*temperature/planck_constant*np.exp((-delta_g_act)/(r_constant*temperature))
    v_max_comp_inhibition = k_cat_comp_inhibition*initial_enzyme
    km_comp_inhibition = np.multiply(np.add(np.divide(inhibitor_concentration, k_ic), 1), experimental_km)

    results_dict = {'substrate_concentration':substrate_concentration}

    for km_app, i in zip(km_comp_inhibition, inhibitor_concentration):

        comp_inhibition_rate_numerator = np.multiply(v_max_comp_inhibition, substrate_concentration)
        comp_inhibition_rate_denominator = np.add(km_app, substrate_concentration)

        data = np.multiply(np.divide(comp_inhibition_rate_numerator, comp_inhibition_rate_denominator), 60) # To obtain micromoles/min

        gaussian_data = [np.random.normal(num, 0.5) for num in data]
        nonneg_gauss_data = [0 if j < 0 else j for j in gaussian_data]

        results_dict["comp_inhibitor_{}_um".format(i)] = nonneg_gauss_data

    return results_dict

def mixed_inhibitor_data(planck_constant=6.626E-34, r_constant=8.314, boltz_const=1.380E-23, delta_g_act=50000, temperature=295, k_ic=300000, k_iu=300000, lower_inhibitor_conc=20000, upper_inhibitor_conc=300000, lower_substrate_conc=0, upper_substrate_conc=10000, experimental_km=450, initial_enzyme=4.3E-3/250):

    # General mixed inhibitor parameters to be used
    inhibitor_concentration = np.linspace(lower_inhibitor_conc, upper_inhibitor_conc, 5) # If we start from 0 it'll be the same as without inhibitor
    substrate_concentration = np.linspace(lower_substrate_conc, upper_substrate_conc, (upper_substrate_conc-lower_substrate_conc)//2)

    k_cat_mixed_inhibition = boltz_const*temperature/planck_constant*np.exp((-delta_g_act)/(r_constant*temperature))
    mixed_inhibition_v_max = np.divide(np.multiply(k_cat_mixed_inhibition, initial_enzyme), (np.add(1, np.divide(inhibitor_concentration, k_iu))))
    mixed_inhibition_km = np.divide(np.multiply(np.add(1, np.divide(inhibitor_concentration, k_ic)), experimental_km), np.add(1, np.divide(inhibitor_concentration, k_iu)))

    results_dict = {'substrate_concentration':substrate_concentration}

    for km_app, v_max_app, i in zip(mixed_inhibition_km, mixed_inhibition_v_max, inhibitor_concentration):

        mixed_inhibition_rate_numerator = np.multiply(v_max_app, substrate_concentration)
        mixed_inhibition_rate_denominator = np.add(km_app, substrate_concentration)

        data = np.multiply(np.divide(mixed_inhibition_rate_numerator, mixed_inhibition_rate_denominator), 60) # To obtain micromoles/min

        gaussian_data = [np.random.normal(num, 0.5) for num in data]
        nonneg_gauss_data = [0 if j < 0 else j for j in gaussian_data]

        results_dict["mixed_inhibitor_{}_um".format(i)] = nonneg_gauss_data

    return results_dict

def uncomp_inhibitor_data(planck_constant=6.626E-34, r_constant=8.314, boltz_const=1.380E-23, delta_g_act=50000, temperature=295, k_iu=85000, lower_inhibitor_conc=20000, upper_inhibitor_conc=300000, lower_substrate_conc=0, upper_substrate_conc=10000, experimental_km=450, initial_enzyme=4.3E-3/250):

    # General mixed inhibitor parameters to be used
    inhibitor_concentration = np.linspace(lower_inhibitor_conc, upper_inhibitor_conc, 5) # If we start from 0 it'll be the same as without inhibitor
    substrate_concentration = np.linspace(lower_substrate_conc, upper_substrate_conc, (upper_substrate_conc-lower_substrate_conc)//2)

    k_cat_uncomp_inhibition = boltz_const*temperature/planck_constant*np.exp((-delta_g_act)/(r_constant*temperature))
    uncomp_inhibition_v_max = np.divide((k_cat_uncomp_inhibition*initial_enzyme), np.add(np.divide(inhibitor_concentration, k_iu), 1))
    uncomp_inhibition_km = np.divide(experimental_km, np.add(np.divide(inhibitor_concentration, k_iu), 1))

    results_dict = {'substrate_concentration':substrate_concentration}

    for km_app, v_max_app, i in zip(uncomp_inhibition_km, uncomp_inhibition_v_max, inhibitor_concentration):

        uncomp_inhibition_rate_numerator = np.multiply(v_max_app, substrate_concentration)
        uncomp_inhibition_rate_denominator = np.add(km_app, substrate_concentration)

        data = np.multiply(np.divide(uncomp_inhibition_rate_numerator, uncomp_inhibition_rate_denominator), 60) # To obtain micromoles/min

        gaussian_data = [np.random.normal(num, 0.5) for num in data]
        nonneg_gauss_data = [0 if j < 0 else j for j in gaussian_data]

        results_dict["uncomp_inhibitor_{}_um".format(i)] = nonneg_gauss_data

    return results_dict

def enzymatic_dilutions_data(planck_constant=6.626E-34, r_constant=8.314, boltz_const=1.380E-23, delta_g_act=50000, temperature=295, experimental_km=450, initial_enzyme=4.3E-3, dilutions_lower_limit=100, dilutions_upper_limit=1500, lower_substrate_conc=0, upper_substrate_conc=10000):

    # Enzyme dilutions parameters
    k_cat_econc_dependance = boltz_const*temperature/planck_constant*np.exp((-delta_g_act)/(r_constant*temperature))
    enzyme_dilutions = np.linspace(dilutions_lower_limit, dilutions_upper_limit, 10, dtype=int)
    diluted_enzyme_conc = np.divide(initial_enzyme, enzyme_dilutions)
    substrate_range = np.linspace(lower_substrate_conc, upper_substrate_conc, (upper_substrate_conc-lower_substrate_conc)//2)

    results_dict = {'substrate_range':substrate_range}

    for enzyme_dil, i in zip(diluted_enzyme_conc, enzyme_dilutions):

        econc_dependant_rate_numerator = np.multiply(np.multiply(k_cat_econc_dependance, enzyme_dil), substrate_range)
        econc_dependant_rate_denominator = np.add(experimental_km, substrate_range)

        data = np.multiply(np.divide(econc_dependant_rate_numerator, econc_dependant_rate_denominator), 60)

        gaussian_data = [np.random.normal(num, 0.5) for num in data]
        nonneg_gauss_data = [0 if i < 0 else i for i in gaussian_data]

        results_dict["enzyme_dil_1/{}_um".format(i)] = nonneg_gauss_data

    return results_dict

def progress_curve_data(enzyme_dilutions_data, lower_time_limit=0, upper_time_limit=10):

    # Progress curve parameters

    v_max_dilutions = [max(enzyme_dilutions_data[i]) for i in enzyme_dilutions_data.keys()]
    del v_max_dilutions[0]
    timespace = np.linspace(lower_time_limit, upper_time_limit, upper_time_limit*4)              # 10 minutes default

    results_dict = {'timespace':timespace}

    for v_max, i in zip(v_max_dilutions, range(len(enzyme_dilutions_data))):

        data = np.multiply(v_max, timespace)
        gaussian_data = [np.random.normal(num, 4.5) for num in data]
        nonneg_gauss_data = [0 if i < 0 else i for i in gaussian_data]

        results_dict["progress_{}".format(list(enzyme_dilutions_data.keys())[i+1])] = nonneg_gauss_data

    return results_dict

def gen_datos_est(funclist, filepath):

    for func in funclist:
        if func.__name__ != 'progress_curve_data':
            data = func()
        elif func.__name__ == 'progress_curve_data':
            data = func(enzymatic_dilutions_data())

        # Generate random data

        with open(filepath+"{}.csv".format(func.__name__), "w") as file:
            wr = csv.writer(file, delimiter=',')
            wr.writerow(data.keys())
            wr.writerows(zip(*data.values()))
        print('Wrote {} data to csv file!'.format(func.__name__))

    return None

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

def fit_micment_mixed_inhibitor(datos_exp_x, inhibitor_concentration, k_iu, k_ic):

    substrate_concentration = datos_exp_x
    mixed_inhibition_v_max = np.divide(np.multiply(8606, 4.3E-3/250), (np.add(1, np.divide(inhibitor_concentration, k_iu))))
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