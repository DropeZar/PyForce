# -*- coding: utf-8 -*-
"""
config.py

Parametros compartidos de la simulacion: geometria de la particula,
indices de refraccion, constantes fisicas y parametros comunes de
grilla/resolucion de PyFocus. El I0 (intensidad) es propio de cada haz
y se define en beams/gauss.py y beams/dona.py, no aca.

Este es el UNICO lugar donde se deben modificar estos valores.
gauss.py, dona.py, particula.py, fuerzas.py y main.py importan de aca
en vez de tener cada uno su propia copia.
"""
import json
import os

from scipy.constants import epsilon_0

# --- Constantes fisicas ---
c = 3e8  # m/s

# --- Nanoparticula ---
radio_particula = 40e-9  # m (radio de la NP de oro)

# --- Medios ---
# n_g y n_d NO son dos medios distintos donde vive la particula -- la
# nanoparticula es una sola y esta en agua para los dos haces. n_d=1.52
# es el indice de inmersion del objetivo que usa PyFocus para enfocar
# el haz dona (necesario para alcanzar NA=1.3 con un objetivo de
# inmersion en aceite); n_g=1.33 es agua, el medio real de la
# particula. Por eso eps_m (la permitividad usada para la
# polarizabilidad en particula.py) se calcula con n_g para ambos haces.
n_g = 1.33   # indice de refraccion del medio real de la particula (agua)
n_d = 1.52   # indice de inmersion del objetivo para el haz dona (no es el medio de la particula)

eps_m = n_g ** 2  # permitividad del agua -- misma particula, mismo medio, para los dos haces

# --- Parametros comunes de grilla PyFocus ---
# (compartidos por gauss.py y dona.py; los parametros propios de cada
# haz -- NA, h, w0, wavelength, polarizacion, I0 -- se quedan en cada
# archivo de haz porque son distintos entre si)
x_steps = 5      # resolucion en x (nm)
z_steps = 5      # resolucion en z (nm)
x_range = 1200      # rango en XY (nm)
z_range = 1200     # rango en Z (nm)

# --- Paso espacial para las derivadas/gradientes en fuerzas.py ---
paso_espacial_nm = 9
dx_fuerzas = paso_espacial_nm * 1e-9  # m
dy_fuerzas = dx_fuerzas
dz_fuerzas = dx_fuerzas


# --- Calibracion foco -> BFP (la escribe calibracion_foco.py) ---
# Si existe calibracion.json, beams/gauss.py y beams/dona.py usan los w0/I0 que
# tenga guardados (calculados a partir de lo que medis en el foco) en vez de los
# valores por defecto de cada archivo. Poner USAR_CALIBRACION = False para
# volver a los valores escritos a mano en los archivos de haz.
USAR_CALIBRACION = True
RUTA_CALIBRACION = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibracion.json')


def cargar_calibracion(nombre):
    """Devuelve {'w0_mm':..., 'I0_mWcm2':...,...} del haz `nombre` ('gauss'/'dona'), o None."""
    if not USAR_CALIBRACION or not os.path.exists(RUTA_CALIBRACION):
        return None
    with open(RUTA_CALIBRACION, encoding='utf-8') as fh:
        return json.load(fh).get(nombre)
