# -*- coding: utf-8 -*-
"""
dona.py

Haz dona / vortice (Laguerre-Gaussiano, contrapropagante, con mascara
de fase). Los parametros propios de este haz (NA, h, w0, wavelength,
polarizacion) se definen aca. Los parametros compartidos con el resto
de la simulacion (I0, resolucion, rango) vienen de config.py.
"""
from PyFocus import sim

from sim_config import x_steps, z_steps, x_range, z_range, cargar_calibracion
from pyfocus_wrapper import campo_foco

# --- Parametros propios de este haz ---
NA = 1.3              # apertura numerica
n = 1.51822           # indice de refraccion del medio: agua= 1.33 , aire= 1.00, aceite=1.53
h = 2.6               # radio de apertura del objetivo (mm)
w0 = 5                # radio del haz gaussiano de entrada (mm)
wavelength = 532      # longitud de onda (nm)
gamma = 45            # polarizacion: 45 + beta=90 = circular derecha
beta = 90
z = 0                 # plano focal (z=0)
I0 = 10               # intensidad inicial de este haz (mW/cm^2)
L = ''                # distancia de propagacion del haz antes del objetivo (m)
R = ''                # radio de la mascara de fase (mm)
ds = ''               # desplazamiento off-axis del haz (mm)
z_int = ''            # posicion de interfaz dielectrica (nm)
figure_name = 'dona'

# w0 e I0 son los del BFP (antes del objetivo). Si hay calibracion.json (ver
# calibracion_foco.py) se usan esos, calculados desde lo medido en el foco.
_cal = cargar_calibracion('dona')
if _cal is not None:
    w0, I0 = _cal['w0_mm'], _cal['I0_mWcm2']

parameters = [NA, n, h, w0, wavelength, gamma, beta, z,
              x_steps, z_steps, x_range, z_range, I0,
              L, R, ds, z_int, figure_name]


def get_field():
    # campo_foco aplica la correccion de lambda/n de PyFocus (si hace falta) y no
    # abre las figuras propias de PyFocus. Ver pyfocus_wrapper.py.
    return campo_foco(sim.VP, NA, n, h, w0, wavelength, gamma, beta, I0,
                      x_steps, z_steps, x_range, z_range, z=z, nombre=figure_name)
