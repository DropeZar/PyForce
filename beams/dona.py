# -*- coding: utf-8 -*-
"""
dona.py

Haz dona / vortice (Laguerre-Gaussiano, contrapropagante, con mascara
de fase). Los parametros propios de este haz (NA, h, w0, wavelength,
polarizacion) se definen aca. Los parametros compartidos con el resto
de la simulacion (I0, resolucion, rango) vienen de config.py.
"""
from PyFocus import sim

from sim_config import x_steps, z_steps, x_range, z_range

# --- Parametros propios de este haz ---
NA = 1.3             # apertura numerica
n = 1.52              # indice de refraccion del medio
h = 3.2               # radio de apertura del objetivo (mm)
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

parameters = [NA, n, h, w0, wavelength, gamma, beta, z,
              x_steps, z_steps, x_range, z_range, I0,
              L, R, ds, z_int, figure_name]


def get_field():
    fields = sim.VP(False, False, *parameters)
    ex_XZ, ey_XZ, ez_XZ, ex_XY, ey_XY, ez_XY = fields
    return {
        "XZ": (ex_XZ, ey_XZ, ez_XZ),
        "XY": (ex_XY, ey_XY, ez_XY),
    }
