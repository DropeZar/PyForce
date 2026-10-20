# -*- coding: utf-8 -*-
"""
gauss.py

Haz gaussiano (contrapropagante, sin mascara de fase).
Los parametros propios de este haz (NA, h, w0, wavelength, polarizacion)
se definen aca. La grilla de PyFocus (resolucion y rango) es
compartida con el otro haz y viene de sim_config.py -- si hay que
cambiarla, se cambia ahi, no aca. w0 e I0 se pisan con los de
calibracion.json si existe (ver calibracion_foco.py).
"""
from PyFocus import sim

from sim_config import x_steps, z_steps, x_range, z_range, cargar_calibracion
from pyfocus_wrapper import campo_foco

# --- Parametros propios de este haz ---
NA = 1              # apertura numerica
n = 1.35             # indice de refraccion del medio del foco (inmersion del objetivo)
h = 3             # radio de apertura del objetivo (mm)
w0 = 4          # radio del haz gaussiano (mm)
wavelength = 592     # longitud de onda (nm)
gamma = 0            # polarizacion: gamma=0, beta=0 -> lineal en x (gamma=45, beta=90 -> circular)
beta = 0
z = 0                # plano focal (z=0)
I0 = 20              # intensidad inicial de este haz (mW/cm^2)
L = ''               # distancia de propagacion del haz antes del objetivo (m)
R = ''               # radio de la mascara de fase (mm)
ds = ''              # desplazamiento off-axis del haz (mm)
z_int = ''           # posicion de interfaz dielectrica (nm)
figure_name = 'gauss'

# w0 e I0 son los del BFP (antes del objetivo). Si hay calibracion.json (ver
# calibracion_foco.py) se usan esos, calculados desde lo medido en el foco.
_cal = cargar_calibracion('gauss')
if _cal is not None:
    w0, I0 = _cal['w0_mm'], _cal['I0_mWcm2']

parameters = [NA, n, h, w0, wavelength, gamma, beta, z,
              x_steps, z_steps, x_range, z_range, I0,
              L, R, ds, z_int, figure_name]


def get_field():
    # campo_foco aplica la correccion de lambda/n de PyFocus (si hace falta) y no
    # abre las figuras propias de PyFocus. Ver pyfocus_wrapper.py.
    return campo_foco(sim.no_mask, NA, n, h, w0, wavelength, gamma, beta, I0,
                      x_steps, z_steps, x_range, z_range, z=z, nombre=figure_name)
