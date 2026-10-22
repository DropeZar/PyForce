# -*- coding: utf-8 -*-
"""
main.py

La simulacion completa:
  1. Calcula los campos de cada haz (gauss.py, dona.py)
  2. Calcula la polarizabilidad (Mie, dipolo a1) para cada longitud de onda (particula.py)
  3. Convierte los campos simulados a unidades SI (fuerzas.calcular_factor_E_foco)
  4. Calcula las fuerzas opticas XY y XZ (fuerzas.py)
  5. Grafica los mapas de fuerza esenciales (graficos.py)

Los parametros compartidos (radio de particula, medios, resolucion) se
definen en sim_config.py; los de cada haz, en beams/gauss.py y beams/dona.py.

Esta pensado para correrse celda por celda (#%%) en Spyder/VS Code, o
entero con `python main.py`.
"""

import numpy as np
from sim_config import eps_m, dx_fuerzas, dy_fuerzas, dz_fuerzas, rango_xy, rango_xz
from beams import gauss, dona
from particula import calc_alpha
from fuerzas import calcular_factor_E_foco, espejar_haz_contrapropagante, fuerzas_xy, fuerzas_xz
from graficos import (graficar_fuerza_total, graficar_comparacion,
                      graficar_componentes, graficar_intensidad)

# ============================================================
# 1) Campos de cada haz
# ============================================================
fields_g = gauss.get_field()
fields_d = dona.get_field()

Ex_g_xy, Ey_g_xy, Ez_g_xy = fields_g["XY"]
Ex_d_xy, Ey_d_xy, Ez_d_xy = fields_d["XY"]
Ex_g_xz, Ey_g_xz, Ez_g_xz = fields_g["XZ"]
Ex_d_xz, Ey_d_xz, Ez_d_xz = fields_d["XZ"]

# ============================================================
# 2) Polarizabilidad (Mie, dipolo a1) a la longitud de onda de cada haz
# ============================================================
alpha_p_g, alpha_pp_g = calc_alpha(gauss.wavelength)
alpha_p_d, alpha_pp_d = calc_alpha(dona.wavelength)

# ============================================================
# 3) Conversion de campos a unidades SI
# ============================================================
# CORREGIDO: el factor depende del indice del medio del FOCO de cada haz
# (PyFocus no incluye el sqrt(n1/n2) de Richards-Wolf; con n_entrada=1 la
# potencia en el foco salia n veces la real). Ver pyfocus_wrapper.factor_E_foco.
factor_E_g = calcular_factor_E_foco(gauss.n)
factor_E_d = calcular_factor_E_foco(dona.n)

Ex_g_xy, Ey_g_xy, Ez_g_xy = Ex_g_xy * factor_E_g, Ey_g_xy * factor_E_g, Ez_g_xy * factor_E_g
Ex_g_xz, Ey_g_xz, Ez_g_xz = Ex_g_xz * factor_E_g, Ey_g_xz * factor_E_g, Ez_g_xz * factor_E_g
Ex_d_xy, Ey_d_xy, Ez_d_xy = Ex_d_xy * factor_E_d, Ey_d_xy * factor_E_d, Ez_d_xy * factor_E_d
Ex_d_xz, Ey_d_xz, Ez_d_xz = Ex_d_xz * factor_E_d, Ey_d_xz * factor_E_d, Ez_d_xz * factor_E_d

# ============================================================
# 4) Fuerzas opticas
# ============================================================
# --- XY ---
# Cada funcion devuelve (total, conservativa, no conservativa) en las dos
# direcciones del plano.
Fx_g, Fy_g, Fc_x_g, Fc_y_g, Fnc_x_g, Fnc_y_g = fuerzas_xy(
    Ex_g_xy, Ey_g_xy, Ez_g_xy, dx_fuerzas, dy_fuerzas,
    alpha_p_g, alpha_pp_g, eps_m)

Fx_d, Fy_d, Fc_x_d, Fc_y_d, Fnc_x_d, Fnc_y_d = fuerzas_xy(
    Ex_d_xy, Ey_d_xy, Ez_d_xy, dx_fuerzas, dy_fuerzas,
    alpha_p_d, alpha_pp_d, eps_m)

Fx_tot = Fx_g + Fx_d
Fy_tot = Fy_g + Fy_d

# --- XZ ---
Fx_g_xz, Fz_g_xz, Fc_x_g_xz, Fc_z_g_xz, Fnc_x_g_xz, Fnc_z_g_xz = fuerzas_xz(
    Ex_g_xz, Ey_g_xz, Ez_g_xz, dx_fuerzas, dz_fuerzas, alpha_p_g, alpha_pp_g, eps_m)

Fx_d_xz, Fz_d_xz, Fc_x_d_xz, Fc_z_d_xz, Fnc_x_d_xz, Fnc_z_d_xz = fuerzas_xz(
    Ex_d_xz, Ey_d_xz, Ez_d_xz, dx_fuerzas, dz_fuerzas, alpha_p_d, alpha_pp_d, eps_m)

a = 1  # peso relativo del haz dona al contrapropagar
# CORREGIDO: el haz dona viaja hacia -z, asi que su fuerza es la imagen
# especular (z -> -z) de la que se simulo. Antes: Fz_g - a*Fz_d, que invierte
# bien la fuerza no conservativa (par en z) pero tambien la de gradiente
# (impar en z), y los gradientes de los dos haces se RESTABAN en vez de sumarse.
Fx_d_xz_cp, Fz_d_xz_cp = espejar_haz_contrapropagante(Fx_d_xz, Fz_d_xz)
Fx_tot_xz = Fx_g_xz + a * Fx_d_xz_cp
Fz_tot_xz = Fz_g_xz + a * Fz_d_xz_cp
#%%
# ============================================================
# 5) Graficos esenciales
# ============================================================
# Las filas de los arreglos XZ van de -z a +z y se grafican con
# origin='lower', asi que +z queda hacia arriba y Fz se pasa tal cual.
graficar_fuerza_total(Fx_tot, Fy_tot, rango_xy, 'Fuerza total (XY)')

graficar_comparacion(
    [(Fx_g, Fy_g, "Gaussiano XY"),
     (Fx_d, Fy_d, "Dona XY"),
     (Fx_tot, Fy_tot, "Resultante XY")],
    rango=rango_xy,
)

graficar_fuerza_total(Fx_tot_xz, Fz_tot_xz, rango_xz, 'Fuerza total (XZ)', eje_y='z (nm)')

graficar_comparacion(
    [(Fx_g_xz, Fz_g_xz, "Gaussiano XZ"),
     (a * Fx_d_xz_cp, a * Fz_d_xz_cp, "Dona XZ"),
     (Fx_tot_xz, Fz_tot_xz, "Resultante XZ")],
    rango=rango_xz, eje_y='z (nm)',
)

graficar_componentes(
    [(Fc_x_d_xz, Fc_z_d_xz, 'F conservativa (gradiente)'),
     (Fnc_x_d_xz, Fnc_z_d_xz, 'F no conservativa')],
    rango=rango_xz,
    titulo_general='Componentes de fuerza - haz Dona (XZ)',
)

graficar_componentes(
    [(Fc_x_d, Fc_y_d, 'F conservativa (gradiente)'),
     (Fnc_x_d, Fnc_y_d, 'F no conservativa')],
    rango=rango_xy, eje_y='y (nm)',
    titulo_general='Componentes de fuerza - haz Dona (XY)',
)
#%%
# n del medio del foco de CADA haz (el que usa PyFocus), no n_g para los dos
graficar_intensidad(Ex_g_xy, Ey_g_xy, Ez_g_xy, rango_xy, gauss.n, 'Gaussiano (XY)')
graficar_intensidad(Ex_d_xy, Ey_d_xy, Ez_d_xy, rango_xy, dona.n, 'Dona (XY)')
#%%
graficar_intensidad(Ex_g_xz, Ey_g_xz, Ez_g_xz, rango_xz, gauss.n, 'Gaussiano (XZ)', eje_y='z (nm)')
graficar_intensidad(Ex_d_xz, Ey_d_xz, Ez_d_xz, rango_xz, dona.n, 'Dona (XZ)', eje_y='z (nm)')
#%%
# Guardado de campos de la dona (SI, post factor_E) para reusar en otros scripts
np.savez('campos_dona_aceite.npz',
         Ex_xy=Ex_d_xy, Ey_xy=Ey_d_xy, Ez_xy=Ez_d_xy,
         Ex_xz=Ex_d_xz, Ey_xz=Ey_d_xz, Ez_xz=Ez_d_xz,
         rango=rango_xy[0], wavelength=dona.wavelength, n_medio=dona.n)

np.savez('campos_gauss_agua.npz',
         Ex_xy=Ex_g_xy, Ey_xy=Ey_g_xy, Ez_xy=Ez_g_xy,
         Ex_xz=Ex_g_xz, Ey_xz=Ey_g_xz, Ez_xz=Ez_g_xz,
         rango=rango_xy[0], wavelength=gauss.wavelength, n_medio=gauss.n)
