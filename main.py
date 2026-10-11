# -*- coding: utf-8 -*-
"""
main.py

La simulacion completa:
  1. Calcula los campos de cada haz (gauss.py, dona.py)
  2. Calcula la polarizabilidad de Kuwata para cada longitud de onda (particula.py)
  3. Convierte los campos simulados a unidades SI (fuerzas.calcular_factor_E)
  4. Calcula las fuerzas opticas XY y XZ (fuerzas.py)
  5. Grafica los mapas de fuerza esenciales (graficos.py)

Los parametros compartidos (radio de particula, medios, resolucion,
I0, etc.) se definen todos en config.py.
"""

import numpy as np
from sim_config import eps_m, dx_fuerzas, dy_fuerzas, dz_fuerzas
from beams import gauss, dona
from particula import calc_alpha
from fuerzas import calcular_factor_E_foco, espejar_haz_contrapropagante, fuerzas_xy, fuerzas_xz
from graficos import graficar_fuerza_total, graficar_comparacion, graficar_componentes, graficar_componentes_XY

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
# 2) Polarizabilidad (Kuwata) a la longitud de onda de cada haz
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
Fx_g, Fy_g, Fg_x_g, Fg_y_g, Fs_x_g, Fs_y_g, Fsc_x_g, Fsc_y_g = fuerzas_xy(
    Ex_g_xy, Ey_g_xy, Ez_g_xy, dx_fuerzas, dy_fuerzas, dz_fuerzas,
    alpha_p_g, alpha_pp_g, eps_m)

Fx_d, Fy_d, Fg_x_d, Fg_y_d, Fs_x_d, Fs_y_d, Fsc_x_d, Fsc_y_d = fuerzas_xy(
    Ex_d_xy, Ey_d_xy, Ez_d_xy, dx_fuerzas, dy_fuerzas, dz_fuerzas,
    alpha_p_d, alpha_pp_d, eps_m)

Fx_tot = Fx_g + Fx_d
Fy_tot = Fy_g + Fy_d

# --- XZ ---
Fx_g_xz, Fz_g_xz, Fg_x_g_xz, Fg_z_g_xz, Fs_x_g_xz, Fs_z_g_xz, Fsc_x_g_xz, Fsc_z_g_xz = fuerzas_xz(
    Ex_g_xz, Ey_g_xz, Ez_g_xz, dx_fuerzas, dz_fuerzas, alpha_p_g, alpha_pp_g, eps_m)

Fx_d_xz, Fz_d_xz, Fg_x_d_xz, Fg_z_d_xz, Fs_x_d_xz, Fs_z_d_xz, Fsc_x_d_xz, Fsc_z_d_xz = fuerzas_xz(
    Ex_d_xz, Ey_d_xz, Ez_d_xz, dx_fuerzas, dz_fuerzas, alpha_p_d, alpha_pp_d, eps_m)

a = 1  # peso relativo del haz dona al contrapropagar
# CORREGIDO: el haz dona viaja hacia -z, asi que su fuerza es la imagen
# especular (z -> -z) de la que se simulo. Antes: Fz_g - a*Fz_d, que invierte
# bien el scattering (par en z) pero tambien la fuerza de gradiente (impar en
# z), y los gradientes de los dos haces se RESTABAN en vez de sumarse.
Fx_d_xz_cp, Fz_d_xz_cp = espejar_haz_contrapropagante(Fx_d_xz, Fz_d_xz)
Fx_tot_xz = Fx_g_xz + a * Fx_d_xz_cp
Fz_tot_xz = Fz_g_xz + a * Fz_d_xz_cp
#%%
# ============================================================
# 5) Graficos esenciales
# ============================================================
# Nota:mismo rango que en sim_config!.
rango_grafico = 1200

graficar_fuerza_total(Fx_tot, Fy_tot, rango_grafico, 'Fuerza total (XY)')

graficar_comparacion(
    [(Fx_g, Fy_g, "Gaussiano XY"),
     (Fx_d, Fy_d, "Dona XY"),
     (Fx_tot, Fy_tot, "Resultante XY")],
    rango=rango_grafico,
)

graficar_fuerza_total(Fx_tot_xz, -Fz_tot_xz, rango_grafico, 'Fuerza total (XZ)')

graficar_comparacion(
    [(Fx_g_xz, -Fz_g_xz, "Gaussiano XZ"),
     (a * Fx_d_xz_cp, -a * Fz_d_xz_cp, "Dona XZ"),
     (Fx_tot_xz, -Fz_tot_xz, "Resultante XZ")],
    rango=rango_grafico,
)

graficar_componentes(
    [(Fg_x_d_xz, Fg_z_d_xz, 'F gradiente'),
     (Fs_x_d_xz, Fs_z_d_xz, 'F scattering'),
     (Fsc_x_d_xz, Fsc_z_d_xz, 'F spin-curl')],
    rango=rango_grafico,
    titulo_general='Componentes de fuerza - haz Dona (XZ)',
)

graficar_componentes_XY(
    [(Fg_x_d, Fg_y_d, 'F gradiente'),
     (Fs_x_d, Fs_y_d, 'F scattering'),
     (Fsc_x_d, Fsc_y_d, 'F spin-curl')],
    rango=rango_grafico,
    titulo_general='Componentes de fuerza - haz Dona (XY)',
)
#%%
from graficos import graficar_intensidad, graficar_fuerza_total, graficar_comparacion, graficar_componentes
# n del medio del foco de CADA haz (el que usa PyFocus), no n_g para los dos
graficar_intensidad(Ex_g_xy, Ey_g_xy, Ez_g_xy, rango_grafico, gauss.n, 'Gaussiano (XY)')
graficar_intensidad(Ex_d_xy, Ey_d_xy, Ez_d_xy, rango_grafico, dona.n, 'Dona (XY)')
#%%
graficar_intensidad(Ex_g_xz, Ey_g_xz, Ez_g_xz, rango_grafico, gauss.n, 'Gaussiano (XZ)')
graficar_intensidad(Ex_d_xz, Ey_d_xz, Ez_d_xz, rango_grafico, dona.n, 'Dona (XZ)')
#%%
# Guardado de campos de la dona (SI, post factor_E) para reusar en otros scripts
np.savez('campos_dona_aceite.npz',
         Ex_xy=Ex_d_xy, Ey_xy=Ey_d_xy, Ez_xy=Ez_d_xy,
         Ex_xz=Ex_d_xz, Ey_xz=Ey_d_xz, Ez_xz=Ez_d_xz,
         rango=rango_grafico, wavelength=dona.wavelength, n_medio=dona.n)

np.savez('campos_gauss_agua.npz',
         Ex_xy=Ex_g_xy, Ey_xy=Ey_g_xy, Ez_xy=Ez_g_xy,
         Ex_xz=Ex_g_xz, Ey_xz=Ey_g_xz, Ez_xz=Ez_g_xz,
         rango=rango_grafico, wavelength=gauss.wavelength, n_medio=gauss.n)

#%%
"para invovar en otro lado"
import numpy as np
data = np.load('campos_dona_aceite.npz')
Ex_d_xy, Ey_d_xy, Ez_d_xy = data['Ex_xy'], data['Ey_xy'], data['Ez_xy']
Ex_d_xz, Ey_d_xz, Ez_d_xz = data['Ex_xz'], data['Ey_xz'], data['Ez_xz']
rango_grafico = data['rango']
