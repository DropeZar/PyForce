# -*- coding: utf-8 -*-

"""
fuerzas.py

Funciones puras para calcular la fuerza optica sobre la NP a partir de los
campos electricos simulados, y para convertir esos campos a unidades SI.

No hay graficos ni prints de diagnostico aca -- eso vive en
graficos.py. Este modulo solo hace cuentas.

La fuerza
---------
La NP se trata como un dipolo puntual, p = eps0 eps_m alpha E, con
alpha = alpha' + i alpha'' (particula.calc_alpha). La fuerza promediada en
el tiempo es (Chaumet & Nieto-Vesperinas, Opt. Lett. 25, 1065 (2000))

    F_i = (eps0 eps_m / 2) Re[ alpha* sum_j E_j* d_i E_j ]

que se separa en dos terminos:

    F_i =  (eps0 eps_m alpha' / 4) d_i |E|^2                  conservativo
         + (eps0 eps_m alpha''/ 2) Im[ sum_j E_j* d_i E_j ]   no conservativo

El conservativo es la fuerza de gradiente (lleva la NP hacia donde la luz es
mas intensa si alpha' > 0). El no conservativo es el que empuja en el
sentido en que viaja la luz; en la literatura a veces se lo separa en
scattering (vector de Poynting) + spin-curl, pero la suma es esta.

La componente i de la fuerza solo necesita derivadas en la direccion i: en
el plano XZ, Fx y Fz salen enteras de derivadas en x y en z (no hacen falta
derivadas en y), y en el plano XY, Fx y Fy de derivadas en x e y.
"""
import numpy as np
from scipy.constants import epsilon_0


def calcular_factor_E_foco(n_foco):
    """
    Factor (constante) para pasar los campos de PyFocus a V/m, usando el
    indice del medio del foco. PyFocus usa I0 literalmente como |E0|^2 y no
    incluye el sqrt(n1/n2) de Richards-Wolf; con n_entrada = 1 (version
    anterior) E salia sqrt(n_foco) veces mas grande y las fuerzas n_foco
    veces. Detalle y verificacion en pyfocus_wrapper.factor_E_foco.
    """
    from pyfocus_wrapper import factor_E_foco
    return factor_E_foco(n_foco)


def espejar_haz_contrapropagante(Fx, Fz):
    """
    Fuerzas XZ de un haz que se simulo propagando hacia +z, pasadas al haz
    IDENTICO que viaja hacia -z (imagen especular z -> -z):
        Fx'(x, z) =  Fx(x, -z)        Fz'(x, z) = -Fz(x, -z)
    Con foco simetrico en z, la fuerza conservativa (impar en z) conserva su
    sentido restaurador y la no conservativa (par en z) invierte su sentido.
    Las filas de los arreglos XZ de PyFocus van de -z a +z, asi que "z -> -z"
    es dar vuelta el eje 0.
    """
    return Fx[::-1, :], -Fz[::-1, :]


def _fuerza_en_un_plano(ex, ey, ez, d_col, d_fila, alpha_p, alpha_pp, eps_m):
    """
    Fuerza en un plano a partir de los campos en ese plano. Las columnas
    (eje 1) son la primera coordenada del plano (x) y las filas (eje 0) la
    segunda (y o z); d_col y d_fila son los pasos de la grilla en m.

    Devuelve (F_col, F_fila, Fc_col, Fc_fila, Fnc_col, Fnc_fila) en N:
    total, conservativa y no conservativa en cada direccion del plano.
    """
    campos = (ex, ey, ez)
    I = sum(np.abs(e) ** 2 for e in campos)

    # derivadas por diferencias finitas (np.gradient: centradas en el interior)
    dI_fila, dI_col = np.gradient(I, d_fila, d_col)
    dE_col = [np.gradient(e, d_col, axis=1) for e in campos]
    dE_fila = [np.gradient(e, d_fila, axis=0) for e in campos]

    # ---- CONSERVATIVA (gradiente) ----
    k_c = epsilon_0 * eps_m * alpha_p / 4
    Fc_col, Fc_fila = k_c * dI_col, k_c * dI_fila

    # ---- NO CONSERVATIVA ----
    k_nc = epsilon_0 * eps_m * alpha_pp / 2
    Fnc_col = k_nc * np.imag(sum(np.conj(e) * de for e, de in zip(campos, dE_col)))
    Fnc_fila = k_nc * np.imag(sum(np.conj(e) * de for e, de in zip(campos, dE_fila)))

    return (Fc_col + Fnc_col, Fc_fila + Fnc_fila, Fc_col, Fc_fila, Fnc_col, Fnc_fila)


def fuerzas_xy(ex, ey, ez, dx, dy, alpha_p, alpha_pp, eps_m):
    """
    Fuerza optica en el plano XY (componentes x e y). Campos en V/m, pasos en
    m, alpha en m^3. Filas de -y a +y, columnas de -x a +x (como devuelven
    pyfocus_wrapper.campo_foco y campo_foco_vidrio).

    Devuelve (Fx, Fy, Fc_x, Fc_y, Fnc_x, Fnc_y), en N: total, conservativa
    (gradiente) y no conservativa.
    """
    return _fuerza_en_un_plano(ex, ey, ez, dx, dy, alpha_p, alpha_pp, eps_m)


def fuerzas_xz(ex, ey, ez, dx, dz, alpha_p, alpha_pp, eps_m):
    """
    Fuerza optica en el plano XZ (y = 0; componentes x y z). Campos en V/m,
    pasos en m, alpha en m^3. Filas de -z a +z, columnas de -x a +x.

    Devuelve (Fx, Fz, Fc_x, Fc_z, Fnc_x, Fnc_z), en N: total, conservativa
    (gradiente) y no conservativa.
    """
    return _fuerza_en_un_plano(ex, ey, ez, dx, dz, alpha_p, alpha_pp, eps_m)
