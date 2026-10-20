# -*- coding: utf-8 -*-

"""
fuerzas.py

Funciones puras para calcular fuerzas opticas (gradiente, scattering,
spin-curl) a partir de los campos electricos simulados, y para
convertir esos campos a unidades SI.

No hay graficos ni prints de diagnostico aca -- eso vive en
graficos.py. Este modulo solo hace cuentas.
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
    Con foco simetrico en z, la fuerza de gradiente (impar en z) conserva su
    sentido restaurador y la de scattering (par en z) invierte su sentido.
    Las filas de los arreglos XZ de PyFocus van de -z a +z, asi que "z -> -z"
    es dar vuelta el eje 0.
    """
    return Fx[::-1, :], -Fz[::-1, :]


def fuerzas_xy(ex, ey, ez, dx, dy, dz, alpha_p, alpha_pp, eps_m, dE_dz=None):
    """
    Fuerza optica (gradiente + scattering + spin-curl) en el plano XY.

    Campos en V/m, pasos en m, alpha en m^3. Las filas tienen que ir de -y a
    +y (como devuelven pyfocus_wrapper.campo_foco e interfaz.py).

    El spin-curl necesita derivadas en z, que un solo plano no tiene:
    - dE_dz = (dEx/dz, dEy/dz, dEz/dz) en V/m^2, si se las tiene (p. ej.
      de planos en z +- dz). Es lo exacto.
    - Si dE_dz es None: dEz/dz sale de div E = 0 (exacto en un medio
      homogeneo, asi los terminos con div E se anulan como deben) y
      dEx/dz = dEy/dz = 0 (aproximacion: falta el aporte de Ez* dEx/dz).
    dz no se usa; queda en la firma por compatibilidad.

    Devuelve (Fx, Fy, Fg_x, Fg_y, Fs_x, Fs_y, Fsc_x, Fsc_y), en N.
    """
    I = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2

    # ---- GRADIENTE ----
    grad_Iy, grad_Ix = np.gradient(I, dy, dx)
    Fg_x = alpha_p * epsilon_0 * eps_m / 4 * grad_Ix
    Fg_y = alpha_p * epsilon_0 * eps_m / 4 * grad_Iy

    # ---- SCATTERING ----
    dEx_dx = np.gradient(ex, dx, axis=1)
    dEy_dx = np.gradient(ey, dx, axis=1)
    dEz_dx = np.gradient(ez, dx, axis=1)

    dEx_dy = np.gradient(ex, dy, axis=0)
    dEy_dy = np.gradient(ey, dy, axis=0)
    dEz_dy = np.gradient(ez, dy, axis=0)

    if dE_dz is None:
        dEx_dz = np.zeros_like(ex)
        dEy_dz = np.zeros_like(ey)
        dEz_dz = -(dEx_dx + dEy_dy)  # div E = 0
    else:
        dEx_dz, dEy_dz, dEz_dz = dE_dz

    Sx = np.imag(np.conj(ex) * dEx_dx + np.conj(ey) * dEy_dx + np.conj(ez) * dEz_dx)
    Sy = np.imag(np.conj(ex) * dEx_dy + np.conj(ey) * dEy_dy + np.conj(ez) * dEz_dy)
    Fs_x = epsilon_0 * eps_m * alpha_pp / 2 * Sx
    Fs_y = epsilon_0 * eps_m * alpha_pp / 2 * Sy

    # ---- SPIN CURL ----
    Fsc_x1 = (np.conj(ex) * dEx_dx + np.conj(ey) * dEx_dy + np.conj(ez) * dEx_dz)
    Fsc_x2 = ex * (np.conj(dEx_dx) + np.conj(dEy_dy) + np.conj(dEz_dz))
    Fsc_x3 = -np.conj(ex) * (dEx_dx + dEy_dy + dEz_dz)
    Fsc_x4 = -(ex * np.conj(dEx_dx) + ey * np.conj(dEx_dy) + ez * np.conj(dEx_dz))
    prefactor = -1j / 4
    Fsc_x = np.real(prefactor * epsilon_0 * eps_m * alpha_pp * (Fsc_x1 + Fsc_x2 + Fsc_x3 + Fsc_x4))

    Fsc_y1 = (np.conj(ex) * dEy_dx + np.conj(ey) * dEy_dy + np.conj(ez) * dEy_dz)
    Fsc_y2 = ey * (np.conj(dEx_dx) + np.conj(dEy_dy) + np.conj(dEz_dz))
    Fsc_y3 = -np.conj(ey) * (dEx_dx + dEy_dy + dEz_dz)
    Fsc_y4 = -(ex * np.conj(dEy_dx) + ey * np.conj(dEy_dy) + ez * np.conj(dEy_dz))
    Fsc_y = np.real(prefactor * epsilon_0 * eps_m * alpha_pp * (Fsc_y1 + Fsc_y2 + Fsc_y3 + Fsc_y4))

    # ---- TOTAL ----
    Fx = Fg_x + Fs_x + Fsc_x
    Fy = Fg_y + Fs_y + Fsc_y

    return Fx, Fy, Fg_x, Fg_y, Fs_x, Fs_y, Fsc_x, Fsc_y


def fuerzas_xz(ex, ey, ez, dx, dz, alpha_p, alpha_pp, eps_m, dE_dy=None):
    """
    Fuerza optica (gradiente + scattering + spin-curl) en el plano XZ
    (y = 0). Filas de -z a +z, columnas de -x a +x.

    El spin-curl necesita derivadas en y, que un solo plano no tiene:
    - dE_dy = (dEx/dy, dEy/dy, dEz/dy) en V/m^2, si se las tiene (p. ej.
      de planos en y = +-dy). Es lo exacto.
    - Si dE_dy es None: dEy/dy sale de div E = 0 (exacto en un medio
      homogeneo) y dEx/dy = dEz/dy = 0. Para un haz polarizado en x
      (Ey = 0 en y = 0) esto es exacto; en general falta el aporte de
      Ey* dE/dy.

    Devuelve (Fx, Fz, Fg_x, Fg_z, Fs_x, Fs_z, Fsc_x, Fsc_z), en N.
    """

    I = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2

    # ---- GRADIENTE ----
    grad_Iz, grad_Ix = np.gradient(I, dz, dx)
    Fg_x = alpha_p * epsilon_0 * eps_m / 4 * grad_Ix
    Fg_z = alpha_p * epsilon_0 * eps_m / 4 * grad_Iz

    # ---- SCATTERING ----
    dEx_dx = np.gradient(ex, dx, axis=1)
    dEy_dx = np.gradient(ey, dx, axis=1)
    dEz_dx = np.gradient(ez, dx, axis=1)

    dEx_dz = np.gradient(ex, dz, axis=0)
    dEy_dz = np.gradient(ey, dz, axis=0)
    dEz_dz = np.gradient(ez, dz, axis=0)

    if dE_dy is None:
        dEx_dy = np.zeros_like(ex)
        dEz_dy = np.zeros_like(ez)
        dEy_dy = -(dEx_dx + dEz_dz)  # div E = 0
    else:
        dEx_dy, dEy_dy, dEz_dy = dE_dy

    Sx = np.imag(np.conj(ex) * dEx_dx + np.conj(ey) * dEy_dx + np.conj(ez) * dEz_dx)
    Sz = np.imag(np.conj(ex) * dEx_dz + np.conj(ey) * dEy_dz + np.conj(ez) * dEz_dz)
    Fs_x = epsilon_0 * eps_m * alpha_pp / 2 * Sx
    Fs_z = epsilon_0 * eps_m * alpha_pp / 2 * Sz

    # ---- SPIN CURL ----
    Fsc_x1 = (np.conj(ex) * dEx_dx + np.conj(ey) * dEx_dy + np.conj(ez) * dEx_dz)
    Fsc_x2 = ex * (np.conj(dEx_dx) + np.conj(dEy_dy) + np.conj(dEz_dz))
    Fsc_x3 = -np.conj(ex) * (dEx_dx + dEy_dy + dEz_dz)
    Fsc_x4 = -(ex * np.conj(dEx_dx) + ey * np.conj(dEx_dy) + ez * np.conj(dEx_dz))
    prefactor = -1j / 4
    Fsc_x = np.real(prefactor * epsilon_0 * eps_m * alpha_pp * (Fsc_x1 + Fsc_x2 + Fsc_x3 + Fsc_x4))

    Fsc_z1 = (np.conj(ex) * dEz_dx + np.conj(ey) * dEz_dy + np.conj(ez) * dEz_dz)
    Fsc_z2 = ez * (np.conj(dEx_dx) + np.conj(dEy_dy) + np.conj(dEz_dz))
    Fsc_z3 = -np.conj(ez) * (dEx_dx + dEy_dy + dEz_dz)
    Fsc_z4 = -(ex * np.conj(dEz_dx) + ey * np.conj(dEz_dy) + ez * np.conj(dEz_dz))
    Fsc_z = np.real(prefactor * epsilon_0 * eps_m * alpha_pp * (Fsc_z1 + Fsc_z2 + Fsc_z3 + Fsc_z4))

    # ---- TOTAL ----
    Fx = Fg_x + Fs_x + Fsc_x
    Fz = Fg_z + Fs_z + Fsc_z

    return Fx, Fz, Fg_x, Fg_z, Fs_x, Fs_z, Fsc_x, Fsc_z
