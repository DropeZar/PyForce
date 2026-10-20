# -*- coding: utf-8 -*-
"""
validar_interfaz.py

Chequeos de interfaz.py:

  1) Sin interfaz tiene que dar lo mismo que PyFocus (sim.no_mask, via
     pyfocus_wrapper.campo_foco). Para comparar punto a punto se evalua
     interfaz.py en los mismos puntos que usa PyFocus, que no son
     exactamente los de la grilla nominal:
       - XY: de -(N/2-1) a N/2 pixeles en y pero de
         -N/2 a N/2-1 en x; ademas cada pixel toma el radio redondeado al
         paso radial dr = x_range/sqrt(2)/nr (cart2pol(round_r=True)).
       - XZ: x = indice * dr (no x_steps).
     campo_foco ya devuelve el XY con filas de -y a +y y Ez con el signo
     fisico.
  2) Con interfaz, condiciones de borde en z = z_int: Ex y Ey continuos y
     n1^2 Ez1 = n2^2 Ez2 (D normal continuo).
  3) div E = 0 a los dos lados de la interfaz (diferencias finitas). Con
     el Ez tal como lo devuelve PyFocus (signo opuesto) daria ~2.

  4) fuerzas.py: gradiente + scattering (Poynting) + spin-curl tiene que
     dar la fuerza dipolar exacta (eps/2) Re[alpha* E_j* grad E_j], con las
     derivadas fuera del plano exactas, para polarizacion lineal y circular.

Resultado esperado: errores ~1e-13 en (1), ~1e-8 en (2) (por evaluar a
z_int -+ 1e-6 nm), ~1e-6 en (3) y ~1e-3 en (4) (diferencias finitas).

Uso:  python validar_interfaz.py
"""
import numpy as np
from PyFocus import sim

from beams import gauss
from interfaz import FocoConInterfaz, ejes_xy
from pyfocus_wrapper import campo_foco


def _err(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def comparar_con_pyfocus(x_steps=10, z_steps=10, x_range=1200, z_range=1200):
    pf = campo_foco(sim.no_mask, gauss.NA, gauss.n, gauss.h, gauss.w0, gauss.wavelength,
                    gauss.gamma, gauss.beta, gauss.I0, x_steps, z_steps, x_range, z_range)
    foco = FocoConInterfaz(gauss.NA, gauss.n, gauss.h, gauss.w0, gauss.wavelength,
                           gauss.gamma, gauss.beta, gauss.I0)
    nr = int(np.rint(x_range / x_steps / 2 ** 0.5))
    dr = x_range / 2 ** 0.5 / nr

    # XY en los puntos de PyFocus
    x = ejes_xy(x_steps, x_range)
    X, Y = np.meshgrid(x, x + x_steps)
    rho = np.rint(np.hypot(X, Y) / x_steps) * dr
    E = foco._campo(rho.ravel(), np.arctan2(Y, X).ravel(), [0.0], ('inc',))
    xy = [e[:, 0].reshape(X.shape) for e in E]
    pf_xy = pf['XY']

    # XZ en los puntos de PyFocus
    nx = pf['XZ'][0].shape[1]
    x_xz = (np.arange(nx) - (nr - 1)) * dr
    nz = pf['XZ'][0].shape[0]
    z_xz = (np.arange(nz) - nz // 2) * (z_range / 2 / int(np.rint(z_range / z_steps / 2)))
    xz = foco.plano_xz(x_xz, z_xz)

    print("1) Sin interfaz vs PyFocus (error maximo / maximo de cada componente)")
    for plano, a, b in (('XY', xy, pf_xy), ('XZ', xz, pf['XZ'])):
        I_a = sum(np.abs(e) ** 2 for e in a)
        I_b = sum(np.abs(e) ** 2 for e in b)
        ey = _err(a[1], b[1]) if np.abs(b[1]).max() > 0 else np.abs(a[1]).max()
        print(f"   {plano}: |E|^2 {_err(I_a, I_b):.1e}   Ex {_err(a[0], b[0]):.1e}   "
              f"Ey {ey:.1e}   Ez {_err(a[2], b[2]):.1e}")


def condiciones_de_borde(n2=1.52, z_int=300.0):
    foco = FocoConInterfaz(gauss.NA, gauss.n, gauss.h, gauss.w0, gauss.wavelength,
                           45, 30, gauss.I0, n2=n2, z_int=z_int)  # polarizacion general
    x = np.linspace(-800, 800, 41)
    X, Y = np.meshgrid(x, x * 0.7)
    eps = 1e-6
    E1 = foco.campo_puntos(X, Y, z_int - eps)
    E2 = foco.campo_puntos(X, Y, z_int + eps)
    print(f"2) Condiciones de borde en z_int = {z_int:.0f} nm (n1 = {gauss.n}, n2 = {n2})")
    print(f"   Ex {_err(E1[0], E2[0]):.1e}   Ey {_err(E1[1], E2[1]):.1e}   "
          f"n^2 Ez {_err(gauss.n ** 2 * E1[2], n2 ** 2 * E2[2]):.1e}")


def divergencia(n2=1.52, z_int=300.0, h=0.5):
    foco = FocoConInterfaz(gauss.NA, gauss.n, gauss.h, gauss.w0, gauss.wavelength,
                           45, 30, gauss.I0, n2=n2, z_int=z_int)
    x = np.linspace(-400, 400, 33)
    X, Y = np.meshgrid(x, x * 0.8 + 13)
    print(f"3) div E / max|dEx/dx| (interfaz en z = {z_int:.0f} nm)")
    for z in (z_int - 200, z_int + 200):
        E = lambda dx, dy, dz: foco.campo_puntos(X + dx, Y + dy, z + dz)
        dEx = (E(h, 0, 0)[0] - E(-h, 0, 0)[0]) / (2 * h)
        dEy = (E(0, h, 0)[1] - E(0, -h, 0)[1]) / (2 * h)
        dEz = (E(0, 0, h)[2] - E(0, 0, -h)[2]) / (2 * h)
        print(f"   z = {z:.0f} nm: {np.abs(dEx + dEy + dEz).max() / np.abs(dEx).max():.1e}"
              f"   (con -Ez: {np.abs(dEx + dEy - dEz).max() / np.abs(dEx).max():.1f})")


def fuerza_dipolar(n2=1.52, z_int=300.0):
    from scipy.constants import epsilon_0
    from fuerzas import calcular_factor_E_foco, fuerzas_xz
    from interfaz import ejes_xz
    from particula import calc_alpha
    from sim_config import eps_m
    ap, app = calc_alpha(gauss.wavelength)
    fE = calcular_factor_E_foco(gauss.n)
    x, z = ejes_xz(5, 5, 1200, 1200)
    d = 5e-9
    print("4) gradiente + Poynting + spin-curl vs fuerza dipolar exacta (plano XZ)")
    for pol in ((0, 0), (45, 90)):
        foco = FocoConInterfaz(gauss.NA, gauss.n, gauss.h, gauss.w0, gauss.wavelength,
                               *pol, gauss.I0, n2=n2, z_int=z_int)
        E = [e * fE for e in foco.plano_xz(x, z)]
        Dy = [e * fE * 1e9 for e in foco.derivada_xz_y(x, z)]
        Fx, Fz, *_ = fuerzas_xz(*E, d, d, ap, app, eps_m, dE_dy=Dy)
        Dx = [np.gradient(e, d, axis=1) for e in E]
        Dz = [np.gradient(e, d, axis=0) for e in E]
        a = ap + 1j * app
        Fx_ex = epsilon_0 * eps_m / 2 * np.real(np.conj(a) * sum(np.conj(e) * de for e, de in zip(E, Dx)))
        Fz_ex = epsilon_0 * eps_m / 2 * np.real(np.conj(a) * sum(np.conj(e) * de for e, de in zip(E, Dz)))
        ok = (z < z_int - 10)[:, None] & (np.abs(x) < 800)[None, :]  # lejos de la interfaz y los bordes
        ok[:2], ok[-2:] = False, False
        err = max(np.abs(Fx - Fx_ex)[ok].max(), np.abs(Fz - Fz_ex)[ok].max()) / np.abs(Fz_ex[ok]).max()
        print(f"   polarizacion gamma={pol[0]}, beta={pol[1]}: {err:.1e}")


if __name__ == '__main__':
    comparar_con_pyfocus()
    condiciones_de_borde()
    divergencia()
    fuerza_dipolar()
