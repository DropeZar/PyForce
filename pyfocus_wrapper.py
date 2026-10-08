# -*- coding: utf-8 -*-
"""
pyfocus_wrapper.py

Capa fina entre tu codigo y PyFocus. Hace tres cosas:

1. Corre PyFocus sin que abra figuras ni imprima barras de progreso
   (importante para el barrido de calibracion, que lo llama decenas de veces).

2. Detecta -- con una mini-simulacion de ~3 s -- si tu version de PyFocus
   divide la longitud de onda por n DOS veces (pasa en PyCustomFocus 3.4.0:
   una en main_calculation_handler.py y otra en base.py). Si pasa, el foco
   queda ~n veces mas chico que el real y las amplitudes tambien cambian.
   El remedio es pasarle lambda*n; esa correccion se aplica sola.

3. Da la conversion de campo PyFocus -> SI con la normalizacion de energia
   correcta (ver factor_E_foco).

Todo lo que importa de aca (campo_foco, factor_E_foco, ...) es lo que usan
beams/gauss.py, beams/dona.py y calibracion_foco.py.
"""
import contextlib
import io
from unittest import mock

import numpy as np
from scipy.constants import epsilon_0

from sim_config import c

_LAM_MULT = None  # cache del resultado del chequeo: 'n' (hay bug) o 1 (esta bien)


# ---------------------------------------------------------------------------
# Correr PyFocus en silencio
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _silencio():
    """Sin figuras, sin tqdm, sin logs. No toca las figuras que ya tenias abiertas."""
    import matplotlib.pyplot as plt
    antes = set(plt.get_fignums())
    try:
        with mock.patch.object(plt, 'show', lambda *a, **k: None), \
                contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        for num in set(plt.get_fignums()) - antes:
            plt.close(num)


def _correr(funcion, NA, n, h, w0, wavelength, gamma, beta, I0,
            x_steps, z_steps, x_range, z_range, z=0, nombre='pf'):
    params = [NA, n, h, w0, wavelength, gamma, beta, z,
              x_steps, z_steps, x_range, z_range, I0,
              '', '', '', '', nombre]
    with _silencio():
        return funcion(False, False, *params)


# ---------------------------------------------------------------------------
# Chequeo de la doble division lambda/n
# ---------------------------------------------------------------------------
def _fwhm_x(I, dx):
    fila = I[np.unravel_index(np.argmax(I), I.shape)[0], :]
    m = fila.max()
    idx = np.where(fila >= m / 2)[0]
    i0, i1 = idx[0], idx[-1]
    xl = i0 - 1 + (m / 2 - fila[i0 - 1]) / (fila[i0] - fila[i0 - 1])
    xr = i1 + (fila[i1] - m / 2) / (fila[i1] - fila[i1 + 1])
    return (xr - xl) * dx


def detectar_doble_division(verbose=True, forzar=False):
    """
    Compara el FWHM del foco para n=1 y n=1.5 a la MISMA NA (=0.5) y misma
    lambda. Fisicamente el spot depende de lambda/NA, no de n, asi que los
    dos anchos deben coincidir (~ +-5 %). Si el de n=1.5 sale ~1/1.5 del de
    n=1, PyFocus esta dividiendo lambda por n dos veces.

    Devuelve 'n' (hay que pasar lambda*n) o 1 (todo bien). Se cachea.
    """
    global _LAM_MULT
    if _LAM_MULT is not None and not forzar:
        return _LAM_MULT

    from PyFocus import sim
    anchos = []
    for n in (1.0, 1.5):
        Ex, Ey, Ez = _correr(sim.no_mask, 0.5, n, 4.5, 9.0, 592, 0, 0, 1,
                             x_steps=10, z_steps=1, x_range=2400, z_range=4)[3:6]
        anchos.append(_fwhm_x(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2, 10))
    cociente = anchos[1] / anchos[0]
    hay_bug = cociente < 0.85
    _LAM_MULT = 'n' if hay_bug else 1
    if verbose:
        estado = ("DOBLE division lambda/n detectada -> se pasa lambda*n a PyFocus"
                  if hay_bug else "OK (lambda/n se aplica una sola vez)")
        print(f"[pyfocus_wrapper] FWHM(n=1.5)/FWHM(n=1) = {cociente:.2f} -> {estado}")
    return _LAM_MULT


def lam_pyfocus(wavelength, n):
    """Longitud de onda que hay que PASARLE a PyFocus para que use lambda/n efectiva."""
    return wavelength * n if detectar_doble_division(verbose=False) == 'n' else wavelength


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------
def campo_foco(funcion, NA, n, h, w0, wavelength, gamma, beta, I0,
               x_steps, z_steps, x_range, z_range, z=0, nombre='pf',
               solo_xy=False):
    """
    Corre PyFocus (funcion = sim.no_mask o sim.VP) con la correccion de
    lambda ya aplicada. Devuelve dict {"XY": (Ex,Ey,Ez), "XZ": (Ex,Ey,Ez)}
    en las unidades internas de PyFocus (ver factor_E_foco).

    solo_xy=True: calcula una sola fila en z (rapido); "XZ" queda en None.
    """
    lam = lam_pyfocus(wavelength, n)
    if solo_xy:
        z_steps, z_range = 1, 4
    ex_xz, ey_xz, ez_xz, ex_xy, ey_xy, ez_xy = _correr(
        funcion, NA, n, h, w0, lam, gamma, beta, I0,
        x_steps, z_steps, x_range, z_range, z, nombre)
    return {"XY": (ex_xy, ey_xy, ez_xy),
            "XZ": None if solo_xy else (ex_xz, ey_xz, ez_xz)}


def factor_E_foco(n_foco):
    """
    Factor PyFocus -> V/m.   E_SI = E_PyFocus * sqrt(20 / (n_foco * c * eps0))

    Por que n_FOCO y no n=1 (aire): PyFocus usa I0 literalmente como |E0|^2 y
    NO incluye el factor sqrt(n1/n2) de Richards-Wolf (n1 = medio antes del
    objetivo, n2 = medio del foco). Sin ese factor, la potencia en el foco
    sale n2 veces la que entra (verificado: flujo de Poynting integrado en
    el plano focal / potencia que pasa por la apertura = n, con ~3 % de
    perdida por ventana finita). Dividir el campo por sqrt(n_foco) lo
    corrige y deja ademas algo muy comodo:

        I_foco [W/m^2]  = 1/2 * n_foco * c * eps0 * |E_SI|^2  = 10 * |E_PyFocus|^2
        I_foco [mW/cm^2] = |E_PyFocus|^2     (misma unidad que I0)
    """
    return np.sqrt(20.0 / (n_foco * c * epsilon_0))


def intensidad_foco_mWcm2(Ex, Ey, Ez):
    """Intensidad en el foco [mW/cm^2] a partir de campos PyFocus (sin convertir)."""
    return np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
