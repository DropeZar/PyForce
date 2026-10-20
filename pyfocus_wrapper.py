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

    Ademas pasa la salida de PyFocus a la convencion fisica que usa el resto
    del codigo (fuerzas.py):
      - el plano XY de PyFocus tiene las filas de +y a -y: se da vuelta para
        que vayan de -y a +y (si no, las derivadas en y salen con el signo
        cambiado sin que se den vuelta las componentes Ey);
      - PyFocus devuelve Ez con el signo opuesto al fisico: con su Ez,
        div E da del orden de |dEx/dx| en vez de 0 (con el signo cambiado da 0).
    |E|^2, el gradiente y el scattering no cambian; el spin-curl si, porque
    mezcla Ez con Ex y Ey.

    solo_xy=True: calcula una sola fila en z (rapido); "XZ" queda en None.
    """
    lam = lam_pyfocus(wavelength, n)
    if solo_xy:
        z_steps, z_range = 1, 4
    ex_xz, ey_xz, ez_xz, ex_xy, ey_xy, ez_xy = _correr(
        funcion, NA, n, h, w0, lam, gamma, beta, I0,
        x_steps, z_steps, x_range, z_range, z, nombre)
    return {"XY": (np.flipud(ex_xy), np.flipud(ey_xy), -np.flipud(ez_xy)),
            "XZ": None if solo_xy else (ex_xz, ey_xz, -ez_xz)}


def campo_foco_vidrio(NA, n1, n2, h, w0, wavelength, gamma, beta, I0,
                      x_steps, z_steps, x_range, z_range, z=0, divisiones=200):
    """
    Haz gaussiano enfocado en el medio n1 (agua) con la superficie de un
    medio n2 (vidrio, semi-infinito) EN EL FOCO (z = 0). Usa sim.custom de
    PyFocus con multilayer, que es la unica funcion de PyFocus que tiene en
    cuenta una interfaz.

    Solo para la interfaz en el foco: con la interfaz en otra posicion,
    sim.custom (PyCustomFocus 3.4.0) da mal la onda reflejada de la
    componente x y la transmitida.

    Parametros como campo_foco (wavelength en vacio, h y w0 en mm, I0 en
    mW/cm^2). z es la altura del plano XY y tambien el centro del plano XZ.
    divisiones: puntos en theta y en phi de la integral de PyFocus.

    Devuelve {"XY": (Ex,Ey,Ez), "XZ": (Ex,Ey,Ez), "x_xy", "x_xz", "z_xz"}, en
    las mismas unidades y convencion fisica que campo_foco (filas de -y a +y
    y de -z a +z, Ez con el signo fisico). Arriba del vidrio (z < 0) el
    campo es incidente + reflejada; abajo, la transmitida. Con n2 = n1 es el
    foco sin vidrio, calculado del mismo modo.

    Detalles de sim.custom que se corrigen o hay que tener en cuenta:
    - Grilla distinta de sim.no_mask: XY de nr x nr puntos y XZ de
      nz x nr, con nr = rint(x_range/x_steps/sqrt(2)) y nz = rint(z_range/
      z_steps/2), equiespaciados en [-x_range/2, x_range/2] y
      [z - z_range/2, z + z_range/2]. Por eso se devuelven los ejes: el
      paso real es x_range/(nr-1) y z_range/(nz-1), no x_steps y z_steps.
      Para que x = 0 y la altura z queden en la grilla, nr y nz tienen que
      ser impares.
    - La integral en theta y phi usa paso alpha/N y 2pi/N con N puntos que
      incluyen los dos extremos, asi que suma (N-1)/N de cada integral: el
      campo sale (N-1)^2/N^2 veces el real (~1 % menos con N = 200). Se
      multiplica por (N/(N-1))^2.
    - Con interfaz PyFocus no divide lambda por n (la doble division de
      campo_foco no aplica): se le pasa lambda en vacio.
    """
    from PyFocus import sim

    uno = lambda rho, phi, w0, f, k: 1  # sin mascara: haz gaussiano liso
    params = (False, True, NA, np.array([n1, n2], dtype=float), h, w0, wavelength,
              gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, '', '',
              np.array([np.inf, np.inf]), 0, 'vidrio', divisiones, divisiones, False)
    with _silencio():
        ex_xz, ey_xz, ez_xz, ex_xy, ey_xy, ez_xy = sim.custom(uno, uno, *params)

    corr = (divisiones / (divisiones - 1)) ** 2
    nr = ex_xy.shape[1]
    nz = ex_xz.shape[0]
    x = np.linspace(-x_range / 2, x_range / 2, nr)
    z_xz = z + np.linspace(-z_range / 2, z_range / 2, nz)

    def fisico(ex, ey, ez):  # filas de +y/+z a -y/-z -> al reves, y -Ez
        return tuple(corr * s * np.flipud(e) for s, e in ((1, ex), (1, ey), (-1, ez)))

    return {"XY": fisico(ex_xy, ey_xy, ez_xy), "XZ": fisico(ex_xz, ey_xz, ez_xz),
            "x_xy": x, "x_xz": x, "z_xz": z_xz}


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
