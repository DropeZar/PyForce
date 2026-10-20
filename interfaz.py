# -*- coding: utf-8 -*-
"""
interfaz.py

Campo en el foco de un haz gaussiano (sin mascara de fase) con una interfaz
plana entre dos medios semi-infinitos: n1 (donde esta el foco, p. ej. agua)
y n2 (p. ej. el vidrio del portaobjetos, con aceite de indice igualado
abajo, asi que se lo trata como infinito).

Por que no se usa PyFocus para esto
-----------------------------------
- sim.no_mask y sim.VP ignoran la interfaz: solo la usa sim.custom.
- En sim.custom (PyCustomFocus 3.4.0, custom_mask.py,
  _calculate_factors_with_interface) la onda reflejada de la componente x
  de la entrada NO lleva la fase exp(2i k1 cos(theta) z_int) (la de y si,
  y las formulas comentadas en interface.py tambien). Con polarizacion en x
  el campo reflejado solo es correcto con la interfaz justo en el foco.
- Usa otras grillas (pasos de ~7 y 10 nm) que las de sim.no_mask.

Modelo
------
Richards-Wolf (aproximacion de Debye), como sim.no_mask: cada direccion
(theta, phi) del cono de apertura es una onda plana en n1 que llega a la
interfaz en z = z_int, donde se separa en reflejada y transmitida con los
coeficientes de Fresnel s y p. La integral en phi se hace analiticamente
(funciones de Bessel J0, J1, J2) y la de theta con Gauss-Legendre.

    z < z_int : E = incidente + reflejada   (medio n1)
    z > z_int : E = transmitida             (medio n2)

Con n2 = n1 (o z_int = None) es exactamente el foco sin interfaz.

Convenciones (las mismas que el resto de PyForce)
-------------------------------------------------
- El haz se propaga hacia +z. Interfaz con z_int > 0: el vidrio esta
  "despues" del foco.
- Unidades de campo de PyFocus: |E|^2 = intensidad en mW/cm^2; a V/m con
  fuerzas.calcular_factor_E_foco(n1).
- Grillas iguales a las de sim.no_mask: XY de (N, N) con N = 2*rint(x_range/
  x_steps/2 - 1); XZ de (2*nz - 1, 2*nr - 2) con x en +-x_range/sqrt(2).
  Filas de XY de -y a +y y de XZ de -z a +z, columnas de -x a +x.
- Ez con el signo fisico (Novotny & Hecht, ec. 3.66). sim.no_mask devuelve
  Ez con el signo opuesto respecto de Ex; no cambia |E|^2 ni las fuerzas
  de gradiente y scattering (ver validar_interfaz.py).
"""
import numpy as np
from scipy.special import jv


# ---------------------------------------------------------------------------
# Grillas (mismas formas y pasos que sim.no_mask)
# ---------------------------------------------------------------------------
def ejes_xy(x_steps, x_range):
    N = int(np.rint(x_range / x_steps / 2 - 1) * 2)
    return (np.arange(N) - N // 2) * float(x_steps)


def ejes_xz(x_steps, z_steps, x_range, z_range):
    nz = int(np.rint(z_range / z_steps / 2))
    nr = int(np.rint(x_range / x_steps / 2 ** 0.5))
    x = (np.arange(2 * nr - 2) - (nr - 1)) * float(x_steps)
    z = (np.arange(2 * nz - 1) - (nz - 1)) * float(z_steps)
    return x, z


# ---------------------------------------------------------------------------
# Fresnel (una interfaz, convencion: componentes tangenciales de E)
# ---------------------------------------------------------------------------
def fresnel(n1, n2, cos1):
    """
    Coeficientes de amplitud de E para una interfaz n1 -> n2.
    Para p, los vectores de polarizacion de la onda reflejada y transmitida
    se eligen con la misma componente tangencial que la incidente, asi que a
    incidencia normal r_p = r_s = (n1 - n2)/(n1 + n2).
    Devuelve rs, rp, ts, tp, cos2 (complejo si hubiera reflexion total).
    """
    sin1 = np.sqrt(1 - cos1 ** 2)
    cos2 = np.sqrt(1 - (n1 / n2 * sin1) ** 2 + 0j)
    rs = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
    ts = 2 * n1 * cos1 / (n1 * cos1 + n2 * cos2)
    rp = (n1 * cos2 - n2 * cos1) / (n2 * cos1 + n1 * cos2)
    tp = 2 * n1 * cos1 / (n2 * cos1 + n1 * cos2)
    return rs, rp, ts, tp, cos2


# ---------------------------------------------------------------------------
# Calculo
# ---------------------------------------------------------------------------
class FocoConInterfaz:
    """
    Haz gaussiano enfocado por un objetivo (NA, h) en el medio n1, con una
    interfaz opcional a n2 en z = z_int (nm). Mismos parametros y unidades
    que beams/gauss.py: h y w0 en mm, wavelength en nm (en vacio), gamma y
    beta en grados, I0 en mW/cm^2.
    """

    def __init__(self, NA, n1, h, w0, wavelength, gamma, beta, I0,
                 n2=None, z_int=None, n_theta=400):
        self.n1 = n1
        self.n2 = n1 if n2 is None else n2
        self.z_int = z_int
        self.k1 = 2 * np.pi * n1 / wavelength
        self.k2 = 2 * np.pi * self.n2 / wavelength
        alpha = np.arcsin(NA / n1)
        f_mm = h * n1 / NA  # misma convencion que PyFocus
        f_nm = f_mm * 1e6

        # nodos de Gauss-Legendre en [0, alpha]
        u, w = np.polynomial.legendre.leggauss(n_theta)
        th = (u + 1) * alpha / 2
        self._w = w * alpha / 2
        self._c1, self._s1 = np.cos(th), np.sin(th)

        # amplitud en la pupila (gaussiana, igual que PyFocus) * apodizacion
        # sqrt(cos) * sin (jacobiano)
        gauss = np.exp(-(f_mm * self._s1 / w0) ** 2)
        self._g = gauss * np.sqrt(self._c1) * self._s1 * self._w
        # prefactor de Debye: -i k f / (2 pi) (sin la fase global e^{-ikf})
        self._K = -1j * self.k1 * f_nm / (2 * np.pi) * np.sqrt(I0)
        g, b = np.deg2rad(gamma), np.deg2rad(beta)
        self._jx, self._jy = np.cos(g), np.sin(g) * np.exp(1j * b)

        self._ondas = self._armar_ondas()

    def _armar_ondas(self):
        """(c_s, c_p, C_p, S_z, k_z) de cada onda plana, para cada theta."""
        c1, s1, k1, k2 = self._c1, self._s1, self.k1, self.k2
        uno = np.ones_like(c1)
        ondas = {'inc': (uno, uno, c1, -s1, k1 * c1)}
        if self.z_int is None or self.n2 == self.n1:
            return ondas
        rs, rp, ts, tp, c2 = fresnel(self.n1, self.n2, c1)
        s2 = self.n1 / self.n2 * s1
        fase_r = np.exp(2j * k1 * c1 * self.z_int)
        fase_t = np.exp(1j * (k1 * c1 - k2 * c2) * self.z_int)
        ondas['ref'] = (rs * fase_r, rp * fase_r, c1, s1, -k1 * c1)
        ondas['tr'] = (ts * fase_t, tp * fase_t, c2, -s2, k2 * c2)
        return ondas

    def _armonicos(self, rho, z, nombres):
        """
        A, B, C (cada uno de forma (len(rho), len(z))) de la suma de las ondas
        `nombres`, para entrada polarizada en x:
            Ex = A + B cos(2 phi),  Ey = B sin(2 phi),  Ez = C cos(phi)
        """
        rho = np.asarray(rho, float)
        z = np.asarray(z, float)
        u = self.k1 * np.outer(rho, self._s1)            # (R, T)
        J0, J1, J2 = jv(0, u), jv(1, u), jv(2, u)
        A = B = C = 0
        for nombre in nombres:
            cs, cp, Cp, Sz, kz = self._ondas[nombre]
            P = np.exp(1j * np.outer(kz, z))              # (T, Z)
            A = A + (J0 * (self._g * (cs + cp * Cp))) @ P
            B = B + (J2 * (self._g * (cs - cp * Cp))) @ P
            C = C + (J1 * (self._g * cp * Sz)) @ P
        return np.pi * self._K * A, np.pi * self._K * B, 2j * np.pi * self._K * C

    def _campo(self, rho, phi, z, nombres):
        """Campo (Ex, Ey, Ez) en puntos (rho, phi) (forma (R,)) x z (forma (Z,))."""
        A, B, C = self._armonicos(rho, z, nombres)
        c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
        c2, s2 = np.cos(2 * phi)[:, None], np.sin(2 * phi)[:, None]
        jx, jy = self._jx, self._jy
        # entrada en y = entrada en x rotada 90 grados
        Ex = jx * (A + B * c2) + jy * (B * s2)
        Ey = jx * (B * s2) + jy * (A - B * c2)
        Ez = jx * (C * c) + jy * (C * s)
        return Ex, Ey, Ez

    def campo_puntos(self, x, y, z):
        """Campo en puntos sueltos (x, y arrays de la misma forma; z escalar)."""
        x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        rho, phi = np.hypot(x.ravel(), y.ravel()), np.arctan2(y.ravel(), x.ravel())
        nombres = self._region(z)
        E = self._campo(rho, phi, [z], nombres)
        return tuple(e[:, 0].reshape(x.shape) for e in E)

    def _region(self, z):
        if self.z_int is None or self.n2 == self.n1:
            return ('inc',)
        return ('tr',) if z >= self.z_int else ('inc', 'ref')

    def plano_xy(self, x, z=0.0, bloque=4000):
        """Plano XY (filas -y -> +y, columnas -x -> +x) a la altura z."""
        X, Y = np.meshgrid(x, x)
        rho, phi = np.hypot(X, Y).ravel(), np.arctan2(Y, X).ravel()
        nombres = self._region(z)
        salida = [np.empty(rho.size, complex) for _ in range(3)]
        for i in range(0, rho.size, bloque):
            sl = slice(i, i + bloque)
            for out, e in zip(salida, self._campo(rho[sl], phi[sl], [z], nombres)):
                out[sl] = e[:, 0]
        return tuple(e.reshape(X.shape) for e in salida)

    def plano_xz(self, x, z, y=0.0):
        """Plano XZ a la altura y (filas -z -> +z, columnas -x -> +x)."""
        x = np.asarray(x, float)
        rho, phi = np.hypot(x, y), np.arctan2(y, x)
        z = np.asarray(z, float)
        if self.z_int is None or self.n2 == self.n1:
            E = self._campo(rho, phi, z, ('inc',))
            return tuple(e.T for e in E)
        abajo = z < self.z_int
        E1 = self._campo(rho, phi, z, ('inc', 'ref'))
        E2 = self._campo(rho, phi, z, ('tr',))
        return tuple(np.where(abajo[None, :], a, b).T for a, b in zip(E1, E2))

    def derivada_xy_z(self, x, z=0.0, h=0.5):
        """(dEx/dz, dEy/dz, dEz/dz) en el plano XY a la altura z (por nm)."""
        Ea, Eb = self.plano_xy(x, z + h), self.plano_xy(x, z - h)
        return tuple((a - b) / (2 * h) for a, b in zip(Ea, Eb))

    def derivada_xz_y(self, x, z, h=0.5):
        """(dEx/dy, dEy/dy, dEz/dy) en el plano XZ (y = 0) (por nm)."""
        Ea, Eb = self.plano_xz(x, z, h), self.plano_xz(x, z, -h)
        return tuple((a - b) / (2 * h) for a, b in zip(Ea, Eb))


def campo_gauss(haz, x_steps, z_steps, x_range, z_range, n2=None, z_int=None,
                z_plano=0.0, n_theta=400):
    """
    Campo del modulo de haz `haz` (beams.gauss) con o sin interfaz, en las
    mismas grillas que pyfocus_wrapper.campo_foco. Devuelve
    {"XY": (Ex, Ey, Ez), "XZ": (Ex, Ey, Ez), "x_xy", "x_xz", "z_xz",
     "dXY_dz": derivadas en z del plano XY, "dXZ_dy": derivadas en y del XZ}
    en unidades de PyFocus (las derivadas, por nm). Las derivadas son las que
    necesita el spin-curl en fuerzas.fuerzas_xy / fuerzas_xz.
    """
    foco = FocoConInterfaz(haz.NA, haz.n, haz.h, haz.w0, haz.wavelength,
                           haz.gamma, haz.beta, haz.I0, n2=n2, z_int=z_int,
                           n_theta=n_theta)
    x_xy = ejes_xy(x_steps, x_range)
    x_xz, z_xz = ejes_xz(x_steps, z_steps, x_range, z_range)
    return {"XY": foco.plano_xy(x_xy, z_plano), "XZ": foco.plano_xz(x_xz, z_xz),
            "x_xy": x_xy, "x_xz": x_xz, "z_xz": z_xz,
            "dXY_dz": foco.derivada_xy_z(x_xy, z_plano),
            "dXZ_dy": foco.derivada_xz_y(x_xz, z_xz)}
