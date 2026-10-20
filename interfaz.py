# -*- coding: utf-8 -*-
"""
interfaz.py

Campo en el foco de un haz gaussiano (sin mascara de fase) con una interfaz
plana entre dos medios semi-infinitos: n1 (donde esta el foco, p. ej. agua)
y n2 (p. ej. el vidrio del portaobjetos, con aceite de indice igualado
abajo, asi que se lo trata como infinito).


LA IDEA FISICA
==============

1) Un foco es una suma de ondas planas.
   El objetivo convierte el haz que entra en un cono de rayos que convergen
   al foco. Cada rayo es una ONDA PLANA con su direccion:
       theta = angulo respecto del eje optico (de 0 hasta alpha, con
               sin(alpha) = NA / n1),
       phi   = angulo alrededor del eje.
   El campo en cualquier punto cerca del foco es la SUMA (integral) de todas
   esas ondas planas. Es el metodo de Richards-Wolf, el mismo que usa
   PyFocus por dentro.

2) Con vidrio, cada onda plana se divide en dos.
   Cuando una onda plana llega a la superficie agua/vidrio (z = z_int), una
   parte se REFLEJA (vuelve por el agua) y otra se TRANSMITE (sigue en el
   vidrio, con otro angulo por Snell). Cuanto va a cada lado lo dicen los
   coeficientes de FRESNEL, que dependen del angulo y de la polarizacion
   (s o p).

            haz entrante (agua, n1 = 1.33)
                 \\  |  /
                  \\ | /     <- ondas incidentes
         ==========\\|/========== z = z_int  (superficie del vidrio)
                  / | \\     <- ondas transmitidas (vidrio, n2 = 1.52)

       z < z_int (agua):    E = incidente + reflejada
       z > z_int (vidrio):  E = transmitida

   Con n2 = n1 (o z_int = None) no hay reflexion y queda exactamente el
   foco sin interfaz.

3) El truco para que sea rapido.
   Sumar a fuerza bruta todas las ondas en cada punto (400 valores de theta
   por muchos de phi, por decenas de miles de puntos) seria lento. Pero la
   suma sobre phi se puede hacer A MANO y da funciones de Bessel J0, J1 y J2.
   Entonces, para un haz polarizado en x, el campo en (rho, phi, z) siempre
   tiene esta forma:

       Ex = A + B cos(2 phi)
       Ey =     B sin(2 phi)
       Ez = C cos(phi)

   donde A, B y C dependen solo de la distancia al eje (rho) y de la altura
   (z). Solo queda hacer la suma sobre theta, y eso es una multiplicacion de
   matrices (ver FocoConInterfaz._armonicos).


POR QUE NO SE USA PYFOCUS PARA ESTO
===================================
- sim.no_mask y sim.VP ignoran la interfaz: solo la usa sim.custom.
- En sim.custom (PyCustomFocus 3.4.0, custom_mask.py,
  _calculate_factors_with_interface) la onda reflejada de la componente x
  de la entrada NO lleva la fase exp(2i k1 cos(theta) z_int) (la de y si,
  y las formulas comentadas en interface.py tambien). Con polarizacion en x
  el campo reflejado solo es correcto con la interfaz justo en el foco.
- Usa otras grillas (pasos de ~7 y 10 nm) que las de sim.no_mask.


CONVENCIONES (las mismas que el resto de PyForce)
=================================================
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


COMO SE SABE QUE ESTA BIEN (validar_interfaz.py)
================================================
1) Sin vidrio coincide con PyFocus (sim.no_mask) a 1e-14.
2) En la superficie del vidrio se cumplen las condiciones de borde de
   Maxwell (Ex, Ey y n^2 Ez continuos).
3) div E = 0 a los dos lados de la interfaz.
4) gradiente + scattering + spin-curl (fuerzas.py) da la fuerza dipolar
   exacta.
"""
import numpy as np
from scipy.special import jv


# ---------------------------------------------------------------------------
# PASO 1. Grillas
# ---------------------------------------------------------------------------
# Coordenadas (en nm) donde se calcula el campo, con la misma forma y el mismo
# paso que PyFocus (238 x 238 en XY y 239 x 338 en XZ con la config actual).
# Asi el resto del codigo (fuerzas.py, graficos) funciona igual venga el campo
# de PyFocus o de aca.

def ejes_xy(x_steps, x_range):
    """Coordenadas x (= y) del plano XY, en nm, con x = 0 en el indice N//2."""
    N = int(np.rint(x_range / x_steps / 2 - 1) * 2)
    return (np.arange(N) - N // 2) * float(x_steps)


def ejes_xz(x_steps, z_steps, x_range, z_range):
    """
    Coordenadas x y z del plano XZ, en nm. Como en PyFocus, en x el plano XZ
    llega hasta x_range/sqrt(2) (la esquina de la grilla XY), por eso tiene
    ~sqrt(2) veces mas columnas que el XY.
    """
    nz = int(np.rint(z_range / z_steps / 2))
    nr = int(np.rint(x_range / x_steps / 2 ** 0.5))
    x = (np.arange(2 * nr - 2) - (nr - 1)) * float(x_steps)
    z = (np.arange(2 * nz - 1) - (nz - 1)) * float(z_steps)
    return x, z


# ---------------------------------------------------------------------------
# PASO 2. Coeficientes de Fresnel
# ---------------------------------------------------------------------------
def fresnel(n1, n2, cos1):
    """
    Cuanto campo se refleja y cuanto se transmite en una interfaz n1 -> n2,
    para una onda plana que llega con angulo theta1 (cos1 = cos(theta1)).

    Hay dos casos segun hacia donde apunta el campo electrico:
      s: perpendicular al plano de incidencia (paralelo a la superficie);
      p: dentro del plano de incidencia.
    Cada uno se refleja distinto, por eso hay rs/rp y ts/tp.

    Para p, los vectores de polarizacion de la onda reflejada y transmitida
    se eligen con la misma componente tangencial que la incidente, asi que a
    incidencia normal r_p = r_s = (n1 - n2)/(n1 + n2). Para agua -> vidrio
    eso da -0.067: se refleja ~0.5 % de la intensidad.

    Devuelve rs, rp, ts, tp y cos2 = cos(theta2), el angulo dentro del vidrio
    (por Snell: n1 sin(theta1) = n2 sin(theta2)). cos2 se calcula complejo
    para que funcione tambien si hubiera reflexion total interna.
    """
    sin1 = np.sqrt(1 - cos1 ** 2)
    cos2 = np.sqrt(1 - (n1 / n2 * sin1) ** 2 + 0j)
    rs = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
    ts = 2 * n1 * cos1 / (n1 * cos1 + n2 * cos2)
    rp = (n1 * cos2 - n2 * cos1) / (n2 * cos1 + n1 * cos2)
    tp = 2 * n1 * cos1 / (n2 * cos1 + n1 * cos2)
    return rs, rp, ts, tp, cos2


# ---------------------------------------------------------------------------
# PASOS 3 a 9. El calculo
# ---------------------------------------------------------------------------
class FocoConInterfaz:
    """
    Haz gaussiano enfocado por un objetivo (NA, h) en el medio n1, con una
    interfaz opcional a n2 en z = z_int (nm). Mismos parametros y unidades
    que beams/gauss.py: h y w0 en mm, wavelength en nm (en vacio), gamma y
    beta en grados, I0 en mW/cm^2.

    Uso tipico:
        foco = FocoConInterfaz(NA, n1, h, w0, lam, gamma, beta, I0,
                               n2=1.52, z_int=0.0)
        Ex, Ey, Ez = foco.plano_xy(x, z=-50)     # plano XY a z = -50 nm
        Ex, Ey, Ez = foco.plano_xz(x, z)         # plano XZ (y = 0)
    """

    # --- PASO 3. Preparar el haz (se hace una sola vez) ---------------------
    def __init__(self, NA, n1, h, w0, wavelength, gamma, beta, I0,
                 n2=None, z_int=None, n_theta=400):
        self.n1 = n1
        self.n2 = n1 if n2 is None else n2
        self.z_int = z_int
        # numero de onda en cada medio (rad/nm)
        self.k1 = 2 * np.pi * n1 / wavelength
        self.k2 = 2 * np.pi * self.n2 / wavelength
        # angulo maximo del cono de rayos que salen del objetivo
        alpha = np.arcsin(NA / n1)
        # distancia focal del objetivo (ley del seno, misma convencion que PyFocus)
        f_mm = h * n1 / NA
        f_nm = f_mm * 1e6

        # Angulos theta donde se evalua la integral. En vez de puntos
        # equiespaciados se usan los nodos de Gauss-Legendre: con ~400
        # puntos y sus pesos (self._w) la integral sobre theta queda como una
        # suma ponderada muy precisa.
        u, w = np.polynomial.legendre.leggauss(n_theta)
        th = (u + 1) * alpha / 2
        self._w = w * alpha / 2
        self._c1, self._s1 = np.cos(th), np.sin(th)

        # Cuanta luz lleva cada rayo (amplitud), para cada theta:
        #   gauss      el haz que entra es gaussiano: el rayo de angulo theta
        #              sale de la pupila a una altura f*sin(theta), y ahi la
        #              amplitud es exp(-(f sin(theta) / w0)^2) (igual que PyFocus);
        #   sqrt(cos)  "apodizacion" del objetivo (conservacion de la energia);
        #   sin        factor de la integral en coordenadas esfericas;
        #   self._w    peso de Gauss-Legendre.
        gauss = np.exp(-(f_mm * self._s1 / w0) ** 2)
        self._g = gauss * np.sqrt(self._c1) * self._s1 * self._w

        # Constante global de Richards-Wolf: -i k f / (2 pi), por sqrt(I0)
        # (se omite la fase global e^{-ikf}). Es la misma que usa PyFocus, asi
        # que el campo queda en sus unidades: |E|^2 = intensidad en mW/cm^2.
        self._K = -1j * self.k1 * f_nm / (2 * np.pi) * np.sqrt(I0)

        # Polarizacion a la entrada (vector de Jones): cuanto campo hay en x y
        # cuanto en y. gamma = 0 -> todo en x; gamma = 45, beta = 90 -> circular.
        g, b = np.deg2rad(gamma), np.deg2rad(beta)
        self._jx, self._jy = np.cos(g), np.sin(g) * np.exp(1j * b)

        self._ondas = self._armar_ondas()

    # --- PASO 4. Que ondas planas hay ---------------------------------------
    def _armar_ondas(self):
        """
        Para cada tipo de onda ('inc' incidente, 'ref' reflejada, 'tr'
        transmitida) devuelve 5 arrays (un valor por theta):

          c_s  amplitud de la parte s (1 para la incidente; rs o ts si no)
          c_p  amplitud de la parte p (1 para la incidente; rp o tp si no)
          C_p  componente "horizontal" del vector de polarizacion p
               (cos(theta), o cos(theta2) dentro del vidrio)
          S_z  componente z del vector de polarizacion p
               (-sin(theta) para la que baja, +sin(theta) para la que sube)
          k_z  numero de onda en z: + baja (hacia +z), - sube

        Sin vidrio solo existe la incidente.
        """
        c1, s1, k1, k2 = self._c1, self._s1, self.k1, self.k2
        uno = np.ones_like(c1)
        ondas = {'inc': (uno, uno, c1, -s1, k1 * c1)}
        if self.z_int is None or self.n2 == self.n1:
            return ondas
        rs, rp, ts, tp, c2 = fresnel(self.n1, self.n2, c1)
        s2 = self.n1 / self.n2 * s1  # Snell: sin(theta2)
        # Fases por la posicion de la superficie (z_int): Fresnel relaciona
        # las ondas EN la superficie; estas fases las llevan a z = 0.
        #   reflejada:   va hasta z_int y vuelve -> 2 k1 cos(theta) z_int
        #   transmitida: lo que acumulo en el agua menos lo que "acumularia"
        #                en el vidrio hasta z_int
        # (Esta es la fase que PyFocus olvida para la componente x.)
        fase_r = np.exp(2j * k1 * c1 * self.z_int)
        fase_t = np.exp(1j * (k1 * c1 - k2 * c2) * self.z_int)
        ondas['ref'] = (rs * fase_r, rp * fase_r, c1, s1, -k1 * c1)
        ondas['tr'] = (ts * fase_t, tp * fase_t, c2, -s2, k2 * c2)
        return ondas

    # --- PASO 5. La suma sobre theta (el truco de Bessel) -------------------
    def _armonicos(self, rho, z, nombres):
        """
        Calcula A, B y C (cada uno de forma (len(rho), len(z))) sumando las
        ondas `nombres`, para una entrada polarizada en x:

            Ex = A + B cos(2 phi),   Ey = B sin(2 phi),   Ez = C cos(phi)

        La suma sobre phi ya esta hecha (da las Bessel J0, J1, J2 de
        k1 rho sin(theta)). Para cada onda queda, sumando sobre theta:

            A ~ sum  g (c_s + c_p C_p)  J0  e^{i k_z z}
            B ~ sum  g (c_s - c_p C_p)  J2  e^{i k_z z}
            C ~ sum  g  c_p S_z         J1  e^{i k_z z}

        En el codigo cada suma es UNA multiplicacion de matrices (@):
        (Bessel x amplitud), de forma (puntos rho, theta), por la fase
        e^{i k_z z}, de forma (theta, alturas z). Por eso un plano entero sale
        en segundos.
        """
        rho = np.asarray(rho, float)
        z = np.asarray(z, float)
        u = self.k1 * np.outer(rho, self._s1)            # (R, T): argumento de las Bessel
        J0, J1, J2 = jv(0, u), jv(1, u), jv(2, u)
        A = B = C = 0
        for nombre in nombres:
            cs, cp, Cp, Sz, kz = self._ondas[nombre]
            P = np.exp(1j * np.outer(kz, z))              # (T, Z): fase de cada onda en cada z
            A = A + (J0 * (self._g * (cs + cp * Cp))) @ P
            B = B + (J2 * (self._g * (cs - cp * Cp))) @ P
            C = C + (J1 * (self._g * cp * Sz)) @ P
        # factores que salen de la integral en phi (pi, pi y 2 pi i)
        return np.pi * self._K * A, np.pi * self._K * B, 2j * np.pi * self._K * C

    # --- PASO 6. Armar Ex, Ey, Ez -------------------------------------------
    def _campo(self, rho, phi, z, nombres):
        """
        Campo (Ex, Ey, Ez) en los puntos (rho, phi) (arrays de forma (R,)) y
        las alturas z (forma (Z,)); devuelve arrays de forma (R, Z).

        Con A, B, C arma el campo de la entrada en x. La parte de la entrada
        en y no necesita otra cuenta: un haz polarizado en y es el mismo haz
        polarizado en x rotado 90 grados (de ahi los senos y cosenos
        intercambiados en los terminos con jy).
        """
        A, B, C = self._armonicos(rho, z, nombres)
        c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
        c2, s2 = np.cos(2 * phi)[:, None], np.sin(2 * phi)[:, None]
        jx, jy = self._jx, self._jy
        Ex = jx * (A + B * c2) + jy * (B * s2)
        Ey = jx * (B * s2) + jy * (A - B * c2)
        Ez = jx * (C * c) + jy * (C * s)
        return Ex, Ey, Ez

    # --- PASO 7. Arriba o abajo del vidrio ----------------------------------
    def _region(self, z):
        """Que ondas sumar a la altura z: incidente + reflejada (agua) o transmitida (vidrio)."""
        if self.z_int is None or self.n2 == self.n1:
            return ('inc',)
        return ('tr',) if z >= self.z_int else ('inc', 'ref')

    # --- PASO 8. Planos y puntos --------------------------------------------
    def campo_puntos(self, x, y, z):
        """Campo en puntos sueltos (x, y arrays de la misma forma; z escalar)."""
        x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        rho, phi = np.hypot(x.ravel(), y.ravel()), np.arctan2(y.ravel(), x.ravel())
        nombres = self._region(z)
        E = self._campo(rho, phi, [z], nombres)
        return tuple(e[:, 0].reshape(x.shape) for e in E)

    def plano_xy(self, x, z=0.0, bloque=4000):
        """
        Plano XY a la altura z (filas -y -> +y, columnas -x -> +x).
        Pasa cada punto de la grilla a (rho, phi) y calcula de a `bloque`
        puntos para no llenar la memoria.
        """
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
        """
        Plano XZ a la altura y (filas -z -> +z, columnas -x -> +x).
        Con vidrio calcula las dos versiones del campo (agua y vidrio) y en
        cada fila se queda con la que corresponde segun z.
        """
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

    # --- PASO 9. Derivadas fuera del plano ----------------------------------
    # El scattering y el spin-curl necesitan saber como cambia el campo en la
    # direccion PERPENDICULAR al plano (d/dz en XY, d/dy en XZ), que un solo
    # plano no tiene. Se calcula el plano corrido h = 0.5 nm para cada lado y
    # se hace la diferencia centrada: (E(+h) - E(-h)) / (2h).

    def derivada_xy_z(self, x, z=0.0, h=0.5):
        """(dEx/dz, dEy/dz, dEz/dz) en el plano XY a la altura z (por nm)."""
        Ea, Eb = self.plano_xy(x, z + h), self.plano_xy(x, z - h)
        return tuple((a - b) / (2 * h) for a, b in zip(Ea, Eb))

    def derivada_xz_y(self, x, z, h=0.5):
        """(dEx/dy, dEy/dy, dEz/dy) en el plano XZ (y = 0) (por nm)."""
        Ea, Eb = self.plano_xz(x, z, h), self.plano_xz(x, z, -h)
        return tuple((a - b) / (2 * h) for a, b in zip(Ea, Eb))


# ---------------------------------------------------------------------------
# PASO 10. Atajo para usar desde afuera
# ---------------------------------------------------------------------------
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
