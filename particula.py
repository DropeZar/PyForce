# -*- coding: utf-8 -*-
"""
particula.py

Polarizabilidad de una nanoesfera metalica, con la permitividad del oro
interpolada de la tabla completa de Johnson & Christy (1972).

Hay dos versiones, con la misma firma y las mismas unidades (m^3):

- calc_alpha: aproximacion de Kuwata. Es la que usan main.py y
  grafico_lindo.py.
- calc_alpha_mie: polarizabilidad dipolar EXACTA (teoria de Mie), sacada
  del coeficiente a1 (Bohren & Huffman, con recurrencia descendente de la
  derivada logaritmica D_n(mx) para que sea estable con el indice complejo
  del oro). Sirve para validar Kuwata (graficar_comparacion,
  comparar_radios); para usarla en la simulacion alcanza con cambiar el
  import en main.py.

Uso tipico:
    from particula import calc_alpha
    alpha_p, alpha_pp = calc_alpha(592)          # radio y medio de sim_config.py
    alpha_p, alpha_pp = calc_alpha(532, a=50e-9) # otro radio
"""
import numpy as np

from sim_config import radio_particula, eps_m

# ─── Datos n,k de Johnson & Christy, Phys. Rev. B 6, 4370-4379 (1972) ────────
# via refractiveindex.info (dominio publico, CC0 1.0). Columnas: lambda[um], n, k
_NK_RAW = np.array([
    [0.1879, 1.28, 1.188], [0.1916, 1.32, 1.203], [0.1953, 1.34, 1.226],
    [0.1993, 1.33, 1.251], [0.2033, 1.33, 1.277], [0.2073, 1.30, 1.304],
    [0.2119, 1.30, 1.350], [0.2164, 1.30, 1.387], [0.2214, 1.30, 1.427],
    [0.2262, 1.31, 1.460], [0.2313, 1.30, 1.497], [0.2371, 1.32, 1.536],
    [0.2426, 1.32, 1.577], [0.2490, 1.33, 1.631], [0.2551, 1.33, 1.688],
    [0.2616, 1.35, 1.749], [0.2689, 1.38, 1.803], [0.2761, 1.43, 1.847],
    [0.2844, 1.47, 1.869], [0.2924, 1.49, 1.878], [0.3009, 1.53, 1.889],
    [0.3107, 1.53, 1.893], [0.3204, 1.54, 1.898], [0.3315, 1.48, 1.883],
    [0.3425, 1.48, 1.871], [0.3542, 1.50, 1.866], [0.3679, 1.48, 1.895],
    [0.3815, 1.46, 1.933], [0.3974, 1.47, 1.952], [0.4133, 1.46, 1.958],
    [0.4305, 1.45, 1.948], [0.4509, 1.38, 1.914], [0.4714, 1.31, 1.849],
    [0.4959, 1.04, 1.833], [0.5209, 0.62, 2.081], [0.5486, 0.43, 2.455],
    [0.5821, 0.29, 2.863], [0.6168, 0.21, 3.272], [0.6595, 0.14, 3.697],
    [0.7045, 0.13, 4.103], [0.7560, 0.14, 4.542], [0.8211, 0.16, 5.083],
    [0.8920, 0.17, 5.663], [0.9840, 0.22, 6.350], [1.0880, 0.27, 7.150],
    [1.2160, 0.35, 8.145], [1.3930, 0.43, 9.519], [1.6100, 0.56, 11.21],
    [1.9370, 0.92, 13.78],
])

# Conversion a permitividad: eps = (n + ik)^2 = (n^2 - k^2) + i(2nk)
_lam_nm_tab = _NK_RAW[:, 0] * 1000
_n_tab, _k_tab = _NK_RAW[:, 1], _NK_RAW[:, 2]
JC_DATA = np.column_stack([
    _lam_nm_tab,
    _n_tab ** 2 - _k_tab ** 2,
    2 * _n_tab * _k_tab,
])


def epsilon_gold(wavelength_nm):
    """Permitividad compleja del oro interpolada de la tabla J&C 1972."""
    eps_r = np.interp(wavelength_nm, JC_DATA[:, 0], JC_DATA[:, 1])
    eps_i = np.interp(wavelength_nm, JC_DATA[:, 0], JC_DATA[:, 2])
    return eps_r + 1j * eps_i


def calc_alpha(wavelength_nm, a=radio_particula, eps_m_local=eps_m):
    """
    Polarizabilidad de Kuwata para esfera metalica.

    Parametros
    ----------
    wavelength_nm : float o array
        Longitud de onda en nm.
    a : float
        Radio de la particula en metros. Por defecto, sim_config.radio_particula.
    eps_m_local : float
        Permitividad del medio. Por defecto, sim_config.eps_m.

    Retorna (Re(alpha), Im(alpha)) en m^3.
    """
    lam = wavelength_nm * 1e-9
    eps_p = epsilon_gold(wavelength_nm)
    V = (4 / 3) * np.pi * a ** 3
    x = np.pi * a / lam

    num = 1 - (1 / 10) * (eps_p - eps_m_local) * x ** 2
    denom_CM = (1 / 3) + eps_m_local / (eps_p - eps_m_local)
    denom_ret = -(1 / 30) * (eps_p + 10 * eps_m_local) * x ** 2
    denom_rr = -1j * (4 * np.pi * eps_m_local ** 1.5 / 3) * (V / lam ** 3)

    alpha_K = V * num / (denom_CM + denom_ret + denom_rr)
    return np.real(alpha_K), np.imag(alpha_K)


# ─── Polarizabilidad exacta (teoria de Mie completa) ─────────────────────────

def _mie_a1(m, x):
    """
    Coeficiente de Mie a1 (dipolo electrico), convencion de Bohren & Huffman.

    m : indice de refraccion relativo (n_particula / n_medio), complejo.
    x : parametro de tamano k*a (k = 2*pi*n_medio/lambda), real.

    Usa recurrencia DESCENDENTE para la derivada logaritmica D_n(mx), que es
    la forma numericamente estable para indices complejos (metal absorbente).
    """
    if x < 1e-8:
        return 0.0 + 0.0j

    # margen de terminos suficiente para convergencia (criterio de Wiscombe)
    nmax = max(15, int(np.ceil(x + 4 * x ** (1 / 3) + 2)) + 10)
    mx = m * x

    D = np.zeros(nmax + 1, dtype=complex)  # D[n] = D_n(mx)
    for n in range(nmax, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)

    # funciones de Riccati-Bessel psi_n, chi_n, xi_n = psi_n - i*chi_n
    psi0, chi0 = np.sin(x), np.cos(x)
    psi1 = psi0 / x - chi0
    chi1 = chi0 / x + psi0
    xi0 = psi0 - 1j * chi0
    xi1 = psi1 - 1j * chi1

    D1 = D[1]
    a1 = ((D1 / m + 1 / x) * psi1 - psi0) / ((D1 / m + 1 / x) * xi1 - xi0)
    return a1


def calc_alpha_mie(wavelength_nm, a=radio_particula, eps_m_local=eps_m):
    """
    Polarizabilidad dipolar EXACTA (teoria de Mie completa), extraida del
    coeficiente a1. Misma normalizacion/unidades (m^3) que calc_alpha, para
    poder compararlas directamente.

    alpha = i * (6*pi / k^3) * a1   con k = 2*pi*n_medio/lambda

    (la constante 6*pi, en vez de la 3/2 "de libro", es la que hace que el
    limite a->0 coincida exactamente con el termino cuasiestatico de Kuwata,
    que ya trae el V=(4/3)*pi*a^3 explicito; se valido numericamente).

    Retorna (Re(alpha), Im(alpha)) en m^3.
    """
    wl = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    lam = wl * 1e-9
    eps_p = epsilon_gold(wl)
    n_p = np.sqrt(eps_p)
    n_m = np.sqrt(eps_m_local)
    k = 2 * np.pi * n_m / lam
    x = k * a
    m_rel = n_p / n_m

    alpha = np.empty(wl.shape, dtype=complex)
    for i in range(wl.size):
        a1 = _mie_a1(m_rel[i], x[i])
        alpha[i] = 1j * (6 * np.pi / k[i] ** 3) * a1

    if alpha.size == 1:
        return np.real(alpha[0]), np.imag(alpha[0])
    return np.real(alpha), np.imag(alpha)


# ─── Graficos ─────────────────────────────────────────────────────────────

def graficar_alpha(lam_min=400, lam_max=900, n_puntos=700, a=radio_particula,
                    eps_m_local=eps_m, longitudes_marcadas=(592, 1064)):
    """Grafico de diagnostico de Re/Im/|alpha| en funcion de la longitud de onda (Kuwata)."""
    import matplotlib.pyplot as plt

    lam_nm = np.linspace(lam_min, lam_max, n_puntos)
    ap, app = calc_alpha(lam_nm, a=a, eps_m_local=eps_m_local)
    alpha_c = ap + 1j * app

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lam_nm, alpha_c.real, label=r"Re($\alpha$)", color="tab:blue")
    ax.plot(lam_nm, alpha_c.imag, label=r"Im($\alpha$)", color="tab:red")
    ax.plot(lam_nm, np.abs(alpha_c), label=r"$|\alpha|$", color="tab:green", ls="--")

    for lam0 in longitudes_marcadas:
        if lam_min <= lam0 <= lam_max:
            ax.axvline(lam0, color="gray", lw=0.7, ls=":")

    ax.set_xlabel("Longitud de onda [nm]")
    ax.set_ylabel(r"$\alpha$  [m$^3$]")
    ax.set_title(f"Polarizabilidad de Kuwata — nanoesfera Au, a={a*1e9:.0f} nm")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig, ax


def graficar_comparacion(lam_min=400, lam_max=900, n_puntos=400, a=radio_particula,
                          eps_m_local=eps_m, longitudes_marcadas=(592, 1064)):
    """
    Compara Kuwata vs Mie exacto: Re(alpha), Im(alpha) y |alpha| superpuestos,
    para un radio, en tres paneles.
    """
    import matplotlib.pyplot as plt

    lam_nm = np.linspace(lam_min, lam_max, n_puntos)
    apK, appK = calc_alpha(lam_nm, a=a, eps_m_local=eps_m_local)
    apM, appM = calc_alpha_mie(lam_nm, a=a, eps_m_local=eps_m_local)
    aK, aM = apK + 1j * appK, apM + 1j * appM

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    titulos = [r"Re($\alpha$)", r"Im($\alpha$)", r"$|\alpha|$"]
    datosK = [aK.real, aK.imag, np.abs(aK)]
    datosM = [aM.real, aM.imag, np.abs(aM)]

    for ax, tit, dK, dM in zip(axes, titulos, datosK, datosM):
        ax.plot(lam_nm, dK, label="Kuwata", color="tab:blue")
        ax.plot(lam_nm, dM, label="Mie (exacto)", color="tab:orange", ls="--")
        for lam0 in longitudes_marcadas:
            if lam_min <= lam0 <= lam_max:
                ax.axvline(lam0, color="gray", lw=0.6, ls=":")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_title(tit)
        ax.set_xlabel("Longitud de onda [nm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    lam_res_K = lam_nm[np.argmax(np.abs(aK))]
    lam_res_M = lam_nm[np.argmax(np.abs(aM))]
    fig.suptitle(
        f"Kuwata vs Mie exacto — nanoesfera Au, a={a*1e9:.0f} nm   "
        f"(máx |α|: Kuwata {lam_res_K:.0f} nm, Mie {lam_res_M:.0f} nm)"
    )
    fig.tight_layout()
    plt.show()
    return fig, axes


def comparar_radios(radios_nm=(20, 40, 60, 80), lam_min=450, lam_max=750,
                     n_puntos=400, eps_m_local=eps_m):
    """
    |alpha(lambda)| para varios radios, Kuwata vs Mie, para ver de un vistazo
    cuanto (no) se corre el maximo con Kuwata comparado con Mie.
    """
    import matplotlib.pyplot as plt

    lam_nm = np.linspace(lam_min, lam_max, n_puntos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    cmap = plt.cm.viridis(np.linspace(0, 1, len(radios_nm)))
    for a_nm, color in zip(radios_nm, cmap):
        a = a_nm * 1e-9
        apK, appK = calc_alpha(lam_nm, a=a, eps_m_local=eps_m_local)
        apM, appM = calc_alpha_mie(lam_nm, a=a, eps_m_local=eps_m_local)
        aK = np.abs(apK + 1j * appK)
        aM = np.abs(apM + 1j * appM)
        ax1.plot(lam_nm, aK, color=color, label=f"a={a_nm} nm")
        ax2.plot(lam_nm, aM, color=color, label=f"a={a_nm} nm")

    ax1.set_title("Kuwata")
    ax2.set_title("Mie (exacto)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Longitud de onda [nm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax1.set_ylabel(r"$|\alpha|$  [m$^3$]")
    fig.suptitle("Corrimiento del máximo de |α| con el radio")
    fig.tight_layout()
    plt.show()
    return fig, (ax1, ax2)


if __name__ == '__main__':
    for lam in (592, 1064):
        for nombre, fn in (("Kuwata", calc_alpha), ("Mie", calc_alpha_mie)):
            ap, app = fn(lam)
            print(f"[{nombre:6s}] λ = {lam} nm →  α' = {ap:.3e} m³   α'' = {app:.3e} m³")
    graficar_comparacion()
    comparar_radios()
