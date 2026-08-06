# -*- coding: utf-8 -*-
"""
particula.py

Polarizabilidad de Kuwata para una nanoesfera metalica, con la
permitividad del oro interpolada de la tabla completa de
Johnson & Christy (1972).

Esta es la version validada del calculo (antes vivia en
polarizabilidad_Kuwata.py) -- reemplaza al calc_alpha/epsilon_gold
que estaba duplicado y desactualizado dentro de main_fuerza.py.

Uso tipico:
    from particula import calc_alpha
    alpha_p, alpha_pp = calc_alpha(592)          # usa radio y medio de config.py
    alpha_p, alpha_pp = calc_alpha(532, a=50e-9) # radio distinto puntual
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
        Radio de la particula en metros. Por defecto, config.radio_particula.
    eps_m_local : float
        Permitividad del medio. Por defecto, config.eps_m.

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


def graficar_alpha(lam_min=400, lam_max=900, n_puntos=700, a=radio_particula,
                    eps_m_local=eps_m, longitudes_marcadas=(592, 1064)):
    """Grafico de diagnostico de Re/Im/|alpha| en funcion de la longitud de onda."""
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


if __name__ == '__main__':
    for lam in (592, 1064):
        ap, app = calc_alpha(lam)
        print(f"λ = {lam} nm →  α' = {ap:.3e} m³   α'' = {app:.3e} m³")
        print(f"             ratio α''/α' = {app/ap:.3f}  "
              f"({'absorbente' if app/ap > 0 else 'dieléctrica'})")
    graficar_alpha()
