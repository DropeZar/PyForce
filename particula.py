# -*- coding: utf-8 -*-
"""
particula.py

Polarizabilidad de una nanoesfera de oro con teoria de Mie, con la
permitividad del oro interpolada de la tabla completa de Johnson & Christy
(1972).

- calc_alpha: polarizabilidad dipolar EXACTA, del coeficiente de Mie a1. Es
  la que usa la simulacion (fuerzas.py trabaja en aproximacion dipolar).
- coeficientes_mie: todos los a_n, b_n (Bohren & Huffman).
- secciones_eficaces: Qext, Qsca, Qpr de Mie completo vs solo dipolo, para
  ver cuanto se pierde con la aproximacion dipolar.
- calc_alpha_kuwata: la aproximacion de Kuwata, solo para comparar.

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


# ─── Teoria de Mie ───────────────────────────────────────────────────────────

def coeficientes_mie(m, x, nmax=None):
    """
    Coeficientes de Mie a_n, b_n (n = 1..nmax), convencion de Bohren & Huffman.

    m : indice de refraccion relativo (n_particula / n_medio), complejo.
    x : parametro de tamano k*a (k = 2*pi*n_medio/lambda), real.

    Usa recurrencia DESCENDENTE para la derivada logaritmica D_n(mx), que es
    la forma numericamente estable para indices complejos (metal absorbente),
    y ascendente para las funciones de Riccati-Bessel psi_n, xi_n.
    """
    if nmax is None:  # criterio de Wiscombe
        nmax = int(np.ceil(x + 4 * x ** (1 / 3) + 2))
    mx = m * x
    n_ini = max(nmax, int(np.ceil(abs(mx)))) + 15
    D = np.zeros(n_ini + 1, dtype=complex)  # D[n] = D_n(mx)
    for n in range(n_ini, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)

    # psi_n = x j_n(x), chi_n = -x y_n(x), xi_n = psi_n - i chi_n
    psi_ant, chi_ant = np.sin(x), np.cos(x)          # n = 0
    psi, chi = psi_ant / x - chi_ant, chi_ant / x + psi_ant  # n = 1
    a = np.empty(nmax, dtype=complex)
    b = np.empty(nmax, dtype=complex)
    for n in range(1, nmax + 1):
        xi, xi_ant = psi - 1j * chi, psi_ant - 1j * chi_ant
        fa = D[n] / m + n / x
        fb = D[n] * m + n / x
        a[n - 1] = (fa * psi - psi_ant) / (fa * xi - xi_ant)
        b[n - 1] = (fb * psi - psi_ant) / (fb * xi - xi_ant)
        psi_ant, chi_ant, psi, chi = (psi, chi, (2 * n + 1) / x * psi - psi_ant,
                                      (2 * n + 1) / x * chi - chi_ant)
    return a, b


def _m_x_k(wavelength_nm, a, eps_m_local):
    lam = wavelength_nm * 1e-9
    n_m = np.sqrt(eps_m_local)
    k = 2 * np.pi * n_m / lam
    return np.sqrt(epsilon_gold(wavelength_nm)) / n_m, k * a, k


def calc_alpha(wavelength_nm, a=radio_particula, eps_m_local=eps_m):
    """
    Polarizabilidad dipolar electrica EXACTA (teoria de Mie), a partir del
    coeficiente a1:

        alpha = i * (6*pi / k^3) * a1      con k = 2*pi*n_medio/lambda

    Es la que usa la simulacion. Normalizacion: p = eps0 * eps_m * alpha * E,
    asi que para a -> 0 tiende a 4*pi*a^3 (eps - eps_m)/(eps + 2 eps_m)
    (Clausius-Mossotti) e incluye exacto el retardo y el amortiguamiento
    radiativo. Coincide con miepython a precision de maquina.

    Parametros: wavelength_nm (float o array), a = radio en m (por defecto
    sim_config.radio_particula), eps_m_local = permitividad del medio.
    Retorna (Re(alpha), Im(alpha)) en m^3.
    """
    wl = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    alpha = np.empty(wl.shape, dtype=complex)
    for i, w in enumerate(wl):
        m, x, k = _m_x_k(w, a, eps_m_local)
        a1 = coeficientes_mie(m, x, nmax=1)[0][0]
        alpha[i] = 1j * (6 * np.pi / k ** 3) * a1
    if alpha.size == 1:
        return np.real(alpha[0]), np.imag(alpha[0])
    return np.real(alpha), np.imag(alpha)


calc_alpha_mie = calc_alpha  # nombre viejo


def secciones_eficaces(wavelength_nm, a=radio_particula, eps_m_local=eps_m):
    """
    Eficiencias de extincion, scattering y presion de radiacion para una onda
    plana: Mie completo y solo el dipolo electrico (a1). Sirve para ver
    cuanto se pierde con la aproximacion dipolar que usa fuerzas.py.

    Devuelve {'completo': (Qext, Qsca, Qpr), 'dipolo': (Qext, Qsca, Qpr)}.
    """
    m, x, _ = _m_x_k(wavelength_nm, a, eps_m_local)
    an, bn = coeficientes_mie(m, x)

    def _q(an, bn):
        n = np.arange(1, len(an) + 1)
        qext = 2 / x ** 2 * np.sum((2 * n + 1) * np.real(an + bn))
        qsca = 2 / x ** 2 * np.sum((2 * n + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2))
        g = np.sum(n[:-1] * (n[:-1] + 2) / (n[:-1] + 1)
                   * np.real(an[:-1] * np.conj(an[1:]) + bn[:-1] * np.conj(bn[1:])))
        g += np.sum((2 * n + 1) / (n * (n + 1)) * np.real(an * np.conj(bn)))
        return qext, qsca, qext - 4 / x ** 2 * g

    cero = np.zeros(len(an) + 1, dtype=complex)
    a_dip = cero.copy(); a_dip[0] = an[0]
    return {'completo': _q(np.append(an, 0), np.append(bn, 0)),
            'dipolo': _q(a_dip, cero)}


def calc_alpha_kuwata(wavelength_nm, a=radio_particula, eps_m_local=eps_m):
    """
    Aproximacion de Kuwata (cuasi-estatica con correcciones de tamano), solo
    para comparar con Mie. Para NPs de oro de 80-100 nm se aleja mucho de Mie
    (Im(alpha) hasta ~2 veces menor), por eso ya no se usa.

    El termino de amortiguamiento radiativo es i*4*pi^2*eps_m^(3/2)*V/(3 lambda^3)
    (sale de k^3/(6 pi) con k = 2 pi n/lambda); antes tenia 4*pi.
    Retorna (Re(alpha), Im(alpha)) en m^3.
    """
    lam = wavelength_nm * 1e-9
    eps_p = epsilon_gold(wavelength_nm)
    V = (4 / 3) * np.pi * a ** 3
    x = np.pi * a / lam

    num = 1 - (1 / 10) * (eps_p - eps_m_local) * x ** 2
    denom_CM = (1 / 3) + eps_m_local / (eps_p - eps_m_local)
    denom_ret = -(1 / 30) * (eps_p + 10 * eps_m_local) * x ** 2
    denom_rr = -1j * (4 * np.pi ** 2 * eps_m_local ** 1.5 / 3) * (V / lam ** 3)

    alpha_K = V * num / (denom_CM + denom_ret + denom_rr)
    return np.real(alpha_K), np.imag(alpha_K)


# ─── Graficos ─────────────────────────────────────────────────────────────

def graficar_alpha(lam_min=400, lam_max=900, n_puntos=1001, a=radio_particula,
                    eps_m_local=eps_m,
                    longitudes_marcadas=((532, 'dona', 'tab:green'), (592, 'gauss', 'tab:orange')),
                    archivo=None):
    """
    Re(alpha) y Im(alpha) (Mie) en funcion de la longitud de onda, con las
    longitudes de onda de los haces marcadas y sus valores. Sombrea donde
    Re(alpha) < 0 y marca la resonancia (maximo de Im(alpha)).
    archivo: si se da, guarda la figura ahi (png) en vez de mostrarla.
    """
    import matplotlib.pyplot as plt

    lam_nm = np.linspace(lam_min, lam_max, n_puntos)
    ap, app = calc_alpha(lam_nm, a=a, eps_m_local=eps_m_local)
    esc = 1e-21  # se grafica en unidades de 1e-21 m^3

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lam_nm, ap / esc, color="tab:blue", lw=1.8,
            label=r"$\alpha'$ = Re($\alpha$): gradiente")
    ax.plot(lam_nm, app / esc, color="tab:red", lw=1.8,
            label=r"$\alpha''$ = Im($\alpha$): scattering y spin-curl")

    negativo = ap < 0
    if negativo.any():
        ax.fill_between(lam_nm, 0, 1, where=negativo, color="tab:blue", alpha=0.08,
                        transform=ax.get_xaxis_transform(), label=r"$\alpha' < 0$")
    lam_res = lam_nm[np.argmax(app)]
    ax.axvline(lam_res, color="tab:red", lw=0.8, ls="--")
    ax.text(lam_res + 4, 0.03, f"resonancia\n{lam_res:.0f} nm", color="tab:red",
            fontsize=8, va="bottom", transform=ax.get_xaxis_transform())

    for lam0, nombre, color in longitudes_marcadas:
        if not lam_min <= lam0 <= lam_max:
            continue
        p, pp = calc_alpha(lam0, a=a, eps_m_local=eps_m_local)
        ax.axvline(lam0, color=color, lw=1.2, ls=":")
        ax.plot([lam0, lam0], [p / esc, pp / esc], "o", color=color, ms=6, zorder=5)
        ax.annotate(f"{nombre} ({lam0} nm)\n" + r"$\alpha'$" + f" = {p / esc:.2f}\n"
                    + r"$\alpha''$" + f" = {pp / esc:.2f}",
                    xy=(lam0, pp / esc), xytext=(8 if lam0 > lam_res else -8, 8),
                    textcoords="offset points", fontsize=8, color=color,
                    ha="left" if lam0 > lam_res else "right")  # lejos de la resonancia

    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Longitud de onda [nm]")
    ax.set_ylabel(r"$\alpha$  [$10^{-21}$ m$^3$]")
    ax.set_title(f"Polarizabilidad (Mie, dipolo a1) — NP de Au de {2 * a * 1e9:.0f} nm "
                 f"de diámetro en agua")
    ax.set_xlim(lam_min, lam_max)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if archivo:
        fig.savefig(archivo, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def graficar_comparacion(lam_min=400, lam_max=900, n_puntos=400, a=radio_particula,
                          eps_m_local=eps_m, longitudes_marcadas=(532, 592)):
    """
    Compara Kuwata vs Mie: Re(alpha), Im(alpha) y |alpha| superpuestos,
    para un radio, en tres paneles.
    """
    import matplotlib.pyplot as plt

    lam_nm = np.linspace(lam_min, lam_max, n_puntos)
    apK, appK = calc_alpha_kuwata(lam_nm, a=a, eps_m_local=eps_m_local)
    apM, appM = calc_alpha(lam_nm, a=a, eps_m_local=eps_m_local)
    aK, aM = apK + 1j * appK, apM + 1j * appM

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    titulos = [r"Re($\alpha$)", r"Im($\alpha$)", r"$|\alpha|$"]
    datosK = [aK.real, aK.imag, np.abs(aK)]
    datosM = [aM.real, aM.imag, np.abs(aM)]

    for ax, tit, dK, dM in zip(axes, titulos, datosK, datosM):
        ax.plot(lam_nm, dK, label="Kuwata", color="tab:blue")
        ax.plot(lam_nm, dM, label="Mie", color="tab:orange", ls="--")
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
        f"Kuwata vs Mie — nanoesfera Au, a={a*1e9:.0f} nm   "
        f"(máx |α|: Kuwata {lam_res_K:.0f} nm, Mie {lam_res_M:.0f} nm)"
    )
    fig.tight_layout()
    plt.show()
    return fig, axes


def comparar_radios(radios_nm=(20, 30, 40, 50), lam_min=450, lam_max=750,
                     n_puntos=400, eps_m_local=eps_m):
    """
    |alpha(lambda)| para varios radios, Kuwata vs Mie, para ver cuanto se
    corre el maximo con el tamano.
    """
    import matplotlib.pyplot as plt

    lam_nm = np.linspace(lam_min, lam_max, n_puntos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    cmap = plt.cm.viridis(np.linspace(0, 1, len(radios_nm)))
    for a_nm, color in zip(radios_nm, cmap):
        a = a_nm * 1e-9
        apK, appK = calc_alpha_kuwata(lam_nm, a=a, eps_m_local=eps_m_local)
        apM, appM = calc_alpha(lam_nm, a=a, eps_m_local=eps_m_local)
        ax1.plot(lam_nm, np.abs(apK + 1j * appK), color=color, label=f"a={a_nm} nm")
        ax2.plot(lam_nm, np.abs(apM + 1j * appM), color=color, label=f"a={a_nm} nm")

    ax1.set_title("Kuwata")
    ax2.set_title("Mie")
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
    print(f"Radio = {radio_particula * 1e9:.0f} nm, medio n = {np.sqrt(eps_m):.2f}")
    for lam in (532, 592):
        for nombre, fn in (("Mie", calc_alpha), ("Kuwata", calc_alpha_kuwata)):
            ap, app = fn(lam)
            print(f"[{nombre:6s}] λ = {lam} nm →  α' = {ap:.3e} m³   α'' = {app:.3e} m³")
        q = secciones_eficaces(lam)
        print(f"         Qext, Qsca, Qpr  Mie completo: {np.round(q['completo'], 3)}   "
              f"solo dipolo: {np.round(q['dipolo'], 3)}")
    import os
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'resultados', 'polarizabilidad')
    os.makedirs(carpeta, exist_ok=True)
    graficar_alpha(archivo=os.path.join(carpeta, 'alpha_mie.png'))
    graficar_comparacion()
    comparar_radios()
