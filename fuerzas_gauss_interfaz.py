# -*- coding: utf-8 -*-
"""
fuerzas_gauss_interfaz.py

Fuerzas del haz gaussiano (calibrado, ver calibracion.json) sobre la NP de
oro, sin y con la interfaz agua/vidrio del portaobjetos.

Geometria: objetivo de inmersion en agua -> agua (donde esta la particula)
-> portaobjetos de vidrio, con aceite de indice igualado abajo, asi que el
vidrio se trata como semi-infinito. El haz se propaga hacia +z y el vidrio
empieza en z = d (d = distancia del foco al vidrio). En los graficos +z va
hacia ABAJO (el haz baja desde el objetivo hacia el vidrio).

La particula (radio a) no puede estar a menos de a del vidrio: esa franja y
el vidrio se dejan en blanco en los mapas.

Salida en resultados/gauss_interfaz/:
    fuerzas_xz.png    mapas XZ sin y con interfaz (d = D_PRINCIPAL)
    fuerzas_xy.png    plano focal sin/con interfaz y diferencia
    perfiles.png      I y Fz sobre el eje, Fx en el plano focal, para cada d
    resumen.md        rigidez transversal, Fz en el foco y equilibrios axiales

Uso:  python fuerzas_gauss_interfaz.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from beams import gauss
from fuerzas import calcular_factor_E_foco, fuerzas_xy, fuerzas_xz
from interfaz import FocoConInterfaz, campo_gauss
from particula import calc_alpha
from sim_config import (x_steps, z_steps, x_range, z_range, eps_m, n_vidrio,
                        radio_particula, dx_fuerzas, dy_fuerzas, dz_fuerzas)

DISTANCIAS = (250, 500, 1000)   # foco -> vidrio (nm)
D_PRINCIPAL = 500               # la de los mapas
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'resultados', 'gauss_interfaz')

pN = 1e12
kT = 1.380649e-23 * 295  # J
a_nm = radio_particula * 1e9
factor_E = calcular_factor_E_foco(gauss.n)
alpha_p, alpha_pp = calc_alpha(gauss.wavelength)


# ---------------------------------------------------------------------------
# Calculo
# ---------------------------------------------------------------------------
def _a_SI(E):
    return tuple(e * factor_E for e in E)


def mapas(d):
    """Fuerzas XY (z = 0) y XZ en pN para el foco a distancia d del vidrio (None = sin vidrio)."""
    c = campo_gauss(gauss, x_steps, z_steps, x_range, z_range,
                    n2=None if d is None else n_vidrio, z_int=d)
    Fx, Fy, *_ = fuerzas_xy(*_a_SI(c['XY']), dx_fuerzas, dy_fuerzas, dz_fuerzas,
                            alpha_p, alpha_pp, eps_m)
    Fxz, Fzz, *_ = fuerzas_xz(*_a_SI(c['XZ']), dx_fuerzas, dz_fuerzas,
                              alpha_p, alpha_pp, eps_m)
    Fxz, Fzz = Fxz * pN, Fzz * pN
    if d is not None:  # la particula no puede estar a menos de a del vidrio
        afuera = (c['z_xz'] > d - a_nm)[:, None]
        Fxz = np.where(afuera, np.nan, Fxz)
        Fzz = np.where(afuera, np.nan, Fzz)
    return dict(x_xy=c['x_xy'], Fx_xy=Fx * pN, Fy_xy=Fy * pN,
                x_xz=c['x_xz'], z_xz=c['z_xz'], Fx_xz=Fxz, Fz_xz=Fzz)


def perfil_eje(d, z):
    """I (mW/cm^2) y Fz (pN) sobre el eje x = y = 0, en las posiciones z."""
    foco = FocoConInterfaz(gauss.NA, gauss.n, gauss.h, gauss.w0, gauss.wavelength,
                           gauss.gamma, gauss.beta, gauss.I0,
                           n2=None if d is None else n_vidrio, z_int=d)
    x = np.array([-1.0, 0.0, 1.0]) * x_steps
    E = foco.plano_xz(x, z)
    I = sum(np.abs(e[:, 1]) ** 2 for e in E)
    _, Fz, *_ = fuerzas_xz(*_a_SI(E), dx_fuerzas, dz_fuerzas, alpha_p, alpha_pp, eps_m)
    Fz = Fz[:, 1] * pN
    if d is not None:
        Fz = np.where(z > d - a_nm, np.nan, Fz)
        I = np.where(z > d, np.nan, I)
    return I, Fz


def pozos_axiales(z, Fz):
    """
    Pozos estables de U(z) = -integral de Fz sobre el eje: devuelve
    [(z_eq, profundidad en kT)], con profundidad = la menor de las barreras a
    ambos lados (los bordes del dominio cuentan como barrera).
    """
    ok = ~np.isnan(Fz)
    z, Fz = z[ok], Fz[ok]
    U = -np.concatenate([[0], np.cumsum((Fz[1:] + Fz[:-1]) / 2 * np.diff(z))]) * 1e-21  # pN*nm -> J
    out = []
    for i in range(1, len(z) - 1):
        if U[i] < U[i - 1] and U[i] <= U[i + 1]:
            izq, der = U[:i].max(), U[i + 1:].max()
            out.append((z[i], (min(izq, der) - U[i]) / kT))
    return out


def profundidad_transversal(x, Fx):
    """Profundidad en kT del pozo transversal en el plano focal (mitad x > 0)."""
    der = x >= 0
    U = -np.concatenate([[0], np.cumsum((Fx[der][1:] + Fx[der][:-1]) / 2 * np.diff(x[der]))]) * 1e-21
    return (np.nanmax(U) - U[0]) / kT


def rigidez_x(m):
    """k_x = -dFx/dx en el centro del plano focal (pN/um)."""
    fila = m['Fx_xy'][len(m['x_xy']) // 2]
    i = len(fila) // 2
    return -(fila[i + 1] - fila[i - 1]) / (2 * x_steps) * 1e3


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def _nombre(d):
    return 'sin vidrio' if d is None else f'vidrio a {d} nm'


def graficar_xz(casos, archivo):
    vmax = max(np.nanmax(np.hypot(m['Fx_xz'], m['Fz_xz'])) for m in casos.values())
    fig, axes = plt.subplots(1, len(casos), figsize=(5.2 * len(casos), 5), sharey=True)
    for ax, (d, m) in zip(axes, casos.items()):
        x, z = m['x_xz'], m['z_xz']
        F = np.hypot(m['Fx_xz'], m['Fz_xz'])
        im = ax.imshow(F, extent=[x[0], x[-1], z[0], z[-1]], origin='lower',
                       cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax.streamplot(x, z, np.ma.masked_invalid(m['Fx_xz']), np.ma.masked_invalid(m['Fz_xz']),
                      density=1.4, color='white', linewidth=0.6, arrowsize=0.7)
        if d is not None:
            ax.axhspan(d, z[-1], color='lightsteelblue', zorder=3)
            ax.axhspan(d - a_nm, d, color='0.85', zorder=3)
            ax.text(0, (d + z[-1]) / 2, f'vidrio (n = {n_vidrio})', ha='center',
                    va='center', zorder=4, fontsize=9)
        ax.set_xlim(-600, 600)
        ax.set_ylim(z[-1], z[0])  # +z hacia abajo
        ax.set_title(_nombre(d))
        ax.set_xlabel('x (nm)')
    axes[0].set_ylabel('z (nm)   (haz hacia abajo)')
    fig.colorbar(im, ax=axes, label='|F| (pN)', shrink=0.85)
    fig.suptitle(f'Haz gaussiano en agua, plano XZ — NP Au a = {a_nm:.0f} nm, λ = {gauss.wavelength} nm')
    fig.savefig(archivo, dpi=200, bbox_inches='tight')
    plt.close(fig)


def graficar_xy(sin, con, d, archivo):
    x = sin['x_xy']
    ext = [x[0], x[-1], x[0], x[-1]]
    F_sin = np.hypot(sin['Fx_xy'], sin['Fy_xy'])
    F_con = np.hypot(con['Fx_xy'], con['Fy_xy'])
    dif = np.hypot(con['Fx_xy'] - sin['Fx_xy'], con['Fy_xy'] - sin['Fy_xy']) / F_sin.max() * 100
    vmax = max(F_sin.max(), F_con.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, F, m, tit in ((axes[0], F_sin, sin, 'sin vidrio'), (axes[1], F_con, con, _nombre(d))):
        im = ax.imshow(F, extent=ext, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
        ax.streamplot(x, x, m['Fx_xy'], m['Fy_xy'], density=1.3, color='white',
                      linewidth=0.6, arrowsize=0.7)
        ax.set_title(f'{tit} (z = 0)')
        fig.colorbar(im, ax=ax, label='|F| (pN)')
    im = axes[2].imshow(dif, extent=ext, origin='lower', cmap='magma')
    axes[2].set_title('|F_con − F_sin| / max|F_sin|')
    fig.colorbar(im, ax=axes[2], label='%')
    for ax in axes:
        ax.set_xlim(-500, 500); ax.set_ylim(-500, 500)
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    fig.suptitle('Haz gaussiano, plano focal')
    fig.tight_layout()
    fig.savefig(archivo, dpi=200, bbox_inches='tight')
    plt.close(fig)


def graficar_perfiles(z, perfiles, transv, I_ref, archivo):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
    colores = dict(zip(perfiles, ['k'] + list(plt.cm.plasma(np.linspace(0.15, 0.8, len(perfiles) - 1)))))
    for d, (I, Fz) in perfiles.items():
        ax1.plot(z, I / I_ref, color=colores[d], label=_nombre(d))
        ax2.plot(z, Fz, color=colores[d], label=_nombre(d))
        if d is not None:
            for ax in (ax1, ax2):
                ax.axvline(d, color=colores[d], ls=':', lw=0.8)
    for d, (x, Fx) in transv.items():
        ax3.plot(x, Fx, color=colores[d], label=_nombre(d))
    ax1.set_ylabel('I / I_pico sin vidrio')
    ax1.set_title('Intensidad sobre el eje')
    ax2.set_ylabel('Fz (pN)   (+ = hacia el vidrio)')
    ax2.set_title('Fuerza axial sobre el eje')
    ax2.axhline(0, color='gray', lw=0.5)
    for ax in (ax1, ax2):
        ax.set_xlabel('z (nm)   (foco en 0)')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('Fx (pN)')
    ax3.set_title('Fuerza transversal en el plano focal (y = 0)')
    ax3.axhline(0, color='gray', lw=0.5)
    ax3.set_xlim(-600, 600)
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(archivo, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(CARPETA, exist_ok=True)
    print(f"gauss: NA={gauss.NA}, n={gauss.n}, w0={gauss.w0:.3f} mm, I0={gauss.I0:.3f} mW/cm2, "
          f"lambda={gauss.wavelength} nm;  alpha' = {alpha_p:.3e}, alpha'' = {alpha_pp:.3e} m^3")

    casos = (None,) + DISTANCIAS
    m = {d: mapas(d) for d in casos}
    graficar_xz({None: m[None], D_PRINCIPAL: m[D_PRINCIPAL]},
                os.path.join(CARPETA, 'fuerzas_xz.png'))
    graficar_xy(m[None], m[D_PRINCIPAL], D_PRINCIPAL, os.path.join(CARPETA, 'fuerzas_xy.png'))

    z = np.arange(-1500, max(DISTANCIAS) + 1, 2.0)
    perfiles = {d: perfil_eje(d, z) for d in casos}
    I_ref = np.nanmax(perfiles[None][0])
    transv = {}
    for d in casos:
        fila = len(m[d]['z_xz']) // 2  # z = 0
        transv[d] = (m[d]['x_xz'], m[d]['Fx_xz'][fila])
    graficar_perfiles(z, perfiles, transv, I_ref, os.path.join(CARPETA, 'perfiles.png'))

    # resumen
    lineas = [
        '# Haz gaussiano: fuerzas con y sin el vidrio del portaobjetos', '',
        f'- Haz: NA = {gauss.NA}, n = {gauss.n} (agua), λ = {gauss.wavelength} nm, '
        f'polarización lineal en x, w0 = {gauss.w0:.3f} mm, I0 = {gauss.I0:.3f} mW/cm² '
        f'(calibrado: FWHM = 330 nm, P = 1 mW en la muestra).',
        f'- Partícula: Au, a = {a_nm:.0f} nm, en agua; α\' = {alpha_p:.3e} m³, '
        f'α\'\' = {alpha_pp:.3e} m³ (Kuwata).',
        f'- Vidrio: n = {n_vidrio}, semi-infinito, a distancia d del foco (hacia +z).', '',
        '| caso | I en el foco / sin vidrio | Fz en el foco (pN) | k_x en el foco (pN/µm) | '
        'pozo transversal (kT) | pozos axiales sobre el eje: z (profundidad) |',
        '|---|---|---|---|---|---|',
    ]
    i0 = np.argmin(np.abs(z))
    for d in casos:
        I, Fz = perfiles[d]
        pozos = [(zp, p) for zp, p in pozos_axiales(z, Fz) if p >= 0.1]
        txt = ', '.join(f'{zp:.0f} nm ({p:.1f} kT)' for zp, p in pozos) or 'ninguno'
        lineas.append(f'| {_nombre(d)} | {I[i0] / I_ref:.3f} | {Fz[i0]:.3f} | '
                      f'{rigidez_x(m[d]):.2f} | {profundidad_transversal(*transv[d]):.1f} | {txt} |')
    lineas += ['', f'Todo a P = 1 mW en la muestra (las fuerzas escalan linealmente con P). '
               'Fz > 0 empuja hacia el vidrio. Pozos axiales buscados entre z = −1500 nm '
               'y d − a (la partícula no puede tocar el vidrio), con profundidad ≥ 0.1 kT '
               '(T = 295 K); "ninguno" quiere decir que la fuerza de scattering gana en '
               'todo el eje y la partícula termina contra el vidrio.', '']
    with open(os.path.join(CARPETA, 'resumen.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lineas))
    print('\n'.join(lineas))
