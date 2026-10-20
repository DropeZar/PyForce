# -*- coding: utf-8 -*-
"""
fuerzas_gauss_interfaz.py

Printing optico con el haz gaussiano (calibrado, ver calibracion.json):
fuerzas sobre la NP de oro sin y con la interfaz agua/vidrio del
portaobjetos, con el foco sobre la superficie del vidrio (z = 0).

Geometria: objetivo de inmersion en agua -> agua (donde esta la NP) ->
portaobjetos de vidrio, con aceite de indice igualado abajo, asi que el
vidrio se trata como semi-infinito. El haz se propaga hacia +z y el vidrio
ocupa z > 0. La NP (radio a) llega al vidrio con el centro en z = -a: ese
es el "plano de contacto", donde queda impresa. En los graficos +z va
hacia ABAJO (el haz baja desde el objetivo hacia el vidrio).

Fuerzas: gradiente + scattering + spin-curl (fuerzas.py), con las derivadas
fuera del plano exactas (interfaz.py).

Salida en resultados/gauss_interfaz/:
    fuerzas_xz.png    mapas XZ sin y con vidrio
    fuerzas_xy.png    plano de contacto (z = -a) sin/con vidrio y diferencia
    resumen.md        fuerza hacia el vidrio en el contacto (por componente) y
                      potencia necesaria para llegar a F_IMPRESION_pN

Uso:  python fuerzas_gauss_interfaz.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from beams import gauss
from fuerzas import calcular_factor_E_foco, fuerzas_xy, fuerzas_xz
from interfaz import FocoConInterfaz, ejes_xy, ejes_xz
from particula import calc_alpha
from sim_config import (x_steps, z_steps, x_range, z_range, eps_m, n_vidrio,
                        radio_particula, dx_fuerzas, dy_fuerzas, dz_fuerzas,
                        F_IMPRESION_pN)

Z_VIDRIO = 0.0   # posicion de la superficie del vidrio respecto del foco (nm)
Z_XZ = (-1100, 100)  # ventana en z de los mapas XZ (nm): la zona de agua por donde llega la NP
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'resultados', 'gauss_interfaz')

pN = 1e12
a_nm = radio_particula * 1e9
z_contacto = Z_VIDRIO - a_nm
factor_E = calcular_factor_E_foco(gauss.n)
alpha_p, alpha_pp = calc_alpha(gauss.wavelength)
CASOS = {'sin vidrio': None, 'con vidrio': Z_VIDRIO}


# ---------------------------------------------------------------------------
# Calculo
# ---------------------------------------------------------------------------
def _SI(E, por_nm=False):
    return tuple(e * factor_E * (1e9 if por_nm else 1) for e in E)


def _foco(z_int):
    return FocoConInterfaz(gauss.NA, gauss.n, gauss.h, gauss.w0, gauss.wavelength,
                           gauss.gamma, gauss.beta, gauss.I0,
                           n2=None if z_int is None else n_vidrio, z_int=z_int)


def mapas(z_int):
    """Fuerzas (pN) en el plano XZ y en el plano de contacto XY (z = -a)."""
    foco = _foco(z_int)
    x_xy = ejes_xy(x_steps, x_range)
    x_xz = ejes_xz(x_steps, z_steps, x_range, z_range)[0]
    z_xz = np.arange(Z_XZ[0], Z_XZ[1] + 0.1, z_steps)
    xy = fuerzas_xy(*_SI(foco.plano_xy(x_xy, z_contacto)), dx_fuerzas, dy_fuerzas,
                    dz_fuerzas, alpha_p, alpha_pp, eps_m,
                    dE_dz=_SI(foco.derivada_xy_z(x_xy, z_contacto), por_nm=True))
    xz = fuerzas_xz(*_SI(foco.plano_xz(x_xz, z_xz)), dx_fuerzas, dz_fuerzas,
                    alpha_p, alpha_pp, eps_m,
                    dE_dy=_SI(foco.derivada_xz_y(x_xz, z_xz), por_nm=True))
    xz = [F * pN for F in xz]
    # la NP no puede tener el centro a menos de a del vidrio (aunque se
    # calcule "sin vidrio", se recorta igual para comparar la misma region)
    afuera = (z_xz > z_contacto)[:, None]
    xz = [np.where(afuera, np.nan, F) for F in xz]
    return dict(x_xy=x_xy, xy=[F * pN for F in xy], x_xz=x_xz, z_xz=z_xz, xz=xz)


def fuerza_en_contacto(z_int):
    """Fz (pN) y sus componentes (grad, scat, spin-curl) en el contacto, sobre el eje."""
    x = np.array([-1.0, 0.0, 1.0]) * x_steps
    z = z_contacto + np.array([-1.0, 0.0, 1.0]) * z_steps
    E = _foco(z_int).plano_xz(x, z)
    r = fuerzas_xz(*_SI(E), dx_fuerzas, dz_fuerzas, alpha_p, alpha_pp, eps_m)
    return {k: r[i][1, 1] * pN for k, i in (('total', 1), ('grad', 3), ('scat', 5), ('sc', 7))}


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def _sombrear_vidrio(ax, z_fondo):
    ax.axhspan(Z_VIDRIO, z_fondo, color='lightsteelblue', zorder=3)
    ax.axhspan(z_contacto, Z_VIDRIO, color='0.85', zorder=3)


def graficar_xz(m, archivo):
    vmax = max(np.nanmax(np.hypot(mm['xz'][0], mm['xz'][1])) for mm in m.values())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True)
    for ax, (nombre, mm) in zip(axes, m.items()):
        x, z = mm['x_xz'], mm['z_xz']
        Fx, Fz = mm['xz'][0], mm['xz'][1]
        im = ax.imshow(np.hypot(Fx, Fz), extent=[x[0], x[-1], z[0], z[-1]], origin='lower',
                       cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax.streamplot(x, z, np.ma.masked_invalid(Fx), np.ma.masked_invalid(Fz),
                      density=1.4, color='white', linewidth=0.6, arrowsize=0.7)
        if CASOS[nombre] is not None:
            _sombrear_vidrio(ax, z[-1])
            ax.text(0, (Z_VIDRIO + z[-1]) / 2, f'vidrio (n = {n_vidrio})', ha='center',
                    va='center', zorder=4, fontsize=8)
        else:
            ax.axhspan(z_contacto, z[-1], color='0.85', zorder=3)
        ax.axhline(z_contacto, color='tab:red', lw=0.8, ls='--', zorder=4)
        ax.set_xlim(-700, 700)
        ax.set_ylim(z[-1], z[0])  # +z hacia abajo
        ax.set_title(nombre)
        ax.set_xlabel('x (nm)')
    axes[0].set_ylabel('z (nm)   (haz hacia abajo)')
    fig.colorbar(im, ax=axes, label='|F| (pN)', shrink=0.85)
    fig.suptitle(f'Haz gaussiano, plano XZ (foco en z = 0) — NP Au a = {a_nm:.0f} nm, '
                 f'λ = {gauss.wavelength} nm, P = 1 mW\n'
                 f'línea roja: centro de la NP apoyada en el vidrio (z = −a)', fontsize=10)
    fig.savefig(archivo, dpi=200, bbox_inches='tight')
    plt.close(fig)


def graficar_xy(m, archivo):
    sin, con = m['sin vidrio'], m['con vidrio']
    x = sin['x_xy']
    ext = [x[0], x[-1], x[0], x[-1]]
    F = {k: np.hypot(mm['xy'][0], mm['xy'][1]) for k, mm in m.items()}
    dif = (np.hypot(con['xy'][0] - sin['xy'][0], con['xy'][1] - sin['xy'][1])
           / F['sin vidrio'].max() * 100)
    vmax = max(f.max() for f in F.values())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (nombre, mm) in zip(axes, m.items()):
        im = ax.imshow(F[nombre], extent=ext, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
        ax.streamplot(x, x, mm['xy'][0], mm['xy'][1], density=1.3, color='white',
                      linewidth=0.6, arrowsize=0.7)
        ax.set_title(f'{nombre}')
        fig.colorbar(im, ax=ax, label='|F| (pN)')
    im = axes[2].imshow(dif, extent=ext, origin='lower', cmap='magma')
    axes[2].set_title('|F_con − F_sin| / max|F_sin|')
    fig.colorbar(im, ax=axes[2], label='%')
    for ax in axes:
        ax.set_xlim(-500, 500); ax.set_ylim(-500, 500)
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    fig.suptitle(f'Fuerza lateral en el plano de contacto (z = −a = {z_contacto:.0f} nm)')
    fig.tight_layout()
    fig.savefig(archivo, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(CARPETA, exist_ok=True)
    print(f"gauss: NA={gauss.NA}, n={gauss.n}, w0={gauss.w0:.3f} mm, I0={gauss.I0:.3f} mW/cm2, "
          f"lambda={gauss.wavelength} nm;  alpha' = {alpha_p:.3e}, alpha'' = {alpha_pp:.3e} m^3")

    m = {nombre: mapas(z_int) for nombre, z_int in CASOS.items()}
    graficar_xz(m, os.path.join(CARPETA, 'fuerzas_xz.png'))
    graficar_xy(m, os.path.join(CARPETA, 'fuerzas_xy.png'))

    contacto = {nombre: fuerza_en_contacto(z_int) for nombre, z_int in CASOS.items()}

    # resumen
    lineas = [
        '# Printing con el haz gaussiano: fuerzas con y sin el vidrio', '',
        f'- Haz: NA = {gauss.NA}, n = {gauss.n} (agua), λ = {gauss.wavelength} nm, '
        f'polarización lineal en x, w0 = {gauss.w0:.3f} mm, I0 = {gauss.I0:.3f} mW/cm² '
        f'(calibrado: FWHM = 330 nm, P = 1 mW en la muestra).',
        f'- Partícula: Au, a = {a_nm:.0f} nm, en agua; α\' = {alpha_p:.3e} m³, '
        f'α\'\' = {alpha_pp:.3e} m³ (Mie, dipolo a1).',
        f'- Vidrio: n = {n_vidrio}, semi-infinito, con la superficie en el foco (z = 0). '
        f'Contacto: centro de la NP en z = −a = {z_contacto:.0f} nm.',
        '- Fuerza total = gradiente + scattering + spin-curl.', '',
        '## En el punto de contacto sobre el eje (x = y = 0, z = −a)', '',
        '| caso | Fz total (pN) | gradiente | scattering | spin-curl |',
        '|---|---|---|---|---|',
    ]
    for nombre, p in contacto.items():
        lineas.append(f"| {nombre} | {p['total']:.3f} | {p['grad']:.3f} | "
                      f"{p['scat']:.3f} | {p['sc']:.3f} |")
    lineas += ['', f'## Potencia para imprimir (Fz en el contacto = {F_IMPRESION_pN} pN)', '',
               '| caso | Fz en el contacto a 1 mW (pN) | potencia necesaria (mW) |',
               '|---|---|---|']
    for nombre, p in contacto.items():
        f1 = p['total']
        lineas.append(f'| {nombre} | {f1:.3f} | {F_IMPRESION_pN / f1:.2f} |')
    lineas += ['', 'Fuerzas a P = 1 mW en la muestra (escalan linealmente con P). '
               'Fz > 0 empuja la NP hacia el vidrio.', '']
    with open(os.path.join(CARPETA, 'resumen.md'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lineas))
    print('\n'.join(lineas))
