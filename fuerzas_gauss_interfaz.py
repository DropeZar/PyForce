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
    perfiles.png      Fz sobre el eje al acercarse al vidrio, Fx en el plano
                      de contacto y sus componentes
    resumen.md        fuerza de impresion y guiado lateral en el contacto

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
                        radio_particula, dx_fuerzas, dy_fuerzas, dz_fuerzas)

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


def perfil_eje(z_int, z):
    """Fz (pN) y sus componentes (grad, scat, spin-curl) sobre el eje x = y = 0."""
    x = np.array([-1.0, 0.0, 1.0]) * x_steps
    E = _foco(z_int).plano_xz(x, z)
    r = fuerzas_xz(*_SI(E), dx_fuerzas, dz_fuerzas, alpha_p, alpha_pp, eps_m)
    return {k: r[i][:, 1] * pN for k, i in (('total', 1), ('grad', 3), ('scat', 5), ('sc', 7))}


def en_el_centro(m, i):
    """Valor de la componente i de las fuerzas XY en el eje (x = y = 0)."""
    x = m['x_xy']
    return m['xy'][i][np.argmin(np.abs(x)), np.argmin(np.abs(x))]


def guiado_lateral(m):
    """
    En el plano de contacto, sobre y = 0: pendiente -dFx/dx en el eje
    (pN/um), fuerza lateral maxima hacia el eje (pN) y radio de guiado
    (primer cero de Fx para x > 0: dentro de ese radio la NP es llevada al eje).
    """
    x = m['x_xy']
    fila = m['xy'][0][np.argmin(np.abs(x))]
    i0 = np.argmin(np.abs(x))
    pendiente = -(fila[i0 + 1] - fila[i0 - 1]) / (x[i0 + 1] - x[i0 - 1]) * 1e3
    der = slice(i0 + 1, None)
    xs, fs = x[der], fila[der]
    cero = np.where(fs > 0)[0]
    radio = xs[cero[0]] if len(cero) else np.nan
    return pendiente, -fs.min(), radio


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


def graficar_perfiles(z, eje, m, archivo):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
    estilos = {'sin vidrio': dict(color='k', ls='--'), 'con vidrio': dict(color='tab:blue')}
    for nombre, p in eje.items():
        ax1.plot(z, p['total'], label=nombre, **estilos[nombre])
    for comp, col, lab in (('grad', 'tab:green', 'gradiente'), ('scat', 'tab:red', 'scattering'),
                           ('sc', 'tab:purple', 'spin-curl')):
        ax1.plot(z, eje['con vidrio'][comp], color=col, lw=0.9, alpha=0.8,
                 label=f'{lab} (con vidrio)')
    ax1.axvline(z_contacto, color='tab:red', ls=':', lw=0.8)
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.set_xlabel('z del centro de la NP (nm)   (vidrio en 0)')
    ax1.set_ylabel('Fz (pN)   (+ = hacia el vidrio)')
    ax1.set_title('Fuerza axial sobre el eje')

    for nombre, mm in m.items():
        x = mm['x_xy']
        fila = mm['xy'][0][np.argmin(np.abs(x))]
        ax2.plot(x, fila, label=nombre, **estilos[nombre])
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('Fx (pN)')
    ax2.set_title(f'Fuerza lateral en el plano de contacto (y = 0)')

    con = m['con vidrio']
    x = con['x_xy']
    i0 = np.argmin(np.abs(x))
    for i, col, lab in ((0, 'tab:blue', 'total'), (2, 'tab:green', 'gradiente'),
                        (4, 'tab:red', 'scattering'), (6, 'tab:purple', 'spin-curl')):
        ax3.plot(x, con['xy'][i][i0], color=col, label=lab)
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('Fx (pN)')
    ax3.set_title('Componentes de Fx en el contacto (con vidrio)')
    for ax in (ax2, ax3):
        ax.set_xlim(-600, 600)
        ax.axhline(0, color='gray', lw=0.5)
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

    m = {nombre: mapas(z_int) for nombre, z_int in CASOS.items()}
    graficar_xz(m, os.path.join(CARPETA, 'fuerzas_xz.png'))
    graficar_xy(m, os.path.join(CARPETA, 'fuerzas_xy.png'))

    z = np.arange(-1500, z_contacto + 0.1, 2.0)
    eje = {nombre: perfil_eje(z_int, z) for nombre, z_int in CASOS.items()}
    graficar_perfiles(z, eje, m, os.path.join(CARPETA, 'perfiles.png'))

    # resumen
    lineas = [
        '# Printing con el haz gaussiano: fuerzas con y sin el vidrio', '',
        f'- Haz: NA = {gauss.NA}, n = {gauss.n} (agua), λ = {gauss.wavelength} nm, '
        f'polarización lineal en x, w0 = {gauss.w0:.3f} mm, I0 = {gauss.I0:.3f} mW/cm² '
        f'(calibrado: FWHM = 330 nm, P = 1 mW en la muestra).',
        f'- Partícula: Au, a = {a_nm:.0f} nm, en agua; α\' = {alpha_p:.3e} m³, '
        f'α\'\' = {alpha_pp:.3e} m³ (Kuwata).',
        f'- Vidrio: n = {n_vidrio}, semi-infinito, con la superficie en el foco (z = 0). '
        f'Contacto: centro de la NP en z = −a = {z_contacto:.0f} nm.',
        '- Fuerza total = gradiente + scattering + spin-curl.', '',
        '## En el punto de contacto sobre el eje (x = y = 0, z = −a)', '',
        '| caso | Fz total (pN) | gradiente | scattering | spin-curl |',
        '|---|---|---|---|---|',
    ]
    i_c = np.argmin(np.abs(z - z_contacto))
    for nombre, p in eje.items():
        lineas.append(f"| {nombre} | {p['total'][i_c]:.3f} | {p['grad'][i_c]:.3f} | "
                      f"{p['scat'][i_c]:.3f} | {p['sc'][i_c]:.3f} |")
    lineas += ['', '## Guiado lateral en el plano de contacto (y = 0)', '',
               '| caso | −dFx/dx en el eje (pN/µm) | Fx máxima hacia el eje (pN) | '
               'radio de guiado (nm) |', '|---|---|---|---|']
    for nombre, mm in m.items():
        k, fmax, radio = guiado_lateral(mm)
        lineas.append(f'| {nombre} | {k:.2f} | {fmax:.3f} | {radio:.0f} |')
    lineas += ['', 'Todo a P = 1 mW en la muestra (las fuerzas escalan linealmente con P). '
               'Fz > 0 empuja la NP hacia el vidrio. Radio de guiado: distancia al eje '
               'dentro de la cual la fuerza lateral lleva la NP hacia el eje (primer cero '
               'de Fx).', '']
    with open(os.path.join(CARPETA, 'resumen.md'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lineas))
    print('\n'.join(lineas))
