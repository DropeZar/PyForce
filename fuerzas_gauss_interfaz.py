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

Campos: PyFocus (sim.custom con la interfaz en el foco), via
pyfocus_wrapper.campo_foco_vidrio. "Sin vidrio" se calcula igual, con
n2 = n1, para comparar con el mismo metodo y la misma grilla.

Fuerzas: gradiente + scattering + spin-curl (fuerzas.py). Las derivadas
fuera del plano se aproximan como en fuerzas.py (con div E = 0): en el
plano XZ es exacto para este haz (polarizado en x); en el plano XY el
spin-curl lateral queda aproximado.

Salida en resultados/gauss_interfaz/:
    fuerzas_xz.png    mapas XZ sin y con vidrio
    fuerzas_xy.png    plano de contacto (z = -a) sin/con vidrio y diferencia
    resumen.md        fuerza hacia el vidrio en el contacto (por componente) y
                      potencia necesaria para llegar a F_IMPRESION_pN

Uso:  python fuerzas_gauss_interfaz.py     (~15 min: sim.custom es lento)
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from beams import gauss
from fuerzas import calcular_factor_E_foco, fuerzas_xy, fuerzas_xz
from particula import calc_alpha
from pyfocus_wrapper import campo_foco_vidrio
from sim_config import eps_m, n_vidrio, radio_particula, F_IMPRESION_pN

# Grilla de sim.custom: con estos valores quedan 171 puntos en x (paso
# ~7.1 nm) y 121 en z (paso ~10.1 nm), los dos impares para que x = 0 y la
# altura del plano de contacto caigan justo en la grilla.
X_RANGE, X_STEPS = 1210, 5
Z_RANGE, Z_STEPS = 1210, 5
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'resultados', 'gauss_interfaz')

pN = 1e12
a_nm = radio_particula * 1e9
z_contacto = -a_nm  # vidrio en z = 0
factor_E = calcular_factor_E_foco(gauss.n)
alpha_p, alpha_pp = calc_alpha(gauss.wavelength)
CASOS = {'sin vidrio': gauss.n, 'con vidrio': n_vidrio}  # n2 de cada caso


# ---------------------------------------------------------------------------
# Calculo
# ---------------------------------------------------------------------------
def mapas(n2):
    """
    Fuerzas (pN) en el plano de contacto XY (z = -a) y en el plano XZ
    centrado en esa altura, para el vidrio de indice n2 en z = 0.
    """
    c = campo_foco_vidrio(gauss.NA, gauss.n, n2, gauss.h, gauss.w0, gauss.wavelength,
                          gauss.gamma, gauss.beta, gauss.I0,
                          X_STEPS, Z_STEPS, X_RANGE, Z_RANGE, z=z_contacto)
    x, z = c['x_xz'], c['z_xz']
    dx, dz = (x[1] - x[0]) * 1e-9, (z[1] - z[0]) * 1e-9
    xy = fuerzas_xy(*[e * factor_E for e in c['XY']], dx, dx, dz,
                    alpha_p, alpha_pp, eps_m)
    xz = fuerzas_xz(*[e * factor_E for e in c['XZ']], dx, dz,
                    alpha_p, alpha_pp, eps_m)
    xz = [F * pN for F in xz]
    # la NP no puede tener el centro a menos de a del vidrio (aunque se
    # calcule "sin vidrio", se recorta igual para comparar la misma region)
    afuera = (z > z_contacto + 1e-6)[:, None]
    xz = [np.where(afuera, np.nan, F) for F in xz]
    return dict(x_xy=c['x_xy'], xy=[F * pN for F in xy], x_xz=x, z_xz=z, xz=xz)


def fuerza_en_contacto(m):
    """Fz (pN) y sus componentes en el contacto (x = y = 0, z = -a), del plano XZ."""
    i = np.argmin(np.abs(m['z_xz'] - z_contacto))
    j = np.argmin(np.abs(m['x_xz']))
    return {k: m['xz'][n][i, j] for k, n in (('total', 1), ('grad', 3), ('scat', 5), ('sc', 7))}


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
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
        if nombre == 'con vidrio':
            ax.axhspan(0, z[-1], color='lightsteelblue', zorder=3)
            ax.axhspan(z_contacto, 0, color='0.85', zorder=3)
            ax.text(0, 50, f'vidrio (n = {n_vidrio})', ha='center', va='center',
                    zorder=4, fontsize=8)
        else:
            ax.axhspan(z_contacto, z[-1], color='0.85', zorder=3)
        ax.axhline(z_contacto, color='tab:red', lw=0.8, ls='--', zorder=4)
        ax.set_xlim(-600, 600)
        ax.set_ylim(100, z[0])  # +z hacia abajo; se muestra hasta 100 nm adentro del vidrio
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
        ax.set_title(nombre)
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

    m = {nombre: mapas(n2) for nombre, n2 in CASOS.items()}
    graficar_xz(m, os.path.join(CARPETA, 'fuerzas_xz.png'))
    graficar_xy(m, os.path.join(CARPETA, 'fuerzas_xy.png'))
    contacto = {nombre: fuerza_en_contacto(mm) for nombre, mm in m.items()}

    lineas = [
        '# Printing con el haz gaussiano: fuerzas con y sin el vidrio', '',
        f'- Haz: NA = {gauss.NA}, n = {gauss.n} (agua), λ = {gauss.wavelength} nm, '
        f'polarización lineal en x, w0 = {gauss.w0:.3f} mm, I0 = {gauss.I0:.3f} mW/cm² '
        f'(calibrado: FWHM = 330 nm, P = 1 mW en la muestra).',
        f'- Partícula: Au, a = {a_nm:.0f} nm, en agua; α\' = {alpha_p:.3e} m³, '
        f'α\'\' = {alpha_pp:.3e} m³ (Mie, dipolo a1).',
        f'- Vidrio: n = {n_vidrio}, semi-infinito, con la superficie en el foco (z = 0). '
        f'Contacto: centro de la NP en z = −a = {z_contacto:.0f} nm.',
        '- Campos: PyFocus (sim.custom con la interfaz). Fuerza total = gradiente + '
        'scattering + spin-curl.', '',
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
        lineas.append(f"| {nombre} | {p['total']:.3f} | {F_IMPRESION_pN / p['total']:.2f} |")
    lineas += ['', 'Fuerzas a P = 1 mW en la muestra (escalan linealmente con P). '
               'Fz > 0 empuja la NP hacia el vidrio.', '']
    with open(os.path.join(CARPETA, 'resumen.md'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lineas))
    print('\n'.join(lineas))
