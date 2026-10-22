# -*- coding: utf-8 -*-
"""
fuerzas_gauss_interfaz.py

Fuerzas del haz gaussiano (calibrado, ver calibracion.json) sobre la NP de
oro, sin y con la interfaz agua/vidrio del portaobjetos, mirando el foco.
La superficie del vidrio esta en el foco (z = 0).

Geometria: objetivo de inmersion en agua -> agua -> portaobjetos de vidrio
(con aceite de indice igualado abajo: se trata como semi-infinito). El haz
se propaga hacia +z y el vidrio ocupa z > 0. En los graficos +z va hacia
ABAJO (el haz baja desde el objetivo hacia el vidrio).

Campos: PyFocus (sim.custom con la interfaz en el foco), via
pyfocus_wrapper.campo_foco_vidrio. "Sin vidrio" se calcula igual, con
n2 = n1, para comparar con el mismo metodo y la misma grilla.

Las fuerzas solo tienen sentido en el agua (z < 0). Con vidrio, el plano
focal coincide con la superficie, asi que se toma del lado del agua
(z = 0-), y las derivadas en z se hacen solo con puntos del agua.

Fuerzas: gradiente + scattering + spin-curl (fuerzas.py). Las derivadas
fuera del plano se aproximan como en fuerzas.py (con div E = 0): en el
plano XZ es exacto para este haz (polarizado en x); en el plano XY el
spin-curl lateral queda aproximado.

Salida en resultados/gauss_interfaz/:
    fuerzas_xz.png    mapas XZ sin y con vidrio
    fuerzas_xy.png    plano focal sin/con vidrio y diferencia
    resumen.md        fuerza en el foco (por componente) y potencia
                      necesaria para llegar a F_IMPRESION_pN

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
# ~7.1 nm) y 121 en z (paso ~10.1 nm), los dos impares para que x = 0 y
# el foco caigan justo en la grilla.
X_RANGE, X_STEPS = 1210, 5
Z_RANGE, Z_STEPS = 1210, 5
# Altura del plano focal: un pelo arriba de la superficie, para quedar del
# lado del agua (PyFocus toma z = 0 exacto como vidrio).
Z_FOCO = -1e-3
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'resultados', 'gauss_interfaz')

pN = 1e12
a_nm = radio_particula * 1e9
factor_E = calcular_factor_E_foco(gauss.n)
alpha_p, alpha_pp = calc_alpha(gauss.wavelength)
CASOS = {'sin vidrio': gauss.n, 'con vidrio': n_vidrio}  # n2 de cada caso


def _campo(n2, x_steps, z_steps, x_range, z_range, z):
    return campo_foco_vidrio(gauss.NA, gauss.n, n2, gauss.h, gauss.w0, gauss.wavelength,
                             gauss.gamma, gauss.beta, gauss.I0,
                             x_steps, z_steps, x_range, z_range, z=z)


def _fuerzas_xz_agua(E, x, z, n2):
    """
    Fuerzas XZ (pN) calculando las derivadas solo con puntos del agua
    (z < 0) cuando hay vidrio; en el vidrio quedan NaN.
    """
    dx, dz = (x[1] - x[0]) * 1e-9, (z[1] - z[0]) * 1e-9
    agua = z < 0 if n2 != gauss.n else np.ones_like(z, bool)
    parcial = fuerzas_xz(*[e[agua] * factor_E for e in E], dx, dz,
                         alpha_p, alpha_pp, eps_m)
    salida = []
    for F in parcial:
        completo = np.full((len(z), len(x)), np.nan)
        completo[agua] = F * pN
        salida.append(completo)
    return salida


# ---------------------------------------------------------------------------
# Calculo
# ---------------------------------------------------------------------------
def mapas(n2):
    """Fuerzas (pN) en el plano focal XY (z = 0-) y en el plano XZ centrado en el foco."""
    c = _campo(n2, X_STEPS, Z_STEPS, X_RANGE, Z_RANGE, Z_FOCO)
    x, z = c['x_xz'], c['z_xz']
    dx = (x[1] - x[0]) * 1e-9
    xy = fuerzas_xy(*[e * factor_E for e in c['XY']], dx, dx, dx,
                    alpha_p, alpha_pp, eps_m)
    xz = _fuerzas_xz_agua(c['XZ'], x, z, n2)
    return dict(x_xy=c['x_xy'], xy=[F * pN for F in xy], x_xz=x, z_xz=z, xz=xz)


def fuerza_en_el_foco(n2):
    """
    Fz (pN) y sus componentes en el foco (x = y = 0, z = 0-), con una grilla
    chica y fina (2 nm) que va de z = -20 nm hasta el foco: asi la derivada
    en z usa solo puntos del agua.
    """
    c = _campo(n2, 1.3, 0.95, 20, 20, -10 + Z_FOCO)  # 11 x 11 puntos, paso 2 nm
    x, z = c['x_xz'], c['z_xz']
    F = _fuerzas_xz_agua(c['XZ'], x, z, n2)
    i, j = np.argmin(np.abs(z - Z_FOCO)), np.argmin(np.abs(x))
    return {k: F[n][i, j] for k, n in (('total', 1), ('grad', 3), ('scat', 5), ('sc', 7))}


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def graficar_xz(m, archivo):
    vmax = max(np.nanmax(np.hypot(mm['xz'][0], mm['xz'][1])) for mm in m.values())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5), sharey=True)
    for ax, (nombre, mm) in zip(axes, m.items()):
        x, z = mm['x_xz'], mm['z_xz']
        Fx, Fz = mm['xz'][0], mm['xz'][1]
        im = ax.imshow(np.hypot(Fx, Fz), extent=[x[0], x[-1], z[0], z[-1]], origin='lower',
                       cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax.streamplot(x, z, np.ma.masked_invalid(Fx), np.ma.masked_invalid(Fz),
                      density=1.4, color='white', linewidth=0.6, arrowsize=0.7)
        if nombre == 'con vidrio':
            ax.axhspan(0, z[-1], color='lightsteelblue', zorder=3)
            ax.text(0, z[-1] / 2, f'vidrio (n = {n_vidrio})', ha='center', va='center',
                    zorder=4, fontsize=9)
        ax.axhline(0, color='w', lw=0.8, ls=':', zorder=4)
        ax.set_xlim(-600, 600)
        ax.set_ylim(z[-1], z[0])  # +z hacia abajo
        ax.set_title(nombre)
        ax.set_xlabel('x (nm)')
    axes[0].set_ylabel('z (nm)   (haz hacia abajo, foco en 0)')
    fig.colorbar(im, ax=axes, label='|F| (pN)', shrink=0.85)
    fig.suptitle(f'Haz gaussiano, plano XZ — NP Au de {2 * a_nm:.0f} nm, '
                 f'λ = {gauss.wavelength} nm, P = 1 mW, vidrio en el foco', fontsize=11)
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
    fig.suptitle('Fuerza lateral en el plano focal (z = 0, del lado del agua)')
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
    foco = {nombre: fuerza_en_el_foco(n2) for nombre, n2 in CASOS.items()}

    lineas = [
        '# Haz gaussiano: fuerzas en el foco, con y sin el vidrio', '',
        f'- Haz: NA = {gauss.NA}, n = {gauss.n} (agua), λ = {gauss.wavelength} nm, '
        f'polarización lineal en x, w0 = {gauss.w0:.3f} mm, I0 = {gauss.I0:.3f} mW/cm² '
        f'(calibrado: FWHM = 330 nm, P = 1 mW en la muestra).',
        f'- Partícula: Au, {2 * a_nm:.0f} nm de diámetro, en agua; α\' = {alpha_p:.3e} m³, '
        f'α\'\' = {alpha_pp:.3e} m³ (Mie, dipolo a1).',
        f'- Vidrio: n = {n_vidrio}, semi-infinito, con la superficie en el foco (z = 0).',
        '- Campos: PyFocus (sim.custom con la interfaz). Fuerza total = gradiente + '
        'scattering + spin-curl.', '',
        '## En el foco (x = y = 0, z = 0; con vidrio, del lado del agua)', '',
        '| caso | Fz total (pN) | gradiente | scattering | spin-curl |',
        '|---|---|---|---|---|',
    ]
    for nombre, p in foco.items():
        lineas.append(f"| {nombre} | {p['total']:.3f} | {p['grad']:.3f} | "
                      f"{p['scat']:.3f} | {p['sc']:.3f} |")
    lineas += ['', f'## Potencia para llegar a {F_IMPRESION_pN} pN en el foco', '',
               '| caso | Fz en el foco a 1 mW (pN) | potencia necesaria (mW) |',
               '|---|---|---|']
    for nombre, p in foco.items():
        lineas.append(f"| {nombre} | {p['total']:.3f} | {F_IMPRESION_pN / p['total']:.2f} |")
    lineas += ['', 'Fuerzas a P = 1 mW en la muestra (escalan linealmente con P). '
               'Fz > 0 empuja la NP hacia el vidrio.', '']
    with open(os.path.join(CARPETA, 'resumen.md'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lineas))
    print('\n'.join(lineas))
