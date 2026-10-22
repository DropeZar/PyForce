# -*- coding: utf-8 -*-
"""
intensidad_gauss_interfaz.py

Intensidad del haz gaussiano calibrado (1 mW en la muestra) sin y con la
interfaz agua/vidrio del portaobjetos, con la superficie del vidrio en el
foco (z = 0). Campos de PyFocus (sim.custom via
pyfocus_wrapper.campo_foco_vidrio); "sin vidrio" se calcula igual con
n2 = n1, para comparar con el mismo metodo y la misma grilla.

Intensidad: I = 1/2 n c eps0 |E|^2 con el n de cada medio. Con la
convencion de PyFocus (ver pyfocus_wrapper.factor_E_foco) eso es
|E_PyFocus|^2 en el agua y (n_vidrio/n_agua) |E_PyFocus|^2 en el vidrio.
Se grafica en kW/cm^2. El plano XY es el plano focal, z = 0: con vidrio
cae justo en la superficie y se toma del lado del vidrio.

Salida: resultados/gauss_interfaz/intensidad.png
Uso:    python intensidad_gauss_interfaz.py     (~15 min: sim.custom es lento)

Las intensidades calculadas se guardan en intensidad_gauss_interfaz.npz; si
ese archivo existe se reusa (para retocar el grafico sin recalcular).
Borrarlo para volver a calcular.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from beams import gauss
from pyfocus_wrapper import campo_foco_vidrio, intensidad_foco_mWcm2
from sim_config import n_vidrio

# Grilla de sim.custom: 171 puntos en x (paso ~7.1 nm) y 121 en z (paso
# ~10.1 nm), impares para que x = 0 y z = 0 caigan en la grilla.
X_RANGE, X_STEPS = 1210, 5
Z_RANGE, Z_STEPS = 1210, 5
CASOS = {'sin vidrio': gauss.n, 'con vidrio': n_vidrio}
MWCM2_A_KWCM2 = 1e-6


def intensidades(n2):
    """Intensidad (kW/cm^2) en el plano XZ y en el plano focal XY (z = 0)."""
    c = campo_foco_vidrio(gauss.NA, gauss.n, n2, gauss.h, gauss.w0, gauss.wavelength,
                          gauss.gamma, gauss.beta, gauss.I0,
                          X_STEPS, Z_STEPS, X_RANGE, Z_RANGE, z=0)
    factor_vidrio = n2 / gauss.n  # 1 si no hay vidrio
    I_xz = intensidad_foco_mWcm2(*c['XZ']) * MWCM2_A_KWCM2
    I_xz = np.where((c['z_xz'] >= 0)[:, None], I_xz * factor_vidrio, I_xz)
    I_xy = intensidad_foco_mWcm2(*c['XY']) * MWCM2_A_KWCM2 * factor_vidrio
    return dict(x=c['x_xz'], z=c['z_xz'], I_xz=I_xz, I_xy=I_xy)


def graficar(datos, archivo):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], hspace=0.35, wspace=0.5)
    vmax_xz = max(d['I_xz'].max() for d in datos.values())
    vmax_xy = max(d['I_xy'].max() for d in datos.values())
    estilos = {'sin vidrio': dict(color='k', ls='--'), 'con vidrio': dict(color='tab:blue')}

    # --- fila 1: plano XZ (+z hacia abajo) y perfil sobre el eje ---
    ax_eje = fig.add_subplot(gs[0, 2])
    for col, (nombre, d) in enumerate(datos.items()):
        ax = fig.add_subplot(gs[0, col])
        x, z = d['x'], d['z']
        im = ax.imshow(d['I_xz'], extent=[x[0], x[-1], z[0], z[-1]], origin='lower',
                       cmap='inferno', vmin=0, vmax=vmax_xz, aspect='equal')
        if nombre == 'con vidrio':
            ax.axhline(0, color='w', lw=1)
            ax.text(-560, 60, f'vidrio (n = {n_vidrio})', color='w', fontsize=8, va='top')
            ax.text(-560, -60, f'agua (n = {gauss.n})', color='w', fontsize=8, va='bottom')
        ax.set_ylim(z[-1], z[0])  # +z hacia abajo (el haz baja hacia el vidrio)
        ax.set_xlim(-600, 600)
        ax.set_title(f'{nombre}: plano XZ')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('z (nm)   (haz hacia abajo)' if col == 0 else '')
        fig.colorbar(im, ax=ax, label='I (kW/cm²)', shrink=0.85)
        ax_eje.plot(z, d['I_xz'][:, np.argmin(np.abs(x))], label=nombre, **estilos[nombre])
    ax_eje.axvline(0, color='gray', lw=0.8)
    ax_eje.text(25, 0.02, 'vidrio →', fontsize=8, transform=ax_eje.get_xaxis_transform())
    ax_eje.text(-25, 0.02, '← agua', fontsize=8, ha='right',
                transform=ax_eje.get_xaxis_transform())
    ax_eje.set_xlabel('z (nm)   (foco y superficie del vidrio en 0)')
    ax_eje.set_ylabel('I (kW/cm²)')
    ax_eje.set_title('Intensidad sobre el eje (x = y = 0)')

    # --- fila 2: plano focal XY (z = 0) y corte en x ---
    ax_corte = fig.add_subplot(gs[1, 2])
    for col, (nombre, d) in enumerate(datos.items()):
        ax = fig.add_subplot(gs[1, col])
        x = d['x']
        im = ax.imshow(d['I_xy'], extent=[x[0], x[-1], x[0], x[-1]], origin='lower',
                       cmap='inferno', vmin=0, vmax=vmax_xy)
        ax.set_xlim(-500, 500); ax.set_ylim(-500, 500)
        ax.set_title(f'{nombre}: plano focal (z = 0)', fontsize=11)
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)' if col == 0 else '')
        fig.colorbar(im, ax=ax, label='I (kW/cm²)', shrink=0.85)
        i0 = np.argmin(np.abs(x))
        ax_corte.plot(x, d['I_xy'][i0, :], label=nombre, **estilos[nombre])
    ax_corte.set_xlim(-600, 600)
    ax_corte.set_xlabel('x (nm)')
    ax_corte.set_ylabel('I (kW/cm²)')
    ax_corte.set_title('Corte en x en el plano focal (y = 0, z = 0)')

    for ax in (ax_eje, ax_corte):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'Intensidad del haz gaussiano (λ = {gauss.wavelength} nm, NA = {gauss.NA}, '
                 f'1 mW en la muestra), foco sobre la superficie del vidrio', fontsize=13)
    fig.savefig(archivo, dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    archivo = (sys.argv[1] if len(sys.argv) > 1
               else os.path.join(base, 'resultados', 'gauss_interfaz', 'intensidad.png'))
    cache = os.path.join(base, 'intensidad_gauss_interfaz.npz')
    if os.path.exists(cache):
        npz = np.load(cache)
        datos = {nombre: {k: npz[f'{i}_{k}'] for k in ('x', 'z', 'I_xz', 'I_xy')}
                 for i, nombre in enumerate(CASOS)}
    else:
        datos = {nombre: intensidades(n2) for nombre, n2 in CASOS.items()}
        np.savez(cache, **{f'{i}_{k}': v for i, d in enumerate(datos.values())
                           for k, v in d.items()})
    graficar(datos, archivo)
    for nombre, d in datos.items():
        j = np.argmin(np.abs(d['x']))
        print(f"{nombre}: I max (XZ) = {d['I_xz'].max():.0f} kW/cm2, "
              f"I en el foco = {d['I_xz'][np.argmin(np.abs(d['z'])), j]:.0f} kW/cm2, "
              f"I max plano focal = {d['I_xy'].max():.0f} kW/cm2")
    print('figura:', archivo)
