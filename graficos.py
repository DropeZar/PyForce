# -*- coding: utf-8 -*-
"""
graficos.py

Graficos de los mapas de fuerza optica (magnitud + streamplot).
Se dejo solo lo esencial para visualizar las fuerzas -- los analisis
de ancho de campana (FWHM), conversion irradiancia->potencia y ajustes
de curvas que estaban en el main original se sacaron (eran
exploratorios / tenian formulas duplicadas e inconsistentes entre si;
si los necesitas, estan en el main_fuerza.py viejo como referencia).
"""
import numpy as np
import matplotlib.pyplot as plt


def _extent(shape, rango):
    x = np.linspace(-rango / 2, rango / 2, shape[1])
    y = np.linspace(-rango / 2, rango / 2, shape[0])
    return x, y, [x[0], x[-1], y[0], y[-1]]


def graficar_fuerza_total(Fx, Fy, rango, titulo, eje_x='x (nm)', eje_y='y (nm)'):
    """Mapa de magnitud + streamplot para una fuerza total (XY o XZ)."""
    x, y, extent = _extent(Fx.shape, rango)
    magnitud = np.sqrt(Fx ** 2 + Fy ** 2)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(magnitud, extent=extent, origin='lower', cmap='inferno')
    plt.streamplot(x, y, Fx, Fy, density=2, color='black', linewidth=1)
    plt.title(titulo)
    plt.xlabel(eje_x)
    plt.ylabel(eje_y)
    plt.colorbar(im, label='|F| (N)')
    plt.tight_layout()
    plt.show()


def graficar_comparacion(fuerzas_por_haz, rango, titulo_general='',
                          eje_x='x (nm)', eje_y='y (nm)'):
    """
    Comparacion lado a lado (gauss / dona / resultante) de mapas de
    fuerza. fuerzas_por_haz: lista de tuplas (Fx, Fy, titulo).
    """
    fig, axes = plt.subplots(1, len(fuerzas_por_haz), figsize=(5 * len(fuerzas_por_haz), 5))
    if len(fuerzas_por_haz) == 1:
        axes = [axes]

    x, y, extent = _extent(fuerzas_por_haz[0][0].shape, rango)

    for ax, (Fx, Fy, titulo) in zip(axes, fuerzas_por_haz):
        magnitud = np.sqrt(Fx ** 2 + Fy ** 2)
        im = ax.imshow(magnitud, extent=extent, origin='lower', cmap='inferno')
        ax.streamplot(x, y, Fx, Fy, density=2, color='black', linewidth=1)
        ax.set_title(titulo)
        ax.set_xlabel(eje_x)
        ax.set_ylabel(eje_y)
        plt.colorbar(im, ax=ax, label='|F| (N)')

    if titulo_general:
        fig.suptitle(titulo_general, fontsize=14)
    plt.tight_layout()
    plt.show()


def graficar_componentes(componentes, rango, titulo_general='',
                          eje_x='x (nm)', eje_y='z (nm)'):
    """
    Desglose de una fuerza total en sus componentes (gradiente,
    scattering, spin-curl). componentes: lista de tuplas
    (Fx_componente, Fz_componente, titulo).
    """
    fig, axes = plt.subplots(1, len(componentes), figsize=(5 * len(componentes), 5))
    if len(componentes) == 1:
        axes = [axes]

    x, y, extent = _extent(componentes[0][0].shape, rango)

    for ax, (Fx_c, Fz_c, titulo) in zip(axes, componentes):
        magnitud = np.sqrt(Fx_c ** 2 + Fz_c ** 2)
        sp = ax.streamplot(x, y, Fx_c, -Fz_c, density=2, color=magnitud,
                            cmap='inferno', linewidth=1.5)
        plt.colorbar(sp.lines, ax=ax, label='|F| (N)')
        ax.set_title(titulo)
        ax.set_xlabel(eje_x)
        ax.set_ylabel(eje_y)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    if titulo_general:
        fig.suptitle(titulo_general, fontsize=14)
    plt.tight_layout()
    plt.show()
