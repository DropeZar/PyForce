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

from scipy.constants import epsilon_0 as eps0, c as c_luz

def graficar_intensidad(Ex, Ey, Ez, rango, n_medio, titulo,
                         eje_x='x (nm)', eje_y='y (nm)', corte='x'):
    """
    Mapa de intensidad (kW/cm^2) + perfil 1D a lo largo de un eje.
    Ex, Ey, Ez: campos en SI (ya multiplicados por factor_E).
    corte: 'x' o 'y', eje a lo largo del cual se toma el perfil (fila/columna central).
    """
    x, y, extent = _extent(Ex.shape, rango)

    # I = (1/2) n eps0 c |E|^2  [W/m^2] -> kW/cm^2 = *1e-7
    I = 0.5 * n_medio * eps0 * c_luz * (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    I *= 1e-7

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    im = ax1.imshow(I, extent=extent, origin='lower', cmap='viridis')
    ax1.set_title(f'Intensity on the {eje_x[0]}{eje_y[0]} plane')
    ax1.set_xlabel(eje_x)
    ax1.set_ylabel(eje_y)
    plt.colorbar(im, ax=ax1, label='Intensity (kW/cm²)')

    if corte == 'x':
        fila = I.shape[0] // 2
        perfil, eje = I[fila, :], x
        ax2.set_xlabel(eje_x)
    else:
        col = I.shape[1] // 2
        perfil, eje = I[:, col], y
        ax2.set_xlabel(eje_y)

    ax2.plot(eje, perfil)
    ax2.set_title(f'Intensity along {corte}')
    ax2.set_ylabel('Intensity (kW/cm²)')
    ax2.grid(True)

    if titulo:
        fig.suptitle(titulo, fontsize=13)
    plt.tight_layout()
    plt.show()