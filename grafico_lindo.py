# -*- coding: utf-8 -*-
"""
graficar_resumen.py

Carga campos guardados (.npz) de un haz gauss y un haz dona, recalcula
las fuerzas opticas con fuerzas.py y arma una figura resumen 3x3:
  fila 1 -> mapas XY (gauss / dona / resultante)
  fila 2 -> mapas XZ (idem)
  fila 3 -> corte de Fz en z=0 a lo largo de x, con FWHM marcado

Convencion de signo para MOSTRAR (streamplot y cortes, fila 2 y 3):
  el haz contrapropagante hace que Fz de gauss y Fz de dona tengan
  sentidos fisicos opuestos. Se muestra gauss con -Fz y dona con +Fz
  (dona sin negar), consistente con el main.py original del proyecto.
  Esto es solo para visualizacion -- el calculo de fuerza en si
  (fuerzas.py) no cambia de signo en ningun lado.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sim_config import eps_m, dx_fuerzas, dy_fuerzas, dz_fuerzas
from particula import calc_alpha
from fuerzas import fuerzas_xy, fuerzas_xz


def _cargar_haz(npz_path):
    data = np.load(npz_path)
    alpha_p, alpha_pp = calc_alpha(float(data['wavelength']))
    return data, alpha_p, alpha_pp


def _fuerza_xy_pN(data, alpha_p, alpha_pp):
    Fx, Fy, *_ = fuerzas_xy(data['Ex_xy'], data['Ey_xy'], data['Ez_xy'],
                             dx_fuerzas, dy_fuerzas, dz_fuerzas,
                             alpha_p, alpha_pp, eps_m)
    return Fx * 1e12, Fy * 1e12  # N -> pN


def _fuerza_xz_pN(data, alpha_p, alpha_pp):
    Fx, Fz, *_ = fuerzas_xz(data['Ex_xz'], data['Ey_xz'], data['Ez_xz'],
                             dx_fuerzas, dz_fuerzas,
                             alpha_p, alpha_pp, eps_m)
    return Fx * 1e12, Fz * 1e12


def calcular_fwhm(x, y, ventana=None):
    """
    FWHM de la caracteristica dominante de y(x) (positiva o negativa),
    con interpolacion lineal en los cruces de nivel medio.

    ventana : (x_min, x_max) opcional -- restringe la busqueda del
    extremo a ese rango, util para evitar que un pico angosto y
    espurio (ej. interferencia en el centro de una curva resultante
    con varios lobulos) le gane al lobulo ancho que en realidad
    interesa medir.

    Retorna (ancho, x_izq, x_der) o (None, None, None) si no hay
    cruce de nivel medio dentro del dominio.
    """
    if ventana is not None:
        mask = (x >= ventana[0]) & (x <= ventana[1])
        x_b, y_b = x[mask], y[mask]
    else:
        x_b, y_b = x, y

    if len(x_b) == 0:
        return None, None, None

    idx_local = np.argmax(np.abs(y_b))
    y_ext = y_b[idx_local]
    half = y_ext / 2
    idx_ext = np.where(x == x_b[idx_local])[0][0]  # indice en el array completo

    izq = None
    for i in range(idx_ext, 0, -1):
        if (y[i - 1] - half) * (y[i] - half) <= 0:
            izq = x[i - 1] + (half - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
            break
    der = None
    for i in range(idx_ext, len(x) - 1):
        if (y[i] - half) * (y[i + 1] - half) <= 0:
            der = x[i] + (half - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            break

    if izq is None or der is None:
        return None, None, None
    return der - izq, izq, der


def graficar_resumen(npz_gauss, npz_dona, rango_grafico, titulo_general='',
                      cmap='viridis', figsize=(11, 9), zoom=300,
                      ventana_fwhm_resultante=None):
    """
    rango_grafico : tiene que coincidir EXACTAMENTE con el x_range/z_range
    de sim_config.py usado al generar los .npz -- es la calibracion de
    los ejes, no un parametro de zoom (para eso esta `zoom`).

    ventana_fwhm_resultante : (x_min, x_max) opcional, para acotar donde
    se busca el lobulo dominante del corte resultante (ver calcular_fwhm).
    Ajustar mirando el propio grafico si el FWHM sale raro.
    """
    data_g, ap_g, app_g = _cargar_haz(npz_gauss)
    data_d, ap_d, app_d = _cargar_haz(npz_dona)

    # ---- XY ----
    Fx_g, Fy_g = _fuerza_xy_pN(data_g, ap_g, app_g)
    Fx_d, Fy_d = _fuerza_xy_pN(data_d, ap_d, app_d)
    Fx_tot, Fy_tot = Fx_g + Fx_d, Fy_g + Fy_d

    # ---- XZ ----
    Fx_g_xz, Fz_g_xz = _fuerza_xz_pN(data_g, ap_g, app_g)
    Fx_d_xz, Fz_d_xz = _fuerza_xz_pN(data_d, ap_d, app_d)
    Fz_tot_xz = Fz_g_xz - Fz_d_xz  # combinacion fisica (definicion unica, sin signo de display)

    # convencion de signo SOLO para mostrar (streamplot fila 2 y cortes fila 3)
    disp_Fz_g = -Fz_g_xz
    disp_Fz_d = Fz_d_xz
    disp_Fz_tot = -Fz_tot_xz

    # magnitudes para colormap (no dependen del signo de display)
    mag_xy = [np.hypot(Fx_g, Fy_g), np.hypot(Fx_d, Fy_d), np.hypot(Fx_tot, Fy_tot)]
    mag_xz = [np.hypot(Fx_g_xz, Fz_g_xz), np.hypot(Fx_d_xz, Fz_d_xz), np.hypot(Fx_tot_xz := Fx_g_xz + Fx_d_xz, Fz_tot_xz)]
    vmax_xy, vmax_xz = max(m.max() for m in mag_xy), max(m.max() for m in mag_xz)

    n = mag_xy[0].shape[0]
    x = np.linspace(-rango_grafico / 2, rango_grafico / 2, n)
    extent = [x[0], x[-1], x[0], x[-1]]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, width_ratios=[1, 1, 1, 0.06], height_ratios=[1, 1, 0.6],
                  hspace=0.45, wspace=0.35)

    titulos = ['Gaussiano', 'Dona', 'Resultante']
    campos_xy = [(Fx_g, Fy_g), (Fx_d, Fy_d), (Fx_tot, Fy_tot)]
    campos_xz_plot = [(Fx_g_xz, disp_Fz_g), (Fx_d_xz, disp_Fz_d), (Fx_tot_xz, disp_Fz_tot)]

    # --- fila 1: XY ---
    for i, ((Fx, Fy), tit) in enumerate(zip(campos_xy, titulos)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(mag_xy[i], extent=extent, origin='lower', cmap=cmap,
                        vmin=0, vmax=vmax_xy)
        ax.streamplot(x, x, Fx, Fy, density=1.6, color='black', linewidth=0.7, arrowsize=0.8)
        if zoom:
            ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
        ax.set_title(f'{tit} XY', fontsize=11)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)' if i == 0 else '')

    # --- fila 2: XZ (usa Fz SIN negar -- el invert_yaxis ya resuelve el sentido visual) ---
    campos_xz_plot = [(Fx_g_xz, Fz_g_xz), (Fx_d_xz, -Fz_d_xz), (Fx_tot_xz, Fz_tot_xz)]

    for i, ((Fx, Fz), tit) in enumerate(zip(campos_xz_plot, titulos)):
        ax = fig.add_subplot(gs[1, i])
        nz, nx = Fx.shape
        x_xz = np.linspace(-rango_grafico / 2, rango_grafico / 2, nx)
        z_xz = np.linspace(-rango_grafico / 2, rango_grafico / 2, nz)
        extent_xz = [x_xz[0], x_xz[-1], z_xz[0], z_xz[-1]]

        im2 = ax.imshow(mag_xz[i], extent=extent_xz, origin='lower', cmap=cmap,
                         vmin=0, vmax=vmax_xz)
        ax.streamplot(x_xz, z_xz, Fx, Fz, density=1.6, color='black', linewidth=0.7, arrowsize=0.8)
        ax.invert_yaxis()
        if zoom:
            ax.set_xlim(-zoom, zoom); ax.set_ylim(zoom, -zoom)
        ax.set_title(f'{tit} XZ', fontsize=11)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('z (nm)' if i == 0 else '')
        
    # --- fila 3: corte de Fz en z=0, con escala de fuerza compartida ---
    cortes = [disp_Fz_g, disp_Fz_d, disp_Fz_tot]
    calcular_fwhm_en = [True, False, True]
    ventanas_fwhm = [None, None, ventana_fwhm_resultante]
    
    # Obtener los perfiles en z=0
    perfiles = []
    for perfil_completo in cortes:
        nz, nx = perfil_completo.shape
        fila_central = nz // 2
        perfiles.append(perfil_completo[fila_central, :])
    
    # Misma escala vertical para los tres gráficos
    fuerza_max = max(np.max(np.abs(perfil)) for perfil in perfiles)
    
    # Margen
    fuerza_max *= 1.1
    
    for i, (perfil, tit) in enumerate(zip(perfiles, titulos)):
        ax = fig.add_subplot(gs[2, i])
    
        nx = len(perfil)
        x_xz = np.linspace(-rango_grafico / 2, rango_grafico / 2, nx)
    
        ax.plot(x_xz, perfil, color='tab:blue', linewidth=1.3)
        ax.axhline(0, color='gray', linewidth=0.6)
    
        if calcular_fwhm_en[i]:
            ancho, xi, xd = calcular_fwhm(
                x_xz, perfil, ventana=ventanas_fwhm[i]
            )
    
            if ancho is not None:
                i_izq = np.searchsorted(x_xz, xi)
                i_der = np.searchsorted(x_xz, xd)
    
                y_half = (
                    perfil[min(i_izq, nx - 1)] +
                    perfil[min(i_der, nx - 1)]
                ) / 2
    
                ax.hlines(
                    y_half, xi, xd,
                    color='tab:red',
                    linewidth=1.2,
                    linestyle='--'
                )
    
                ax.annotate(
                    f'{ancho:.0f} nm',
                    xy=((xi + xd) / 2, y_half),
                    xytext=(0, 8 if y_half >= 0 else -14),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    color='tab:red'
                )
    
        if zoom:
            ax.set_xlim(-zoom, zoom)
    
        # MISMA ESCALA DE FUERZA EN LOS TRES
        ax.set_ylim(-fuerza_max, fuerza_max)
    
        ax.set_title(f'{tit} (z=0)', fontsize=10)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('Fz (pN)' if i == 0 else '')

    ax.grid(alpha=0.3)
    # --- colorbars compartidas (una por fila de mapas) ---
    cax1 = fig.add_subplot(gs[0, 3])
    fig.colorbar(im, cax=cax1, label='|F| (pN)')
    cax2 = fig.add_subplot(gs[1, 3])
    fig.colorbar(im2, cax=cax2, label='|F| (pN)')

    if titulo_general:
        fig.suptitle(titulo_general, fontsize=14, y=0.98)

    plt.savefig(f'{titulo_general or "resumen"}.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{titulo_general or "resumen"}.svg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    graficar_resumen('campos_gauss_agua.npz', 'campos_dona_aceite.npz',
                      rango_grafico=1200,  # << debe coincidir con sim_config.x_range de la corrida
                      titulo_general='Trampa óptica contrapropagante',
                      zoom=400,
                      ventana_fwhm_resultante=(-60, 60))  # << ajustar mirando el corte del resultante
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    