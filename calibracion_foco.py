"""
calibracion_foco.py

Pasa de "lo que mido EN EL FOCO" a "los w0 e I0 que hay que ponerle a PyFocus
en el back focal plane (BFP)", y viceversa.

Idea
----------------------------
Barrer en 2D (w0, I0) no hace falta, porque PyFocus es lineal en el campo:
    - la FORMA del foco (ancho del spot, radio del anillo) depende solo de w0
      (y de NA, n, h, lambda), no de I0;
    - la intensidad del foco es proporcional a I0.
Entonces:
    1) se barre w0 y se compara el ancho del foco simulado con el medido
       -> w0* (1D, con brentq para afinar);
    2) con w0* fijo, I0 sale de un cociente (potencia o intensidad pico medidas).
Y si medis las tres cosas (ancho, potencia, intensidad pico) la tercera queda
como chequeo de consistencia entre modelo y experimento.

Que se mide (todo en el plano focal, z = 0)
-------------------------------------------
    w_foco_nm      ancho del foco. Segun `metrica`:
                     'fwhm'          FWHM del lobulo central (gauss)
                     'r_1e2'         radio donde I cae a I_pico/e^2 (el "w0 en el foco")
                     'r_anillo'      radio del maximo del anillo (dona)
                     'grosor_anillo' FWHM del grosor radial del anillo (dona)
    P_mW           potencia en la muestra (la que entra al foco)
    I_pico_kWcm2   intensidad pico en el foco (para la dona, el maximo del anillo)
    T_obj          transmision del objetivo (P_muestra = T_obj * P_apertura). Default 1.

Uso:  correr este archivo celda por celda (#%%), o importar
      calibrar_haz / predecir_foco desde otro script.
"""
import json
import os
import warnings
from datetime import datetime

import numpy as np
from scipy.optimize import brentq
from PyFocus import sim

from sim_config import RUTA_CALIBRACION
from pyfocus_wrapper import (campo_foco, detectar_doble_division,
                             intensidad_foco_mWcm2)

KWCM2_A_MWCM2 = 1e6          # 1 kW/cm^2 = 1e6 mW/cm^2
WM2_A_MWCM2 = 0.1            # 1 W/m^2 = 0.1 mW/cm^2

_FUNCIONES = {'gauss': sim.no_mask, 'dona': sim.VP}
_METRICA_DEFAULT = {'gauss': 'fwhm', 'dona': 'r_anillo'}


# ---------------------------------------------------------------------------
# Medicion de metricas sobre el mapa de intensidad simulado
# ---------------------------------------------------------------------------
def _cruce(perfil, nivel, i0, paso):
    """Indice fraccional donde perfil baja de `nivel`, yendo desde i0 en sentido `paso` (+1/-1)."""
    i = i0
    while 0 <= i + paso < len(perfil):
        if perfil[i + paso] < nivel:
            y0, y1 = perfil[i], perfil[i + paso]
            return i + paso * (y0 - nivel) / (y0 - y1)
        i += paso
    return np.nan


def _pico_parabolico(perfil, j):
    """Posicion fraccional del maximo cerca del indice j (ajuste parabolico de 3 puntos)."""
    if j <= 0 or j >= len(perfil) - 1:
        return float(j)
    a, b, c_ = perfil[j - 1], perfil[j], perfil[j + 1]
    den = a - 2 * b + c_
    return j if den == 0 else j + 0.5 * (a - c_) / den


def _metricas(I, dx_nm):
    """Todas las metricas de ancho que tienen sentido para el mapa I (2D)."""
    N = I.shape[0]
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    out = {'I_pico': float(I.max())}

    # --- lobulo central (gauss): cortes por el maximo, promedio x/y ---
    m = I[iy, ix]
    for clave, nivel, factor in (('fwhm', m / 2, 1.0), ('r_1e2', m / np.e ** 2, 0.5)):
        anchos = []
        for perfil, i in ((I[iy, :], ix), (I[:, ix], iy)):
            izq, der = _cruce(perfil, nivel, i, -1), _cruce(perfil, nivel, i, +1)
            anchos.append((der - izq) * dx_nm * factor)
        out[clave + '_nm'] = _media(anchos)

    # --- anillo (dona): centro del arreglo; radio y grosor del maximo radial ---
    ic = N // 2
    radios, grosores = [], []
    for perfil in (I[ic, :], I[:, ic]):
        for lado in (+1, -1):
            semi = perfil[ic:] if lado == 1 else perfil[ic::-1]
            j = int(np.argmax(semi))
            radios.append(_pico_parabolico(semi, j) * dx_nm)
            nivel = semi[j] / 2
            dentro, fuera = _cruce(semi, nivel, j, -1), _cruce(semi, nivel, j, +1)
            grosores.append((fuera - dentro) * dx_nm)
    out['r_anillo_nm'] = _media(radios)
    out['grosor_anillo_nm'] = _media(grosores)
    return out


def _media(v):
    """nanmean sin el warning cuando todo es NaN (metricas que no aplican al haz)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(v))


def _grid(x_steps=5, x_range=2000):
    return dict(x_steps=x_steps, z_steps=1, x_range=x_range, z_range=4)


def _haz_id(haz):
    nombre = getattr(haz, 'figure_name', None)
    if nombre not in _FUNCIONES:
        raise ValueError("calibracion_foco espera el modulo beams.gauss o beams.dona "
                         f"(figure_name='gauss'/'dona'), recibi: {nombre!r}")
    return nombre


# ---------------------------------------------------------------------------
# Direccion BFP -> foco
# ---------------------------------------------------------------------------
def area_efectiva_m2(w0_mm, h_mm):
    """
    P_apertura = I0 * area_efectiva. Es la integral de I0*exp(-2 rho^2/w0^2)
    sobre la apertura del objetivo (radio h): lo que realmente pasa por el objetivo.
    """
    w, h = w0_mm * 1e-3, h_mm * 1e-3
    return np.pi * w ** 2 / 2 * (1 - np.exp(-2 * h ** 2 / w ** 2))


def evaluar_foco(haz, w0_mm, **grid):
    """Un solo punto: simula el foco para un w0 en el BFP. I0 = 1 mW/cm^2 (todo es lineal en I0)."""
    nombre = _haz_id(haz)
    g = _grid(**grid)
    campos = campo_foco(_FUNCIONES[nombre], haz.NA, haz.n, haz.h, w0_mm, haz.wavelength,
                        haz.gamma, haz.beta, 1.0, g['x_steps'], g['z_steps'],
                        g['x_range'], g['z_range'], solo_xy=True)
    I = intensidad_foco_mWcm2(*campos['XY'])
    res = _metricas(I, g['x_steps'])
    res['w0_mm'] = float(w0_mm)
    res['ganancia'] = res['I_pico']            # I_pico_foco / I0_BFP  (adimensional)
    return res


def barrido_w0(haz, w0_mm_lista=None, n_puntos=9, **grid):
    """Barrido de w0 en el BFP. Por defecto de 0.4*h a 5*h (de sub-llenado a sobre-llenado)."""
    if w0_mm_lista is None:
        w0_mm_lista = np.geomspace(0.4 * haz.h, 5 * haz.h, n_puntos)
    return [evaluar_foco(haz, w, **grid) for w in w0_mm_lista]


def predecir_foco(haz, w0_mm=None, I0_mWcm2=None, T_obj=1.0, verbose=True, **grid):
    """
    "Que da en el foco" con los w0/I0 del BFP (por defecto los que tiene el modulo
    del haz). Es lo que hay que comparar con lo medido.
    """
    nombre = _haz_id(haz)
    w0_mm = haz.w0 if w0_mm is None else w0_mm
    I0 = haz.I0 if I0_mWcm2 is None else I0_mWcm2
    r = evaluar_foco(haz, w0_mm, **grid)
    A = area_efectiva_m2(w0_mm, haz.h)
    out = dict(haz=nombre, w0_mm=w0_mm, I0_mWcm2=I0,
               llenado_w0_sobre_h=w0_mm / haz.h,
               P_BFP_total_mW=np.pi * (w0_mm * 1e-3) ** 2 / 2 * I0 * 10 * 1e3,
               P_apertura_mW=I0 * 10 * A * 1e3,
               P_muestra_mW=T_obj * I0 * 10 * A * 1e3,
               I_pico_kWcm2=T_obj * I0 * r['ganancia'] / KWCM2_A_MWCM2,
               fwhm_nm=r['fwhm_nm'], r_1e2_nm=r['r_1e2_nm'],
               r_anillo_nm=r['r_anillo_nm'], grosor_anillo_nm=r['grosor_anillo_nm'])
    if verbose:
        print(f"\n--- {nombre}: w0={w0_mm:.3g} mm (w0/h = {out['llenado_w0_sobre_h']:.2f}), "
              f"I0={I0:.3g} mW/cm2 en el BFP ---")
        print(f"  potencia total del haz antes del objetivo : {out['P_BFP_total_mW']:.3f} mW")
        print(f"  potencia que pasa por la apertura         : {out['P_apertura_mW']:.3f} mW")
        print(f"  intensidad pico en el foco                : {out['I_pico_kWcm2']:.3g} kW/cm2")
        if nombre == 'gauss':
            print(f"  FWHM del foco                             : {out['fwhm_nm']:.0f} nm")
            print(f"  radio 1/e^2 del foco                      : {out['r_1e2_nm']:.0f} nm")
        else:
            print(f"  radio del anillo                          : {out['r_anillo_nm']:.0f} nm")
            print(f"  grosor del anillo (FWHM)                  : {out['grosor_anillo_nm']:.0f} nm")
    return out


# ---------------------------------------------------------------------------
# Direccion foco -> BFP  (la calibracion en si)
# ---------------------------------------------------------------------------
def calibrar_haz(haz, w_foco_nm, metrica=None, P_mW=None, I_pico_kWcm2=None,
                 T_obj=1.0, w0_rango_mm=None, n_puntos=9, tol_w0_mm=1e-3,
                 verbose=True, **grid):
    """
    Devuelve el dict con w0_mm e I0_mWcm2 (BFP) que reproducen lo medido en el foco.

    w_foco_nm     valor medido de la metrica elegida (nm)
    P_mW          potencia en la muestra (mW)            -> fija I0
    I_pico_kWcm2  intensidad pico en el foco (kW/cm^2)   -> fija I0 (si no hay P_mW),
                  o chequeo de consistencia (si estan las dos)
    """
    nombre = _haz_id(haz)
    metrica = metrica or _METRICA_DEFAULT[nombre]
    clave = metrica + '_nm'
    if P_mW is None and I_pico_kWcm2 is None:
        raise ValueError(
            "El ancho del foco solo fija w0: pasa tambien P_mW o I_pico_kWcm2 "
            "para poder fijar I0."
        )
    avisos = []
    detectar_doble_division(verbose=verbose)

    # 1) barrido grueso en w0
    if w0_rango_mm is None:
        w0_rango_mm = (0.4 * haz.h, 5 * haz.h)
    lista = np.geomspace(w0_rango_mm[0], w0_rango_mm[1], n_puntos)
    if verbose:
        print(f"[{nombre}] barrido de w0 entre {lista[0]:.2f} y {lista[-1]:.2f} mm "
              f"({n_puntos} puntos)...")
    barrido = [evaluar_foco(haz, w, **grid) for w in lista]
    valores = np.array([b[clave] for b in barrido])
    f = valores - w_foco_nm

    # 2) buscar el cruce y afinar con brentq (sobre simulaciones reales)
    cruces = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    if len(cruces) == 0:
        lo, hi = np.nanmin(valores), np.nanmax(valores)
        if w_foco_nm < lo:
            motivo = (f"El foco medido ({w_foco_nm:.0f} nm) es MAS CHICO que el minimo que "
                      f"da el modelo ({lo:.0f} nm, haz sobre-llenado). Revisar NA efectiva, "
                      f"la definicion de la metrica o la medicion.")
        else:
            motivo = (f"El foco medido ({w_foco_nm:.0f} nm) es mas ANCHO que el maximo del "
                      f"barrido ({hi:.0f} nm). Bajar el limite inferior de w0_rango_mm "
                      f"(o ampliar x_range).")
        raise ValueError(f"[{nombre}] {motivo}")
    if len(cruces) > 1:
        avisos.append("La metrica cruza el valor medido mas de una vez; se usa el primer cruce.")
    k = cruces[0]
    w0_sol = brentq(lambda w: evaluar_foco(haz, w, **grid)[clave] - w_foco_nm,
                    lista[k], lista[k + 1], xtol=tol_w0_mm)
    punto = evaluar_foco(haz, w0_sol, **grid)

    # 3) I0 desde potencia o intensidad pico
    A = area_efectiva_m2(w0_sol, haz.h)
    G = punto['ganancia']
    if P_mW is not None:
        I0 = (P_mW * 1e-3) / (T_obj * A) * WM2_A_MWCM2      # W/m2 -> mW/cm2
    else:
        I0 = I_pico_kWcm2 * KWCM2_A_MWCM2 / (T_obj * G)
    I_pico_pred_kWcm2 = T_obj * I0 * G / KWCM2_A_MWCM2

    res = dict(haz=nombre, w0_mm=float(w0_sol), I0_mWcm2=float(I0),
               llenado_w0_sobre_h=float(w0_sol / haz.h),
               P_BFP_total_mW=float(np.pi * (w0_sol * 1e-3) ** 2 / 2 * I0 * 10 * 1e3),
               P_apertura_mW=float(I0 * 10 * A * 1e3),
               objetivo=dict(metrica=metrica, w_foco_nm=w_foco_nm, P_mW=P_mW,
                             I_pico_kWcm2=I_pico_kWcm2, T_obj=T_obj),
               prediccion=dict(**{clave: float(punto[clave])},
                               I_pico_kWcm2=float(I_pico_pred_kWcm2)),
               barrido=dict(w0_mm=[float(w) for w in lista], **{clave: valores.tolist()},
                            ganancia=[b['ganancia'] for b in barrido]),
               avisos=avisos)

    # chequeo de consistencia si se midieron P e I_pico a la vez
    if P_mW is not None and I_pico_kWcm2 is not None:
        desv = (I_pico_pred_kWcm2 - I_pico_kWcm2) / I_pico_kWcm2 * 100
        res['consistencia_I_pico_pct'] = float(desv)
        if abs(desv) > 15:
            avisos.append(f"Con el ancho y la potencia medidos, el modelo predice "
                          f"{I_pico_pred_kWcm2:.3g} kW/cm2 de pico y mediste "
                          f"{I_pico_kWcm2:.3g} ({desv:+.0f} %): ancho y potencia no son "
                          f"consistentes con la intensidad pico (perfil no ideal, "
                          f"aberraciones, T_obj, medicion).")
    if res['llenado_w0_sobre_h'] > 3:
        avisos.append("w0/h > 3: el haz esta casi uniforme en la apertura; el ancho del foco "
                      "casi no depende de w0 y la solucion es poco sensible.")

    if verbose:
        _imprimir(res, haz, w_foco_nm, metrica)
    return res


def _imprimir(res, haz, w_foco_nm, metrica):
    print(f"\n=== Calibracion {res['haz']} ===")
    print(f"  medido en el foco : {metrica} = {w_foco_nm:.0f} nm"
          + (f", P = {res['objetivo']['P_mW']:.3g} mW" if res['objetivo']['P_mW'] else "")
          + (f", I_pico = {res['objetivo']['I_pico_kWcm2']:.3g} kW/cm2"
             if res['objetivo']['I_pico_kWcm2'] else ""))
    print(f"  --> w0 (BFP)      = {res['w0_mm']:.3f} mm   (w0/h = {res['llenado_w0_sobre_h']:.2f})")
    print(f"  --> I0 (BFP)      = {res['I0_mWcm2']:.4g} mW/cm2")
    print(f"      potencia total antes del objetivo = {res['P_BFP_total_mW']:.3f} mW, "
          f"por la apertura = {res['P_apertura_mW']:.3f} mW")
    print(f"      I_pico predicha en el foco = {res['prediccion']['I_pico_kWcm2']:.3g} kW/cm2")
    if 'consistencia_I_pico_pct' in res:
        print(f"      desvio vs I_pico medida = {res['consistencia_I_pico_pct']:+.1f} %")
    for a in res['avisos']:
        print("  ! " + a)


# ---------------------------------------------------------------------------
# Pasar el resultado a la config
# ---------------------------------------------------------------------------
def guardar_calibracion(res, ruta=None):
    """
    Escribe w0/I0 del haz en calibracion.json. beams/gauss.py y beams/dona.py lo leen
    al importarse (sim_config.cargar_calibracion) y pisan sus w0/I0 por defecto.
    """
    ruta = ruta or RUTA_CALIBRACION
    datos = {}
    if os.path.exists(ruta):
        with open(ruta, encoding='utf-8') as fh:
            datos = json.load(fh)
    datos[res['haz']] = dict(
        w0_mm=res['w0_mm'], I0_mWcm2=res['I0_mWcm2'],
        objetivo=res['objetivo'], prediccion=res['prediccion'],
        fecha=datetime.now().strftime('%Y-%m-%d %H:%M'))
    with open(ruta, 'w', encoding='utf-8') as fh:
        json.dump(datos, fh, indent=2, ensure_ascii=False)
    print(f"[calibracion] {res['haz']}: guardado en {ruta}")


def borrar_calibracion(nombre=None, ruta=None):
    """Quita un haz (o todos) del json para volver a los w0/I0 por defecto de cada archivo de haz."""
    ruta = ruta or RUTA_CALIBRACION
    if not os.path.exists(ruta):
        return
    if nombre is None:
        os.remove(ruta)
        return
    with open(ruta, encoding='utf-8') as fh:
        datos = json.load(fh)
    datos.pop(nombre, None)
    with open(ruta, 'w', encoding='utf-8') as fh:
        json.dump(datos, fh, indent=2, ensure_ascii=False)


def graficar_barrido(res):
    """Ancho del foco y ganancia de intensidad vs w0 del BFP, con la solucion marcada."""
    import matplotlib.pyplot as plt
    clave = res['objetivo']['metrica'] + '_nm'
    b = res['barrido']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(b['w0_mm'], b[clave], 'o-')
    ax1.axhline(res['objetivo']['w_foco_nm'], color='tab:red', ls='--', label='medido')
    ax1.axvline(res['w0_mm'], color='gray', ls=':', label=f"w0* = {res['w0_mm']:.2f} mm")
    ax1.set_xscale('log'); ax1.set_xlabel('w0 en el BFP (mm)')
    ax1.set_ylabel(f"{res['objetivo']['metrica']} en el foco (nm)")
    ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(b['w0_mm'], np.array(b['ganancia']) / 1e6, 'o-', color='tab:green')
    ax2.axvline(res['w0_mm'], color='gray', ls=':')
    ax2.set_xscale('log'); ax2.set_xlabel('w0 en el BFP (mm)')
    ax2.set_ylabel('I_pico(foco) / I0(BFP)  (x1e6)')
    ax2.grid(alpha=.3)
    fig.suptitle(f"Calibracion {res['haz']}")
    fig.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from beams import gauss, dona

    #%% 1) Que foco dan los w0/I0 que tenes hoy en el BFP
    predecir_foco(gauss)
    predecir_foco(dona)

    #%% 2) Carga lo que MEDIS en el foco (dejar None lo que no midas) y calibra.
    #    ancho en nm; P_mW en la muestra; I_pico en kW/cm^2.
    # -----------------------------------------------------------------------
    # DATOS EXPERIMENTALES
    #
    # Del informe de Laboratorio 6/7:
    #   - Para el haz de 532 nm se midio un ancho focal de
    #         FWHM = (126.4 +/- 3.4) nm
    #   - En otra medicion se reportan irradiancias de
    #         7.46 +/- 0.25 mW/um^2
    #         8.80 +/- 0.32 mW/um^2
    #
    # IMPORTANTE:
    # El FWHM es una medida del foco y sirve para calibrar w0.
    #
    # Conversion:
    #   1 mW/um^2 = 100 kW/cm^2
    # -----------------------------------------------------------------------
    MEDICIONES = {
        'gauss': dict(
            w_foco_nm=330,
            metrica='fwhm',
            P_mW=1,
            I_pico_kWcm2=None,   #
            T_obj=1.0
        ),

        'dona': dict(
            w_foco_nm=None,
            metrica='r_anillo',
            P_mW=None,
            I_pico_kWcm2=None,
            T_obj=1.0
        ),
    }
    haces = {'gauss': gauss, 'dona': dona}
    for nombre, m in MEDICIONES.items():
        if m['w_foco_nm'] is None:
            continue
        res = calibrar_haz(haces[nombre], **m)
        graficar_barrido(res)
        guardar_calibracion(res)     # <- esto es "pasarselo a la config"
