# Simulación de trampa óptica (haces contrapropagantes gauss + dona)

## Cómo está organizado

- **`sim_config.py`** — Único lugar donde se definen los parámetros compartidos
  (radio de la partícula, índices de refracción de los medios, `I0`,
  resolución/rango de PyFocus, paso espacial para las derivadas). Si hay
  que cambiar alguno de estos valores, se cambia acá — no en los otros
  archivos.
- **`particula.py`** — Polarizabilidad de Kuwata para la nanoesfera de
  oro, con la tabla completa de Johnson & Christy (1972). `calc_alpha(wavelength_nm)`
  usa por defecto el radio y el medio de `sim_config.py`.
- **`beams/gauss.py`**, **`beams/dona.py`** — Definición de cada haz en
  PyFocus (`NA`, `h`, `w0`, `wavelength`, polarización) y su `get_field()`.
  Los parámetros que comparten entre sí (I0, resolución, rango) vienen de
  `sim_config.py`.
- **`fuerzas.py`** — Funciones puras: `calcular_factor_E` (conversión de
  campos simulados a unidades SI) y `fuerzas_xy` / `fuerzas_xz` (fuerza de
  gradiente + scattering + spin-curl). Sin gráficos ni prints.
- **`graficos.py`** — Funciones de graficado reutilizables: mapa de
  magnitud + streamplot (`graficar_fuerza_total`), comparación lado a lado
  gauss/dona/resultante (`graficar_comparacion`), y desglose en
  componentes (`graficar_componentes`).
- **`pyfocus_wrapper.py`** — Capa entre el código y PyFocus: corre PyFocus sin
  abrir figuras, detecta/corrige la doble división de λ por n (ver abajo) y
  da el factor de conversión a SI (`factor_E_foco`).
- **`calibracion_foco.py`** — De lo medido en el foco a `w0`/`I0` del BFP (ver abajo).
- **`main.py`** — Orquesta todo: campos → polarizabilidad → conversión de
  unidades → fuerzas → gráficos esenciales.


## Cosas pendientes / a confirmar (ver también comentarios en el código)

1. Revisar bien el cálculo de polarizabilidad porque el la posición de 
los máximos cambia muy poco con el tamaño de la NP. Puede ser tranquilamente
el límite del modelo. El cálculo depende de un factor de tamaño x = np.pi * a / lam
entonces poner radio 60,40,20 no cambia tanto porque va dividido por la long de onda.

2. Valores reales para la simulación: Como las simulaciones necesitan
que le digas los datos antes de pasar por el objetivo, no sé cómo
sacar la irradiancia I0 y el w0 del haz antes. 
**(Resuelto con `calibracion_foco.py`, ver abajo.)**

3. Calibración de campo a unidades SI (`calcular_factor_E`)

Se revisó el código fuente real de PyFocus (`_calculate_amplitude_factors`
en `model/focus_field_calculators/base.py`) y se confirmó que PyFocus usa
`I0` literalmente como `|E0|²` — sin el factor físico `n·c·ε₀/2` que
relaciona intensidad real con campo eléctrico real. Por eso el factor de
conversión a V/m es una **constante** (no depende del campo simulado ni
del punto que se evalúe): `factor_E = sqrt(20 / (n_entrada · c · ε₀))`,
con `n_entrada = 1` (aire) porque el `I0` de este montaje se define antes
del objetivo. **(Corregido en esta versión, ver "Correcciones" al final:
el factor correcto usa el n del medio del foco.)** El detalle completo está en el docstring de
`calcular_factor_E` en `fuerzas.py`.


## Calibración foco → BFP (`calibracion_foco.py`)

`w0` e `I0` de `beams/gauss.py` y `beams/dona.py` son los del back focal plane
(antes del objetivo). Como lo que se mide es lo que hay en el foco:

1. `predecir_foco(gauss)` / `predecir_foco(dona)`: qué foco (ancho, intensidad pico,
   potencia) dan los `w0`/`I0` actuales. Se compara con lo medido.
2. `calibrar_haz(haz, w_foco_nm=..., P_mW=... y/o I_pico_kWcm2=...)`: invierte el
   problema. Como PyFocus es lineal en el campo, no hace falta barrer (w0, I0) en
   2D: la forma del foco depende solo de `w0` (se barre y se afina con brentq) e
   `I0` sale por cociente con la potencia o la intensidad pico medidas. Si se
   miden las tres cosas, la tercera es un chequeo de consistencia.
3. `guardar_calibracion(res)` escribe `calibracion.json`; `beams/*.py` lo leen solos
   (`sim_config.cargar_calibracion`). `USAR_CALIBRACION = False` en `sim_config.py`
   vuelve a los valores escritos a mano. `borrar_calibracion()` limpia el json.

Métricas de ancho: `'fwhm'` y `'r_1e2'` (gauss), `'r_anillo'` y `'grosor_anillo'` (dona).
Unidades: ancho en nm, `P_mW` en la muestra, `I_pico_kWcm2` en kW/cm², `I0` en mW/cm².

## Correcciones de esta versión

- **λ/n aplicada dos veces (PyCustomFocus 3.4.0).** `main_calculation_handler.py` y
  `base.py` dividen ambos la longitud de onda por n: con n>1 el foco sale ~n veces más
  chico. `pyfocus_wrapper.detectar_doble_division()` lo mide en cada entorno (3 s) y,
  si hace falta, pasa λ·n.
- **Factor campo→SI.** PyFocus no incluye el sqrt(n1/n2) de Richards-Wolf: con
  `n_entrada=1` la potencia en el foco salía n veces la real. Ahora
  `calcular_factor_E_foco(n_foco)` (con el n de cada haz). Con eso,
  I_foco[mW/cm²] = |E_PyFocus|².
- **Haz contrapropagante en XZ.** `Fz_g - Fz_d` invertía también la fuerza de gradiente
  (impar en z). Ahora `espejar_haz_contrapropagante` (z → -z) en `main.py` y `grafico_lindo.py`.
- `graficar_intensidad` usa el n de cada haz; `main.py` guardaba `n_medio=n_g`; el `np.load`
  del final apuntaba a un .npz que no existe.
- Los .npz y figuras que vienen del zip anterior se hicieron con el pipeline viejo:
  hay que volver a correr `main.py`.
