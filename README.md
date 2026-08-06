# Simulación de trampa óptica (haces contrapropagantes gauss + dona)

## Cómo está organizado

- **`config.py`** — Único lugar donde se definen los parámetros compartidos
  (radio de la partícula, índices de refracción de los medios, `I0`,
  resolución/rango de PyFocus, paso espacial para las derivadas). Si hay
  que cambiar alguno de estos valores, se cambia acá — no en los otros
  archivos.
- **`particula.py`** — Polarizabilidad de Kuwata para la nanoesfera de
  oro, con la tabla completa de Johnson & Christy (1972). `calc_alpha(wavelength_nm)`
  usa por defecto el radio y el medio de `config.py`.
- **`beams/gauss.py`**, **`beams/dona.py`** — Definición de cada haz en
  PyFocus (`NA`, `h`, `w0`, `wavelength`, polarización) y su `get_field()`.
  Los parámetros que comparten entre sí (I0, resolución, rango) vienen de
  `config.py`.
- **`fuerzas.py`** — Funciones puras: `calcular_factor_E` (conversión de
  campos simulados a unidades SI) y `fuerzas_xy` / `fuerzas_xz` (fuerza de
  gradiente + scattering + spin-curl). Sin gráficos ni prints.
- **`graficos.py`** — Funciones de graficado reutilizables: mapa de
  magnitud + streamplot (`graficar_fuerza_total`), comparación lado a lado
  gauss/dona/resultante (`graficar_comparacion`), y desglose en
  componentes (`graficar_componentes`).
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

3. Calibración de campo a unidades SI (`calcular_factor_E`)

Se revisó el código fuente real de PyFocus (`_calculate_amplitude_factors`
en `model/focus_field_calculators/base.py`) y se confirmó que PyFocus usa
`I0` literalmente como `|E0|²` — sin el factor físico `n·c·ε₀/2` que
relaciona intensidad real con campo eléctrico real. Por eso el factor de
conversión a V/m es una **constante** (no depende del campo simulado ni
del punto que se evalúe): `factor_E = sqrt(20 / (n_entrada · c · ε₀))`,
con `n_entrada = 1` (aire) porque el `I0` de este montaje se define antes
del objetivo. El detalle completo está en el docstring de
`calcular_factor_E` en `fuerzas.py`.
