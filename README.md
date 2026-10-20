# PyForce

Simulación de fuerzas ópticas sobre una nanopartícula de oro en una **trampa
óptica contrapropagante**: un haz gaussiano y un haz dona (vórtice) enfocados
por objetivos opuestos. Los campos en el foco se calculan con
[PyFocus](https://github.com/Stefani-Lab/PyFocus) (Richards-Wolf) y a partir de
ellos se obtienen las fuerzas de gradiente, scattering y spin-curl en los planos
XY y XZ.

Proyecto de Laboratorio 6 y 7.

## Instalación

Requiere Python 3.10 (es la versión con la que se probó PyCustomFocus 3.4.0).

```bash
conda create -n pyfocus python=3.10
conda activate pyfocus
pip install -r requirements.txt
```

## Uso

```bash
python main.py            # campos -> fuerzas -> gráficos; guarda los campos en .npz
python grafico_lindo.py   # figura resumen 3x3 a partir de los .npz
python particula.py       # comparación Kuwata vs Mie de la polarizabilidad
python calibracion_foco.py
```

`main.py` y `calibracion_foco.py` están divididos en celdas (`#%%`) para
correrlos de a partes en Spyder o VS Code.

Para reusar los campos guardados en otro script:

```python
import numpy as np
data = np.load('campos_dona_aceite.npz')
Ex_xy, Ey_xy, Ez_xy = data['Ex_xy'], data['Ey_xy'], data['Ez_xy']   # V/m
Ex_xz, Ey_xz, Ez_xz = data['Ex_xz'], data['Ey_xz'], data['Ez_xz']
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `sim_config.py` | **Único lugar** con los parámetros compartidos: radio de la partícula, índices de los medios, grilla de PyFocus, paso de las derivadas y extensión de los planos XY/XZ. |
| `beams/gauss.py`, `beams/dona.py` | Parámetros propios de cada haz (`NA`, `n`, `h`, `w0`, `λ`, polarización, `I0`) y su `get_field()`. |
| `particula.py` | Polarizabilidad del oro (tabla de Johnson & Christy 1972): Kuwata (`calc_alpha`, la que se usa) y Mie exacto (`calc_alpha_mie`, para validar). |
| `fuerzas.py` | Funciones puras: factor de conversión a V/m, fuerzas en XY y XZ, espejado del haz contrapropagante. |
| `graficos.py` | Mapas de magnitud + streamplot, comparación entre haces, desglose en componentes e intensidad. |
| `grafico_lindo.py` | Figura resumen 3x3 (XY, XZ y corte de Fz en z = 0 con FWHM) a partir de los `.npz`. |
| `pyfocus_wrapper.py` | Capa entre el código y PyFocus: lo corre sin abrir figuras, corrige la doble división λ/n y da el factor de conversión a SI. |
| `calibracion_foco.py` | De lo medido en el foco a los `w0`/`I0` del back focal plane (ver abajo). |
| `calibracion_foco_gauss.py` | Predicción rápida del foco del haz gaussiano para unos `w0`/`I0` a mano. |
| `calibracion.json` | Resultado de la calibración; lo leen `beams/*.py` al importarse. |
| `Objetivos.txt` | NA, n, h y f de los objetivos disponibles. |

## Detalles de la física y de PyFocus

**Unidades de PyFocus.** PyFocus usa `I0` literalmente como `|E0|²` (sin el
factor `n·c·ε₀/2`) y no incluye el `sqrt(n1/n2)` de Richards-Wolf. El factor
para pasar a V/m es entonces una constante que depende del índice del medio del
foco: `E_SI = E_PyFocus · sqrt(20 / (n_foco · c · ε₀))`. Con eso,
`I_foco [mW/cm²] = |E_PyFocus|²`. Ver `pyfocus_wrapper.factor_E_foco`.

**λ/n aplicada dos veces (PyCustomFocus 3.4.0).** `main_calculation_handler.py`
y `base.py` dividen ambos la longitud de onda por n: con n > 1 el foco sale
~n veces más chico. `pyfocus_wrapper.detectar_doble_division()` lo mide en cada
entorno (~3 s) y, si hace falta, le pasa λ·n.

**Grilla.** El paso entre puntos es `x_steps`/`z_steps` (5 nm); las derivadas
de `fuerzas.py` usan ese mismo paso. El plano XY cubre `x_range` en x e y. El
plano XZ cubre `z_range` en z pero `√2·x_range` en x (PyFocus llega hasta la
esquina de la grilla XY), y sus filas van de −z a +z.

**Haz contrapropagante.** La dona viaja hacia −z: sus fuerzas XZ son la imagen
especular (z → −z) de las simuladas (`fuerzas.espejar_haz_contrapropagante`).

**Spin-curl en XZ.** `fuerzas_xz` devuelve el término de spin-curl por separado
pero no lo suma al total (sí lo hace `fuerzas_xy`).

## Calibración foco → BFP (`calibracion_foco.py`)

`w0` e `I0` de `beams/gauss.py` y `beams/dona.py` son los del back focal plane
(antes del objetivo), pero lo que se mide es lo que hay en el foco:

1. `predecir_foco(gauss)` / `predecir_foco(dona)`: qué foco (ancho, intensidad
   pico, potencia) dan los `w0`/`I0` actuales, para comparar con lo medido.
2. `calibrar_haz(haz, w_foco_nm=..., P_mW=... y/o I_pico_kWcm2=...)`: invierte
   el problema. Como PyFocus es lineal en el campo, la forma del foco depende
   solo de `w0` (se barre y se afina con `brentq`) e `I0` sale por cociente con
   la potencia o la intensidad pico. Si se miden las tres cosas, la tercera
   queda como chequeo de consistencia.
3. `guardar_calibracion(res)` escribe `calibracion.json`, que `beams/*.py` leen
   solos. `USAR_CALIBRACION = False` en `sim_config.py` vuelve a los valores
   escritos a mano; `borrar_calibracion()` limpia el json.

Métricas de ancho: `'fwhm'` y `'r_1e2'` (gauss), `'r_anillo'` y
`'grosor_anillo'` (dona). Unidades: ancho en nm, `P_mW` en la muestra,
`I_pico_kWcm2` en kW/cm², `I0` en mW/cm².

## Pendientes

- La posición del máximo de la polarizabilidad de Kuwata cambia poco con el
  radio de la NP. Puede ser el límite del modelo; `particula.comparar_radios()`
  lo compara con Mie exacto.
- El haz gaussiano usa `n = 1.35` en el foco mientras que la partícula está en
  agua (`n_g = 1.33` en `sim_config.py`): revisar cuál corresponde.
- Falta calibrar el haz dona (`calibracion.json` solo tiene el gauss).
