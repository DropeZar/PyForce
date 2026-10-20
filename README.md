# PyForce

Simulación de las fuerzas ópticas en el **printing óptico** de nanopartículas
de oro: la luz empuja la partícula, que está en agua, hasta el portaobjetos de
vidrio, donde queda impresa. Se usan dos haces enfocados por objetivos opuestos
(contrapropagantes): un haz gaussiano y un haz dona (vórtice). Los campos en el
foco se calculan con [PyFocus](https://github.com/Stefani-Lab/PyFocus)
(Richards-Wolf), o con `interfaz.py` cuando hace falta la interfaz agua/vidrio,
y a partir de ellos se obtienen las fuerzas de gradiente, scattering y
spin-curl en los planos XY y XZ.

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
python fuerzas_gauss_interfaz.py   # printing con el gauss, con y sin vidrio -> resultados/gauss_interfaz/
python validar_interfaz.py         # chequeos de interfaz.py y fuerzas.py
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
| `fuerzas.py` | Funciones puras: factor de conversión a V/m, fuerzas (gradiente + scattering + spin-curl) en XY y XZ, espejado del haz contrapropagante. |
| `graficos.py` | Mapas de magnitud + streamplot, comparación entre haces, desglose en componentes e intensidad. |
| `grafico_lindo.py` | Figura resumen 3x3 (XY, XZ y corte de Fz en z = 0 con FWHM) a partir de los `.npz`. |
| `pyfocus_wrapper.py` | Capa entre el código y PyFocus: lo corre sin abrir figuras, corrige la doble división λ/n, pasa la salida a la convención física (Ez y eje y) y da el factor de conversión a SI. |
| `calibracion_foco.py` | De lo medido en el foco a los `w0`/`I0` del back focal plane (ver abajo). |
| `calibracion_foco_gauss.py` | Predicción rápida del foco del haz gaussiano para unos `w0`/`I0` a mano. |
| `calibracion.json` | Resultado de la calibración; lo leen `beams/*.py` al importarse. |
| `interfaz.py` | Foco de un haz gaussiano con una interfaz plana (agua/vidrio): Richards-Wolf + Fresnel, mismas grillas y unidades que PyFocus, y derivadas exactas fuera del plano. |
| `validar_interfaz.py` | Chequea `interfaz.py` (coincide con `sim.no_mask` a 1e-14, condiciones de borde, div E = 0) y que el total de `fuerzas.py` sea la fuerza dipolar exacta. |
| `fuerzas_gauss_interfaz.py` | Printing con el gauss calibrado: fuerzas sin vidrio y con el vidrio en el foco. |
| `resultados/` | Figuras y resúmenes publicados (ver [Resultados](#resultados)). |
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

**Descomposición de la fuerza.** La fuerza sobre el dipolo,
`F = (ε/2) Re[α* Σ E_j* ∇E_j]`, se separa en gradiente + scattering (vector de
Poynting) + spin-curl, y `fuerzas_xy`/`fuerzas_xz` suman las tres. El
scattering tiene que ser el de Poynting: el término "orbital"
`Im[E_j* ∇E_j]` ya incluye el spin-curl, y sumarlo de nuevo lo cuenta dos veces.
`validar_interfaz.py` comprueba que el total coincide con la fuerza dipolar.

**Derivadas fuera del plano.** Scattering y spin-curl necesitan derivadas
perpendiculares al plano calculado (∂/∂z en XY, ∂/∂y en XZ). `interfaz.py` las
da exactas. Con un solo plano de PyFocus, la de la componente normal sale de
div E = 0 y las otras se toman como cero. Para polarización lineal en x eso
es exacto en XZ, pero con polarización circular (la dona) el spin-curl queda
~50 % mal.

**Interfaz agua/vidrio.** El portaobjetos (n = 1.52, con aceite de índice
igualado abajo) se trata como semi-infinito. PyFocus no sirve para esto:
`sim.no_mask`/`sim.VP` ignoran la interfaz, y en `sim.custom` (3.4.0) la onda
reflejada de la componente x no lleva la fase `exp(2i k1 cosθ z_int)`, así que
con polarización en x solo es correcta con la interfaz en el foco.
`interfaz.py` lo resuelve por su cuenta (ver su docstring).

**Convención de PyFocus.** `sim.no_mask`/`sim.VP` devuelven Ez con el signo
opuesto al físico (con su Ez, div E ≠ 0) y el plano XY con las filas de +y a −y.
`pyfocus_wrapper.campo_foco` corrige las dos cosas. Quedan dos detalles del
plano XY de PyFocus: la grilla está corrida medio píxel en y, y el radio de
cada píxel se redondea, lo que mete ruido en las derivadas.

## Resultados

### Printing con el haz gaussiano, con y sin vidrio (`resultados/gauss_interfaz/`)

Gauss calibrado (FWHM = 330 nm, P = 1 mW), NP de Au de 40 nm en agua, foco
sobre la superficie del vidrio (z = 0). La NP queda impresa con el centro en
z = −a = −40 nm (plano de contacto). Tabla completa en
[`resumen.md`](resultados/gauss_interfaz/resumen.md).

![Fuerzas XZ](resultados/gauss_interfaz/fuerzas_xz.png)
![Perfiles](resultados/gauss_interfaz/perfiles.png)
![Plano de contacto](resultados/gauss_interfaz/fuerzas_xy.png)

- **Empuje hacia el vidrio en el contacto: baja casi a la mitad** (de 0.081 a
  0.043 pN a 1 mW). La reflexión en el vidrio (≈ 0.5 % en intensidad) forma
  una onda estacionaria con período λ/2n ≈ 220 nm. Justo en el contacto, su
  gradiente apunta hacia arriba (−0.035 pN) y le resta al scattering
  (+0.112 pN). El spin-curl también se opone (−0.034 pN), con y sin vidrio.
- **En el camino hacia el vidrio**, Fz oscila alrededor del valor sin vidrio
  pero es positiva en todo el último 1.4 µm sobre el eje: la onda
  estacionaria no frena a la NP antes de llegar al vidrio.
- **Guiado lateral en el contacto: cambia poco.** La fuerza que lleva la NP
  al eje baja ~9 % (−dFx/dx de 1.41 a 1.28 pN/µm, máximo de 0.140 a
  0.129 pN). El radio dentro del cual la NP es llevada al eje queda en
  ~465 nm. Casi toda la fuerza lateral es de gradiente.

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
- Agregar la interfaz al haz dona (necesita la máscara de vórtice en
  `interfaz.py`) y simular el printing con los dos haces y el vidrio. Eso
  además da las derivadas exactas fuera del plano para la dona; con
  PyFocus, su spin-curl queda aproximado (ver arriba).
- Falta calibrar el haz dona (`calibracion.json` solo tiene el gauss).
