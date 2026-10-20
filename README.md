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
python particula.py       # polarizabilidad de Mie, Mie completo vs dipolo, comparación con Kuwata
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
| `particula.py` | Polarizabilidad del oro con teoría de Mie (tabla de Johnson & Christy 1972): `calc_alpha` (dipolo a1, la que se usa), coeficientes y secciones eficaces completas, y Kuwata solo para comparar. |
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

## Fuerzas sobre la partícula

La NP de oro (diámetro 100 nm) es chica frente a la longitud de onda en agua
(λ/n ≈ 400–450 nm), así que se la trata como un **dipolo** inducido
`p = ε₀ εm α E`. La polarizabilidad α sale de la teoría de Mie (coeficiente
a1, `particula.calc_alpha`). Sus dos partes juegan papeles distintos:

- **α′ (parte real):** la NP responde en fase con el campo. Da la fuerza de
  gradiente.
- **α″ (parte imaginaria):** la NP absorbe y dispersa luz. Da el scattering y
  el spin-curl.

Para el oro, cerca de la resonancia plasmónica, α″ es del orden de α′ o
mayor:

| λ (haz) | α′ (m³) | α″ (m³) |
|---|---|---|
| 592 nm (gauss) | 2.33e-21 | 3.33e-21 |
| 532 nm (dona) | −0.38e-21 | 2.44e-21 |

La fuerza sobre el dipolo, `F = (ε₀ εm / 2) Re[α* Σⱼ Eⱼ* ∇Eⱼ]`, se separa en
tres términos (`fuerzas.py` los calcula por separado y los suma):

1. **Gradiente** — `F_grad = (ε₀ εm α′ / 4) ∇|E|²`.
   Es conservativa. Con α′ > 0 lleva la NP hacia donde la luz es más
   intensa; con α′ < 0, hacia donde es menos intensa. A 532 nm, α′ < 0 para
   esta NP: el gradiente de la dona la empuja hacia el centro oscuro del
   anillo, no hacia el anillo.
2. **Scattering (presión de radiación)** — `F_scat = (ε₀ εm α″ / 2) Im[E* × (∇ × E)]`.
   Sigue el flujo de energía (vector de Poynting), o sea, empuja en el
   sentido en que viaja la luz. Para el oro es la fuerza dominante: es la que
   **lleva la NP hasta el vidrio y la imprime**.
3. **Spin-curl** — `F_sc = (ε₀ εm α″ / 2) Im[(E* · ∇) E]`.
   Viene del rotor de la densidad de espín de la luz. En un foco muy cerrado
   el campo tiene componente Ez en cuadratura con Ex, y aparece aunque el haz
   esté polarizado linealmente. En el foco del gauss **se opone al
   scattering** y le resta ~15 %.

Gradiente + scattering + spin-curl da exactamente la fuerza dipolar
(`validar_interfaz.py` lo comprueba). El scattering tiene que ser el de
Poynting: el término "orbital" `Im[Eⱼ* ∇Eⱼ]` ya incluye el spin-curl.

**Para imprimir** hace falta que la fuerza hacia el vidrio en el punto de
contacto (centro de la NP a un radio del vidrio) llegue a
**≈ 0.6 pN** (medido; `sim_config.F_IMPRESION_pN`). Lateralmente, el
gradiente lleva la NP hacia el eje del haz, lo que fija dónde queda impresa.

**Validez del modelo dipolar.** Con a = 50 nm, los multipolos más altos de Mie
pesan poco. Comparando la presión de radiación de una onda plana con Mie
completo y solo con a1 (`particula.secciones_eficaces`), el dipolo la
subestima un 1 % a 592 nm y un 4 % a 532 nm. No se incluyen la interacción de
la NP con su propia imagen en el vidrio ni fuerzas de contacto (van der
Waals, electrostáticas).

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

Gauss calibrado (FWHM = 330 nm, P = 1 mW), NP de Au de 100 nm de diámetro en
agua, foco sobre la superficie del vidrio (z = 0). La NP queda impresa con el
centro en z = −a = −50 nm (punto de contacto). Tabla completa en
[`resumen.md`](resultados/gauss_interfaz/resumen.md).

![Fuerzas XZ](resultados/gauss_interfaz/fuerzas_xz.png)
![Plano de contacto](resultados/gauss_interfaz/fuerzas_xy.png)

Fuerza hacia el vidrio en el punto de contacto, sobre el eje, a 1 mW:

| caso | total | gradiente | scattering | spin-curl | potencia para 0.6 pN |
|---|---|---|---|---|---|
| sin vidrio | 1.59 pN | +0.01 | +1.87 | −0.28 | 0.38 mW |
| con vidrio | 1.44 pN | −0.14 | +1.86 | −0.28 | 0.42 mW |

- **El scattering es lo que imprime:** empuja la NP hacia el vidrio con
  ~1.9 pN a 1 mW. El spin-curl le resta ~15 %.
- **El vidrio reduce el empuje en el contacto ~10 %.** La reflexión en el
  vidrio (≈ 0.5 % en intensidad) forma una onda estacionaria. Justo en el
  contacto, su gradiente apunta hacia arriba (−0.14 pN).
- **Con 0.42 mW alcanza para imprimir** (0.6 pN en el contacto).
- **En el plano de contacto la fuerza lateral apunta al eje** y casi no
  cambia con el vidrio (~3 %): la NP se imprime centrada en el foco.

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

- Las fuerzas usan la aproximación dipolar (α de Mie con a1). Para ir más
  allá (multipolos en un haz enfocado) hace falta Mie generalizada
  (GLMT / matriz T); con a = 50 nm la diferencia esperada es de 1–4 %.
- Agregar la interfaz al haz dona (necesita la máscara de vórtice en
  `interfaz.py`) y simular el printing con los dos haces y el vidrio. Eso
  además da las derivadas exactas fuera del plano para la dona; con
  PyFocus, su spin-curl queda aproximado (ver arriba).
- Falta calibrar el haz dona (`calibracion.json` solo tiene el gauss).
