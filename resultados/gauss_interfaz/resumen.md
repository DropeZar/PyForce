# Printing con el haz gaussiano: fuerzas con y sin el vidrio

- Haz: NA = 1, n = 1.33 (agua), λ = 592 nm, polarización lineal en x, w0 = 5.658 mm, I0 = 4.624 mW/cm² (calibrado: FWHM = 330 nm, P = 1 mW en la muestra).
- Partícula: Au, a = 50 nm, en agua; α' = 2.330e-21 m³, α'' = 3.331e-21 m³ (Mie, dipolo a1).
- Vidrio: n = 1.52, semi-infinito, con la superficie en el foco (z = 0). Contacto: centro de la NP en z = −a = -50 nm.
- Campos: PyFocus (sim.custom con la interfaz). Fuerza total = gradiente + scattering + spin-curl.

## En el punto de contacto sobre el eje (x = y = 0, z = −a)

| caso | Fz total (pN) | gradiente | scattering | spin-curl |
|---|---|---|---|---|
| sin vidrio | 1.591 | 0.009 | 1.865 | -0.283 |
| con vidrio | 1.440 | -0.134 | 1.846 | -0.272 |

## Potencia para imprimir (Fz en el contacto = 0.6 pN)

| caso | Fz en el contacto a 1 mW (pN) | potencia necesaria (mW) |
|---|---|---|
| sin vidrio | 1.591 | 0.38 |
| con vidrio | 1.440 | 0.42 |

Fuerzas a P = 1 mW en la muestra (escalan linealmente con P). Fz > 0 empuja la NP hacia el vidrio.
