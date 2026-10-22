# Haz gaussiano: fuerzas en el foco, con y sin el vidrio

- Haz: NA = 1, n = 1.33 (agua), λ = 592 nm, polarización lineal en x, w0 = 5.658 mm, I0 = 4.624 mW/cm² (calibrado: FWHM = 330 nm, P = 1 mW en la muestra).
- Partícula: Au, 100 nm de diámetro, en agua; α' = 2.330e-21 m³, α'' = 3.331e-21 m³ (Mie, dipolo a1).
- Vidrio: n = 1.52, semi-infinito, con la superficie en el foco (z = 0).
- Campos: PyFocus (sim.custom con la interfaz). Fuerza total = conservativa (gradiente) + no conservativa.

## En el foco (x = y = 0, z = 0; con vidrio, del lado del agua)

| caso | Fz total (pN) | conservativa (gradiente) | no conservativa |
|---|---|---|---|
| sin vidrio | 1.594 | 0.000 | 1.593 |
| con vidrio | 1.581 | -0.004 | 1.585 |

## Potencia para llegar a 0.6 pN en el foco

| caso | Fz en el foco a 1 mW (pN) | potencia necesaria (mW) |
|---|---|---|
| sin vidrio | 1.594 | 0.38 |
| con vidrio | 1.581 | 0.38 |

Fuerzas a P = 1 mW en la muestra (escalan linealmente con P). Fz > 0 empuja la NP hacia el vidrio.
