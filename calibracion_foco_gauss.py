# -*- coding: utf-8 -*-
"""
calibracion_foco_gauss.py

Prediccion rapida del foco del haz gaussiano para unos w0/I0 del BFP
puestos a mano (sin tocar beams/gauss.py ni calibracion.json).
Toda la logica vive en calibracion_foco.py.
"""
from beams import gauss
from calibracion_foco import predecir_foco

if __name__ == '__main__':
    # Parametros BFP del haz gaussiano (mm y mW/cm^2)
    predecir_foco(gauss, w0_mm=3.7, I0_mWcm2=10.0)
