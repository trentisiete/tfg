# Animaciones para la presentacion

## Animacion 0: que es un modelo sustituto

Archivo principal:

```powershell
presentacion\animacion_0_modelo_sustituto.py
```

Render rapido para revisar:

```powershell
.\.venv\Scripts\python.exe -m manim -ql presentacion\animacion_0_modelo_sustituto.py Animacion0ModeloSustituto
```

Render en alta calidad para insertar en la presentacion:

```powershell
.\.venv\Scripts\python.exe -m manim -qh presentacion\animacion_0_modelo_sustituto.py Animacion0ModeloSustituto
```

Video final:

```powershell
presentacion\renders\animacion_0_modelo_sustituto_1080p60.mp4
```

Captura final:

```powershell
presentacion\renders\animacion_0_modelo_sustituto_final.png
```

## Animacion 1: prior de un Proceso Gaussiano y efecto del kernel

Archivo principal:

```powershell
presentacion\animacion_1_gp_prior_kernel.py
```

La escena usa Manim para la animacion y genera las ecuaciones desde sintaxis LaTeX con fuente Computer Modern. La cache de esas ecuaciones se guarda en:

```powershell
presentacion\assets\latex_cache
```

Instalar dependencias de la presentacion:

```powershell
.\.venv\Scripts\python.exe -m pip install -r presentacion\requirements-presentacion.txt
```

Render rapido para revisar:

```powershell
.\.venv\Scripts\python.exe -m manim -ql presentacion\animacion_1_gp_prior_kernel.py Animacion1GPPriorKernel
```

Render en alta calidad para insertar en la presentacion:

```powershell
.\.venv\Scripts\python.exe -m manim -qh presentacion\animacion_1_gp_prior_kernel.py Animacion1GPPriorKernel
```

Video final ya generado:

```powershell
presentacion\renders\animacion_1_gp_prior_kernel_1080p60.mp4
```

Salida original de Manim:

```powershell
media\videos\animacion_1_gp_prior_kernel\1080p60\Animacion1GPPriorKernel.mp4
```

## Animacion 2: actualizacion posterior al observar datos

Archivo principal:

```powershell
presentacion\animacion_2_gp_posterior.py
```

Render rapido para revisar:

```powershell
.\.venv\Scripts\python.exe -m manim -ql presentacion\animacion_2_gp_posterior.py Animacion2GPPosterior
```

Render en alta calidad para insertar en la presentacion:

```powershell
.\.venv\Scripts\python.exe -m manim -qh presentacion\animacion_2_gp_posterior.py Animacion2GPPosterior
```

Video final:

```powershell
presentacion\renders\animacion_2_gp_posterior_1080p60.mp4
```
