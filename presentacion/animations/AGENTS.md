# Instrucciones para agentes creadores de animaciones

Este archivo es la fuente de contexto para cualquier agente que cree o modifique animaciones de la presentacion del TFG en `presentacion/`. Antes de tocar una animacion, lee este archivo completo.

Hook asociado:

- Manifest: `.agents/hooks.json`
- Script: `.agents/hooks/read_presentacion_agents.ps1`
- El hook imprime directamente este archivo para inyectar el contexto en agentes de animacion.

## Objetivo general

La presentacion necesita animaciones profesionales para defender un TFG sobre modelos sustitutos, procesos gaussianos y optimizacion con evaluaciones costosas. Las animaciones deben explicar ideas matematicas visualmente, no funcionar como diapositivas con texto.

Prioridad de diseno:

1. La idea se entiende por movimiento, color, ritmo y composicion.
2. El texto solo etiqueta lo imprescindible.
3. Las ecuaciones deben ser LaTeX o estilo LaTeX.
4. El foco visual debe estar claro en cada momento.
5. No mostrar varias ideas importantes compitiendo a la vez.

## Preferencias del usuario

El usuario quiere un estilo cercano a 3Blue1Brown o Welch Labs: limpio, matematico, elegante y con buen ritmo visual. Le gusta la calidad de diseno de las animaciones ya creadas, pero no quiere ejemplos que parezcan triviales.

Comentarios importantes del usuario:

- "Poco texto, mas animacion, no tanta info de texto y lectura."
- "La fuente quiero que sea una muy parecida a la de LaTeX y si hay ecuaciones, que sean ecuaciones en LaTeX."
- "El foco sea facil de mantener, no que haya que mirar muchas cosas a la vez."
- Sobre la animacion de modelo sustituto: el primer ejemplo era demasiado sencillo porque el sustituto parecia perfectamente ajustado.
- Sobre la caja negra decorada con formas internas: no tenia sentido. No reutilizar esa idea sin replantearla.

## Reglas visuales

- Usar formato 16:9, 1920x1080.
- Render final preferido: `-qh` con Manim, 1080p60.
- Mantener pocos elementos persistentes en pantalla.
- Evitar texto explicativo largo salvo una frase final si es conceptualmente necesaria.
- Si una frase final aparece, debe entrar al final, no competir durante la animacion.
- No usar decoracion arbitraria. Todo elemento visual debe tener una razon semantica clara.
- Si se representa una caja negra, debe parecer un sistema opaco/costoso, no una funcion conocida dibujada dentro.
- Una caja negra puede usar: bloque oscuro, entrada/salida, reloj/barra/coste, vibracion sutil, borde activo, o espera visible.
- Evitar curvas internas decorativas en la caja negra si sugieren que conocemos la funcion real.
- En una animacion de modelos sustitutos, no hacer que `\hat f(x)` interpole perfectamente todos los puntos salvo que el guion lo pida explicitamente.
- Mostrar desajuste pequeno entre datos reales y sustituto cuando se quiera transmitir aproximacion.
- Evitar ejemplos demasiado suaves o triviales para problemas de caja negra.
- Los puntos candidatos pueden aparecer rapidamente para contrastar con evaluaciones reales lentas.

## Paletas usadas

Animaciones oscuras, estilo GP:

- Fondo: `#05070d`
- Ejes: `#6f788a`
- Texto secundario: `#aeb6c6`
- Blanco: `#f2f4fb`
- Azul: `#58c4dd`
- Verde: `#83c167`
- Amarillo: `#f4d35e`
- Rosa: `#ff6f91`
- Morado: `#9a7ff0`

Animacion clara de modelo sustituto:

- Fondo: `#E8E8E8`
- Texto principal: `#1F2933`
- Azul principal / datos: `#174C5B`
- Ambar / coste: `#C9822B`
- Verde / sustituto rapido: `#2F7D6D`
- Gris secundario: `#5F6368`
- Blanco suave: `#F8FAFC`

## Tipografia y ecuaciones

El sistema no tiene una instalacion completa de LaTeX/dvisvgm, por lo que las animaciones actuales renderizan ecuaciones con Matplotlib mathtext y fuente Computer Modern a PNG transparente.

Patron usado:

- Funcion `_latex(tex, height, color)` devuelve un `ImageMobject`.
- Funcion `_latex_png(tex, color)` guarda PNGs cacheados en `presentacion/assets/latex_cache`.
- `mpl.rcParams["mathtext.fontset"] = "cm"`.
- `mpl.rcParams["font.family"] = "serif"`.

Para etiquetas de interfaz en la animacion clara se usa `Text(..., font="Segoe UI")`.

No usar `Text` para ecuaciones LaTeX si produce espaciado raro o mala calidad.

## Archivos actuales

Animacion 0, modelo sustituto:

- Fuente: `presentacion/animacion_0_modelo_sustituto.py`
- Video: `presentacion/renders/animacion_0_modelo_sustituto_1080p60.mp4`
- Captura final: `presentacion/renders/animacion_0_modelo_sustituto_final.png`
- Tema: fondo claro.
- Idea: pocas evaluaciones reales costosas de `f(x)` permiten ajustar una aproximacion barata `\hat f(x)`, que sirve para explorar candidatos y decidir que evaluar despues.
- Restricciones conceptuales: no introducir procesos gaussianos, incertidumbre, EI, kernels ni `x_next`.
- Estado de diseno: la parte de datos y sustituto va en buena direccion; revisar la caja negra si se modifica, porque el usuario rechazo decoracion interna sin sentido.

Animacion 1, prior GP y kernel:

- Fuente: `presentacion/animacion_1_gp_prior_kernel.py`
- Video: `presentacion/renders/animacion_1_gp_prior_kernel_1080p60.mp4`
- Guion: `presentacion/renders/animacion_1_gp_prior_kernel_guion.md`
- Tema: fondo oscuro.
- Idea: muestras de un GP prior y efecto de la longitud de escala `\ell` sobre correlacion y suavidad.

Animacion 2, posterior GP:

- Fuente: `presentacion/animacion_2_gp_posterior.py`
- Video: `presentacion/renders/animacion_2_gp_posterior_1080p60.mp4`
- Tema: fondo oscuro.
- Idea: al observar datos, la media posterior se adapta y la incertidumbre baja cerca de observaciones.

## Comandos de render

Instalar dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip install -r presentacion\requirements-presentacion.txt
```

Render rapido:

```powershell
.\.venv\Scripts\python.exe -m manim -ql presentacion\animacion_0_modelo_sustituto.py Animacion0ModeloSustituto
```

Render final:

```powershell
.\.venv\Scripts\python.exe -m manim -qh presentacion\animacion_0_modelo_sustituto.py Animacion0ModeloSustituto
```

Copiar entregable final:

```powershell
Copy-Item -LiteralPath media\videos\animacion_0_modelo_sustituto\1080p60\Animacion0ModeloSustituto.mp4 -Destination presentacion\renders\animacion_0_modelo_sustituto_1080p60.mp4 -Force
```

Crear captura final:

```powershell
ffmpeg -y -ss 00:00:38 -i presentacion\renders\animacion_0_modelo_sustituto_1080p60.mp4 -frames:v 1 presentacion\renders\animacion_0_modelo_sustituto_final.png
```

Validar video:

```powershell
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1 presentacion\renders\animacion_0_modelo_sustituto_1080p60.mp4
```

## Criterios para nuevas animaciones

Antes de implementar:

1. Identificar la idea central que debe entenderse sin narracion.
2. Decidir que objetos visuales llevan esa idea.
3. Reducir el texto a etiquetas.
4. Evitar introducir conceptos de animaciones posteriores antes de tiempo.
5. Si hay duda de diseno conceptual, preguntar al usuario.

Durante la implementacion:

- Reutilizar patrones de las animaciones existentes.
- Mantener estilos, colores y helpers coherentes.
- No crear abstracciones grandes si la escena es unica.
- Usar datos sinteticos con forma visual honesta: ni demasiado perfecta ni demasiado caotica.
- Si se muestra una aproximacion, que se vea como aproximacion.
- Si se muestra coste, que el tiempo de animacion lo comunique.
- Si se muestra rapidez, usar muchas consultas en rafaga sin reloj ni barra.

Despues de implementar:

- Compilar con `py_compile`.
- Render rapido `-ql`.
- Extraer capturas de momentos clave.
- Revisar solapes, jerarquia visual y si hay demasiado texto.
- Render final `-qh`.
- Copiar a `presentacion/renders`.
- Generar PNG final si puede servir para PowerPoint.
- Validar `1920x1080`, fps y duracion con `ffprobe`.

## Advertencias especificas

- No borrar archivos importantes ni renders existentes sin permiso.
- No introducir GP, incertidumbre o Expected Improvement en la animacion 0.
- No hacer que la caja negra parezca una funcion conocida.
- No hacer que un modelo sustituto parezca validacion experimental.
- No usar texto largo como sustituto de animacion.
- No usar decoraciones sin significado.
