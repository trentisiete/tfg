# Guia de estilo - defensa TFG

Fuente canonica: `defensa.pdf`, paginas 1-4. El archivo `defensa (1).pptx` se reviso solo como comprobacion de contenido, pero no debe usarse como referencia visual si al abrirlo o descargarlo se degrada: el PDF manda para posicion, espaciado, formulas, flechas y jerarquia. Las paginas 5 en adelante del PDF no deben tratarse como referencia visual final.

## Vision general

Presentacion academica, moderna y sobria. El estilo debe sentirse limpio, cientifico y directo: mucho aire, pocas palabras, diagramas simples y una narrativa visual de "datos -> informacion -> respuestas -> decision" o "funcion costosa -> datos -> modelo -> decision". La prioridad es que cada diapositiva explique una sola idea con claridad inmediata.

## Formato base

- Lienzo 16:9.
- Fondo completo gris claro calido: `#E8E8E8`.
- Sin decoracion ornamental, sin gradientes y sin fotos de relleno.
- Margenes amplios: en 1920x1080 usar aprox. 80-110 px laterales y 65-85 px superiores. En 1280x720 usar aprox. 55-75 px laterales y 45-60 px superiores.
- Numero de pagina pequeno en esquina inferior derecha, color texto oscuro, sin caja.

## Paleta

- Texto principal: `#1F2933` - casi negro azulado.
- Verde petroleo principal: `#174C5B` - seccion, bullets clave, nodos de datos, tarjetas destacadas.
- Verde medio: `#2F7D6D` - decision, resultado positivo, tercer elemento de secuencia.
- Naranja mostaza: `#C9822B` - modelo, coste, trade-off, metrica secundaria.
- Gris texto secundario: `#5F6368` o `#6B7280`.
- Gris linea/borde: `#CFCFCF`, `#C9C9C9`, `#CCCCCC`.
- Blanco de tarjetas: `#F3F3F3` o `#FFFFFF`.
- Rojo apagado solo para alerta o limite: `#B45353`.

Regla: el gris claro domina. El verde petroleo da identidad. El naranja y el verde medio aparecen como acentos, no como fondos grandes salvo necesidad narrativa.

## Tipografia

- Familia dominante: Google Sans.
- Variantes observadas: Google Sans, Google Sans Medium, Google Sans SemiBold.
- Fuente secundaria detectada en el PPTX: Quattrocento Sans, usada de forma puntual o heredada.
- Fallback si falta Google Sans: Aptos o Arial, manteniendo pesos y espaciado.

Escala orientativa:

- Portada: titulo muy grande, centrado, 60-76 pt aprox., peso semibold.
- Titulo de diapositiva: 38-48 pt, arriba izquierda, peso regular/medium.
- Etiqueta de seccion: 11-14 pt, mayusculas, semibold, verde petroleo.
- Subtitulo o frase guia: 20-26 pt, una o dos lineas maximo.
- Texto de tarjeta: 18-24 pt.
- Texto secundario/captions: 12-16 pt.
- Evitar microtexto salvo pie o numero de pagina.

## Composicion

### Portada

- Todo centrado vertical y horizontalmente.
- Titulo en 3 lineas maximo, muy grande, color `#1F2933`.
- Institucion debajo en gris.
- Tipo de trabajo en verde petroleo.
- Autor, tutora y grado al final, pequeno y gris.
- Sin logos ni ornamentos si no son necesarios.

### Diapositiva explicativa

- Estructura superior: etiqueta de seccion pequena + titulo grande.
- Debajo, una frase guia o definicion breve.
- Cuerpo organizado en 2 columnas, tarjetas o diagramas de flujo.
- Separadores finos horizontales o verticales en gris claro cuando ayudan a ordenar.
- Cierre inferior con una idea fuerte en texto grande o semibold.

### Tarjetas

- Tarjetas blancas o gris muy claro, borde fino `#CFCFCF`, esquinas redondeadas moderadas.
- Sombra ausente o casi imperceptible.
- Interior con mucho padding.
- Microbarra superior corta en verde petroleo o naranja para marcar categoria.
- Una tarjeta destacada puede usar fondo verde petroleo `#174C5B` y texto blanco.

### Diagramas

- Usar nodos circulares grandes para procesos: datos `#174C5B`, modelo `#C9822B`, decision `#2F7D6D`.
- Flechas finas y discretas, en gris claro o verde petroleo. En flujos con tarjetas, usar flechas pequenas centradas verticalmente entre bloques, no flechas grandes.
- Las formulas deben aparecer grandes, aisladas y con color de acento si representan una funcion o modelo.
- Si una formula se rompe al pasar a PPTX, recrearla como ecuacion editable o como imagen vector/raster limpia; no copiar texto extraido del PDF porque puede introducir cuadros o subindices danados.
- Los diagramas deben contar una transicion, no decorar.

## Lenguaje visual para conceptos tecnicos

- Modelos sustitutos: representar como puente entre funcion real costosa y decision.
- Procesos gaussianos: mostrar prediccion + incertidumbre, no solo ecuaciones. Usar banda o region de confianza si se dibuja una curva.
- Optimizacion bayesiana: separar claramente "lo que sabemos" de "donde conviene evaluar".
- Adquisicion / EI: usar color naranja para el criterio/modelo y verde medio para la decision final.
- Presupuesto limitado: puede representarse con linea roja apagada, pequeno contador o etiqueta discreta.

## Criterios actualizados para slides matematicas v2

- Las diapositivas matematicas no deben copiar una disposicion fija: el disenador debe organizar cada slide segun la idea que haya que defender oralmente.
- Priorizar composiciones abiertas con separadores finos, alineacion clara y mucho aire. Usar cajas solo cuando aporten estructura; evitar tarjetas grandes si una secuencia lineal o dos/tres zonas abiertas explican mejor.
- Las formulas importantes deben renderizarse como assets LaTeX limpios en `presentacion/powerpoint/assets/latex/slide_XX_v2/`, preferiblemente en PNG y SVG, con manifest. No depender de texto plano para simbolos como `m(x)`, `k(x,x')`, `K_X`, `\mathbf{f}_X` si la consistencia visual importa.
- Cada formula debe cumplir una funcion narrativa. Si todas las ecuaciones son importantes, ordenarlas como derivacion o definicion progresiva, no como escaparate.
- En slides de GP, mantener la cadena conceptual:
  - pesos gaussianos -> valores de funcion gaussianos;
  - marginalizar pesos -> matriz de covarianzas;
  - kernel -> definicion directa de covarianzas;
  - GP -> distribucion sobre funciones definida por media y kernel.
- Para la definicion de GP, el visual recomendado es un prior sin datos observados: curvas finas para funciones plausibles, curva teal para media `m(x)` y banda naranja suave para variabilidad inducida por `k(x,x')`.
- No mostrar puntos observados en una slide de prior GP; reservarlos para condicionamiento, posterior predictiva o adquisicion.
- En formulas con vectores columna, revisar el render a tamano final: los elementos como `f(x_1)`, `\vdots`, `f(x_n)` no deben solaparse ni tocar titulos o separadores.
- Si una ecuacion compuesta queda mas clara como un unico asset, crearla asi. Ejemplo disponible: `assets/latex/slide_07_v2/finite_vector_gaussian.png` para `\mathbf{f}_X=[f(x_1),\ldots,f(x_n)]^\top\sim\mathcal{N}(\mathbf{m}_X,K_X)`.
- La frase final debe funcionar como puente a la siguiente slide, no como resumen generico. En GP/kernel: destacar que la forma de `k(x,x')` determina que funciones son plausibles.

## Estilo de texto

- Texto justo. Preferir frases nominales y verbos directos.
- Evitar parrafos largos; dividir en bloques de 1-2 lineas.
- En bullets, resaltar el concepto inicial en semibold y dejar la explicacion en gris.
- Mantener tono academico claro, no comercial.
- Cada diapositiva debe tener una frase de takeaway visible o deducible.

## Patrones reutilizables

- Portada centrada.
- Slide de problema aplicada: etiqueta pequena + titulo grande + frase breve arriba; debajo, flujo horizontal de 4 tarjetas.
- Flujo de 4 tarjetas: tres tarjetas claras con borde fino y una tarjeta final verde petroleo como pregunta/conclusion.
- Tarjeta de muestra: numero grande en verde petroleo, metadatos debajo y matriz de puntos de colores como mini-visual.
- Tarjeta de informacion: lista de variables con bullets grandes; usar subtitulos pequenos en gris.
- Tarjeta de respuestas: lista vertical con puntos de color y flechas de direccion alineadas a la derecha.
- Tarjeta de decision: fondo verde petroleo, texto blanco, pregunta grande en 3 lineas y una linea divisoria fina antes de la explicacion.
- Slide de modelo sustituto: definicion breve + flujo de formulas arriba + dos columnas explicativas debajo + takeaway final en franja inferior.
- Slide de proceso: tarjetas o formulas conectadas horizontalmente.
- Slide de comparacion: dos columnas separadas por linea vertical fina.
- Slide de decision: tarjeta destacada verde petroleo a la derecha con pregunta o conclusion.
- Slide matematica: formula grande arriba o centro, explicacion en bullets escuetos debajo.

## Evitar

- Fondos blancos puros a pantalla completa.
- Gradientes, sombras fuertes, iconos decorativos o exceso de colores.
- Bloques densos de texto.
- Graficas con muchos ejes, leyendas o etiquetas pequenas.
- Diapositivas sin jerarquia clara.
- Mezclar demasiados estilos de tarjeta en una misma slide.

## Regla practica para nuevas diapositivas

Antes de crear una slide, definir:

1. Idea unica de la diapositiva.
2. Mensaje principal en una frase.
3. Elemento visual central: tarjeta, flujo, comparacion, formula o grafica.
4. Color narrativo: verde petroleo para datos/problema, naranja para modelo/coste, verde medio para decision/propuesta.

Si una diapositiva necesita mas de 3 bloques principales, probablemente debe dividirse en dos.

## Metodo operativo recomendado para agentes disenadores

Este bloque resume el criterio que mejor ha funcionado en las diapositivas v2. Debe tener prioridad cuando una nueva slide matematica, de GP, SBO o modelos sustitutos tenga demasiadas ideas posibles.

### Principio de diseno

- Cada diapositiva debe defender una unica idea oralmente defendible. Si el contenido pide "todo", seleccionar la idea que hace avanzar la narracion y dejar el resto para la siguiente slide.
- No copiar layouts anteriores por inercia. Primero entender la funcion narrativa de la diapositiva y despues elegir la estructura: dos zonas, tres zonas, flujo, comparacion, formula central o figura.
- La composicion debe sentirse abierta: mucho aire, separadores finos, pocas cajas y alineaciones limpias. Una caja se usa solo si contiene una figura, una formula compuesta o un bloque que necesita marco.
- Las frases deben ser precisas y cortas. Evitar texto explicativo que el ponente puede decir oralmente; en pantalla debe quedar lo que estructura la defensa.
- El cierre inferior no debe ser un resumen generico. Debe actuar como puente narrativo hacia la siguiente parte.

### Metodo antes de construir

Antes de crear el PPTX, escribir mentalmente o en notas:

1. Idea unica: que debe recordar la audiencia.
2. Mecanismo matematico: que relacion, formula o condicionamiento sostiene la idea.
3. Visual central: que objeto visual explica mejor esa idea.
4. Color narrativo: teal para identidad/GP/datos, naranja para modelo/incertidumbre/trade-off, verde para decision/SBO.
5. Que se omite: contenido correcto pero no necesario en esta slide.

Si no se puede responder a esos cinco puntos, la slide todavia no esta lista para disenarse.

### Slides matematicas y de procesos gaussianos

- La matematica debe aparecer como mecanismo, no como escaparate. Una formula grande vale mas que tres formulas pequenas sin lectura.
- Usar LaTeX renderizado para toda expresion matematica relevante, incluso expresiones cortas como `k(x,x')`, `m(x)`, `K_X`, `f_X`, `x_*`, `f_*`, `mu(x)` o `sigma(x)` cuando formen parte de leyendas o frases clave.
- Preferir una ecuacion compuesta como unico asset cuando haya matrices, vectores columna, condicionamientos o expresiones largas. Esto evita solapes y mantiene jerarquia.
- Revisar visualmente subindices, transpuestas, estrellas, primas, barras, matrices y parentesis. Si un simbolo se ve raro en mathtext, cambiar a una notacion equivalente y limpia.
- Distinguir variables aleatorias de observaciones realizadas cuando sea matematicamente importante. Ejemplo: `Y` como variable aleatoria y `Y=y` al condicionar.
- En GP, no mezclar prior, posterior, kernel, prediccion y adquisicion en la misma slide salvo que el objetivo sea precisamente un mapa de flujo.
- En figuras de GP:
  - prior: sin puntos observados;
  - posterior: puntos observados, media posterior y banda de incertidumbre;
  - adquisicion: separar prediccion/incertidumbre de criterio de decision.
- Una figura debe calcularse con matplotlib u otro metodo correcto, no dibujarse a ojo si representa un fenomeno probabilistico.

### Patrones que han funcionado en v2

- Slide 5 v2: derivacion abierta a la izquierda y visual pequena a la derecha. Buena para explicar un puente conceptual.
- Slide 6 v2: tres zonas abiertas con separadores verticales. Buena para una transicion secuencial de representaciones.
- Slide 7 v2: definicion formal + definicion finita + visual de prior. Buena para formalizar sin perder intuicion.
- Slide 8 v2: definicion central + ejemplo RBF + hiperparametros. Buena para convertir una definicion en una hipotesis interpretable.
- Slide 9 v2: catalogo compacto + comparacion visual calculada. Buena para mostrar familias sin hacer una lista plana.
- Slide 10 v2: variables -> conjunta -> condicionamiento -> puente a decision. Buena para explicar un mecanismo matematico principal.

Estos patrones son referencias, no plantillas. Si una slide nueva pide otra organizacion, el disenador debe cambiarla.

### Figuras y graficos

- Las figuras deben ser pedagogicas y matematicamente honestas. Si una curva representa un posterior GP, calcular el posterior; si representa un kernel, graficar la funcion correcta.
- Evitar ejes pesados, muchas marcas, leyendas grandes o anotaciones redundantes. En esta presentacion las graficas deben parecer integradas en una slide academica, no exportes de paper.
- Mantener fondo transparente o gris claro, lineas finas, texto minimo y colores de la paleta.
- Usar bandas suaves para incertidumbre: naranja con baja opacidad funciona bien; media en teal; puntos observados en texto oscuro con borde claro.
- Las figuras pueden ir dentro de un marco gris muy claro con borde fino si necesitan contenerse visualmente. No convertir cada elemento en tarjeta.

### Workflow tecnico recomendado

- Crear cada slide como PPTX independiente con nombre `diapositiva_XX_v2.pptx`.
- Crear workspace temporal en `tmp/presentations/slide_XX_v2/`.
- Guardar notas en `.txt`: `slide-plan.txt`, `source-notes.txt`, `qa/visual-qa.txt`.
- Generar formulas y figuras en `presentacion/powerpoint/assets/latex/slide_XX_v2/`, con PNG, SVG y `manifest.txt`.
- Construir el PPTX con `@oai/artifact-tool`, no con mutaciones OOXML manuales.
- Renderizar siempre a PNG antes de entregar.
- Mirar el render completo, no solo asumir que el script funciono.
- Si algo se ve denso, pequeno, solapado o dificil de defender oralmente, simplificar y recompilar.

### QA visual obligatoria

Antes de entregar, comprobar:

- Slide count correcto: normalmente 1.
- Numero de pagina correcto.
- Fondo `#E8E8E8`.
- Jerarquia de titulo, subtitulo, cuerpo y cierre.
- No hay texto cortado, solapado ni demasiado pequeno.
- Las formulas se leen a tamano de defensa.
- Los simbolos matematicos son correctos y consistentes.
- Las figuras tienen funcion narrativa y no son decorativas.
- El color naranja no domina salvo que la idea central sea incertidumbre, coste o trade-off.
- El cierre conecta con la siguiente parte del discurso.

### Criterio final de aceptacion

Una slide esta terminada solo si se puede explicar oralmente en 60-90 segundos siguiendo la mirada natural de la composicion. Si el ponente necesita pedir disculpas por la densidad, la slide debe simplificarse.
