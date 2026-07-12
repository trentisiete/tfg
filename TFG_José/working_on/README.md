# Guía de trabajo para cambios del TFG

Este archivo es la guía operativa para trabajar sobre el TFG. Debe leerse antes de proponer cualquier cambio. Su función es convertir las indicaciones generales y las futuras anotaciones de la profesora en reglas prácticas de edición.

## Objetivo

Ayudar a reescribir o ajustar partes concretas del TFG sin modificar directamente los archivos finales, salvo que se indique explícitamente lo contrario. El trabajo debe producir propuestas claras, localizadas y listas para trasladarse a LaTeX.

## Material de referencia

- `TFG_José/chapters`, `TFG_José/frontmatter`, `TFG_José/appendices`, `TFG_José/config` y `TFG_José/main.tex` contienen el texto real del TFG. Por defecto, no se modifican directamente.
- `TFG_José/markdown` contiene versiones en Markdown pensadas para leer y buscar contexto con más facilidad.
- `TFG_José/working_on` es la zona de trabajo. Aquí se escriben las propuestas, sustituciones y notas operativas.

## Reglas primitivas

1. Revisar siempre el texto existente antes de escribir una propuesta.
2. Mantener el estilo académico y la forma de argumentar que ya tiene el TFG.
3. Evitar repeticiones. Si una idea ya aparece en otra parte, decidir si conviene referenciarla, condensarla o sustituir el fragmento afectado.
4. Trabajar con poco espacio: normalmente los cambios deben ser sustituciones precisas, no añadidos largos.
5. Que el texto sea breve no significa que deba ser superficial. Las propuestas tienen que explicar bien la idea, con rigor y sin frases genéricas.
6. No inventar contexto. Si la información necesaria no se encuentra en los archivos del TFG, se debe preguntar antes de escribir.
7. Antes de cambiar una zona, revisar qué otras partes del documento se relacionan con esa idea y explicarlo fuera del texto propuesto.
8. Las salidas destinadas al TFG deben estar en LaTeX y seguir buenas prácticas de redacción y formato LaTeX.
9. Las anotaciones de la profesora deben transformarse en criterios reutilizables. No basta con resolver el caso puntual: hay que registrar aquí el aprendizaje general cuando sea útil.

## Flujo de trabajo

1. Leer este README.
2. Identificar la anotación o el fragmento que se quiere corregir.
3. Buscar contexto en el LaTeX real y, si ayuda, en la versión Markdown.
4. Revisar secciones relacionadas para detectar repeticiones, contradicciones o dependencias.
5. Preparar una propuesta en `TFG_José/working_on`, sin tocar los archivos finales por defecto.
6. Indicar con precisión desde dónde hasta dónde debe aplicarse el cambio.
7. Incluir el nuevo texto en LaTeX.
8. Explicar aparte qué relaciones con otras partes del TFG se han tenido en cuenta.

## Formato recomendado para cada propuesta

Cada propuesta de cambio debería incluir:

- **Anotación o motivo:** qué se está intentando corregir.
- **Ubicación:** archivo, sección y fragmento afectado.
- **Rango de sustitución:** texto inicial y texto final que delimitan el cambio.
- **Relación con otras partes:** secciones o ideas conectadas que conviene vigilar.
- **Texto propuesto:** versión final en LaTeX, lista para copiar al documento.
- **Notas fuera del LaTeX:** supuestos o información que falta.

## Criterios de estilo

- Priorizar claridad, precisión y continuidad con el texto ya escrito.
- Usar terminología de forma consistente.
- No introducir explicaciones paralelas si el capítulo ya las desarrolla en otro lugar.
- Evitar afirmaciones demasiado generales si no están apoyadas por el contexto del TFG.
- Cuidar las transiciones: el cambio debe encajar con el párrafo anterior y el siguiente.
- Escribir lo necesario y no más. Si una idea puede quedar clara con menos texto, se prefiere la versión breve.

## Aprendizajes de las anotaciones de la profesora

- Separar el protocolo experimental de la herramienta de implementación. Primero deben quedar claros los pasos replicables: datos, particiones, preprocesamiento, modelos, métricas y presupuesto. Después puede explicarse el entorno técnico: lenguaje, paquetes, clases propias y equipo de ejecución.
- Definir explícitamente las líneas base. Si se usa un modelo Dummy, hay que indicar qué estadístico predice, con qué datos se calcula y qué papel tiene en la comparación.
- Evitar repetir introducciones ya dadas. Cuando una idea se ha adelantado antes, usar una transición breve, por ejemplo "Como se ha adelantado", y avanzar hacia el detalle nuevo.
- Toda tabla o figura debe estar referenciada en el texto y acompañada de una frase que explique qué aporta. No basta con incluir `\caption` y `\label`.
- No incluir dudas, pendientes ni indicaciones de trabajo dentro del texto LaTeX propuesto. Si falta información, se resuelve antes o se deja en notas fuera del bloque LaTeX.
- No trasladar instrucciones internas del usuario ni partes del prompt al texto propuesto.
- Si la plantilla denomina las tablas como "Cuadro", las referencias textuales deben usar "Cuadro" para mantener coherencia con el documento final.
- No presentar como ganador claro un resultado que en realidad está empatado o tiene diferencias pequeñas. En esos casos, hablar de familias consistentes, empates prácticos o tendencias.
- Cuando un cuadro jerarquiza factores, añadir una lectura práctica: en qué debe centrarse el lector y qué factores quedan como secundarios.
- Evitar nombres internos de archivos y columnas en la narración principal si no ayudan al lector. Describir primero el dato real: especie, dieta, réplica, variable medida o criterio de agrupación.
- Si se usan etiquetas abreviadas de dietas en resultados, explicar una vez su regla de lectura y reutilizarlas después de forma consistente.
- No proponer reescrituras de una zona que la profesora ya haya validado positivamente, salvo que una anotación posterior pida modificarla de forma explícita.
- Cuando una tabla define familias de modelos usadas en más de un experimento, ubicarla en el marco común y referenciarla desde cada experimento. Las secciones específicas deben explicar solo lo que cambia en su protocolo.
- Si una tabla del marco común incluye elementos que no aparecen en todos los experimentos, añadir explícitamente el alcance de uso. No presentar como común una familia que solo se emplea en un caso.
- Evitar sufijos como `NoARD` en la narración principal y en cuadros explicativos. Si no hay ARD, se omite; si hay ARD, se menciona porque cambia la interpretación del modelo.
- No introducir por primera vez la fórmula o estructura de un modelo dentro de resultados. Los resultados deben interpretar modelos ya definidos en la metodología o en el protocolo del experimento correspondiente.
- Cuando se use una métrica técnica, acompañarla con su lectura práctica. Por ejemplo, en LODO, "MAE macro" debe explicarse como error medio por dieta. Evitar explicar métricas alternativas si no se usan después en resultados.
- En figuras que mezclan color normalizado y números en unidades originales, la leyenda y el pie deben explicar ambas escalas. No llamar MAE a una barra de color si representa error relativo o normalizado.
- No presentar FCR como porcentaje. FCR es una razón de conversión; quitina y proteína sí se expresan en porcentaje.
- En revisiones de espacio, eliminar primero duplicados literales, etiquetas repetidas y referencias duplicadas. Después compactar cierres o cautelas repetidas, pero conservar las explicaciones que sostienen el protocolo experimental.
