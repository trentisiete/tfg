# Índice operativo para cerrar anexos del TFG

Este archivo guarda el plan de trabajo acordado para reforzar los anexos sin tocar el cuerpo del TFG salvo hallazgo crítico. La prioridad es defender el tema real del trabajo: modelos sustitutos probabilísticos, procesos gaussianos, búsqueda de entradas óptimas y uso responsable de incertidumbre con datos escasos.

## Criterio general

- El cuerpo del TFG se considera congelado.
- Solo se tocaría el cuerpo si aparece un error metodológico o matemático que cambie la interpretación del trabajo.
- Los anexos deben aportar trazabilidad, configuración experimental, reproducibilidad y defensa técnica.
- Las propuestas se redactarán completas en LaTeX dentro de `TFG_José/working_on`; el alumno decidirá qué incorpora al documento real.

## Diagnóstico de anexos existentes

### Anexo A: material complementario de benchmarks

Ya encaja aquí:

- dominio detallado de Borehole;
- flujo del proceso de infill;
- semillas;
- figuras de mejora del incumbent, MAE, diagnóstico probabilístico y efecto de condiciones.

Posibles añadidos si se decide ampliar A:

- tabla compacta de configuración benchmark desde el código;
- explicación de `n_train = {1, d, 4d}`;
- resumen de EI: `xi`, presupuesto del optimizador, Differential Evolution y fallback;
- aclaración de que los grids de tuning de benchmark existen en código, pero la lectura principal del TFG se basa en las familias GP comparadas y en el ciclo activo.

### Anexo B: visualización complementaria del infill

Debe mantenerse como anexo visual. No necesita más metodología.

Prioridad:

- corregir ortografía y tildes;
- no añadir más contenido salvo que sobre espacio y se quiera reforzar Forrester como ejemplo pedagógico.

### Anexo C: material complementario del caso real

Ya encaja aquí:

- inventario de datasets generados;
- valores observados por dieta;
- paridad del mejor GP;
- intervalos predictivos;
- paisaje prospectivo de EI.

Posibles añadidos si se decide ampliar C:

- tabla de variables usadas en modo reducido y completo;
- explicación corta de LODO externo/interno centrada en fuga de información;
- nota de limitaciones del caso real: 11 dietas, 3 réplicas, EI prospectivo.

### Nuevo Anexo D recomendado: configuración computacional y reproducibilidad

Este es el añadido principal recomendado. Evita sobrecargar A y C con detalles de código y responde directamente a preguntas de tribunal sobre:

- qué scripts generan qué resultados;
- qué modelos y grids se usaron;
- cómo se evita fuga de información;
- qué outputs sostienen las figuras;
- cómo regenerar el caso real.

Archivo de borrador completo:

- `TFG_José/working_on/anexo_D_configuracion_reproducibilidad.tex`

## Orden de trabajo

1. Crear Anexo D completo en LaTeX.
2. Revisar si el Anexo D duplica demasiado contenido de A o C.
3. Preparar, si hace falta, una inserción breve para Anexo A con configuración benchmark.
4. Preparar, si hace falta, una inserción breve para Anexo C con variables y fuga de información.
5. Corregir ortografía de anexos B y C.
6. Revisar `main.tex` si se decide añadir un nuevo `\include{appendices/D_configuracion_reproducibilidad}`.
7. Compilar PDF y revisar referencias, lista de figuras y lista de tablas.

## Decisión editorial recomendada

La opción más limpia es añadir un nuevo Anexo D. Así:

- A sigue centrado en benchmarks;
- B sigue siendo visual;
- C sigue centrado en caso real;
- D concentra código, grids, reproducibilidad y defensa técnica.

Si el documento final no admite un nuevo anexo por extensión total o estética, el contenido de D puede dividirse:

- configuración benchmark hacia A;
- configuración LODO y outputs del caso real hacia C;
- tabla de scripts como una sección final de C o como apéndice técnico mínimo.

