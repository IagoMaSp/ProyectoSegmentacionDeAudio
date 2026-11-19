# Design System — Segmentación de audio

## Dirección visual
- **Mundo:** Columna Estratigráfica (`seed:88ebdbb2b0e84bed`) — perfil técnico de laboratorio sobre papel marfil mate, reglas estructurales de `1px` y bloques planos de color para distinguir categorías de señal sin ruido visual.
- **Modo:** Operate (herramienta de análisis y demostración académica).
- **Lo que no se usa:** gradientes decorativos, sombras difusas, glassmorphism, esquinas redondeadas en paneles, emojis, púrpura, grillas de tarjetas, iconografía de relleno. Los paneles son rectángulos con borde de 1 px y cabecera en piedra.

## Paleta
- **Fondo (`--bg`):** `#FBFBF8` (marfil técnico)
- **Superficie (`--surface`):** `#FFFFFF`; **superficie secundaria (`--surface-muted`):** `#F3F2EC`
- **Tinta:** `--ink #171717`, `--ink-muted #52524E`, `--ink-faint #8A8983`
- **Reglas:** `--rule #171717` (ejes y bordes de panel), `--rule-soft #D6D5CE`, `--rule-faint #E8E7E0` (grillas de gráficos)
- **Categorías (relleno plano, sin modulación):** `musica #2563EB`, `voz #059669`, `ruido #D97706`, `silencio #94A3B8`
- **Estados:** error `#B91C1C` sobre `#FEF2F2`; fila en reproducción `#EFF6FF`; fila abierta con regla izquierda de 3 px en tinta.
- **Espectrogramas y matrices de confusión:** escala de grises de tinta sobre marfil (de `#FBFBF8` a `#171717`). Es una escala de datos, no un degradado decorativo.

## Tipografía y escala
- **UI y títulos:** `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`
- **Datos, ejes, parámetros y leyendas de figuras:** `ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace`
- **Numerales:** `font-variant-numeric: tabular-nums` en tiempos, duraciones, porcentajes y tablas; coma decimal.
- **Rótulos de sección:** versalitas simuladas (mayúsculas, `0.75rem`, `letter-spacing .06em`, tinta secundaria).

## Componentes
- **Pestañas:** texto con subrayado de 3 px en tinta para la activa; sin fondo.
- **Panel:** borde de 1 px, cabecera `--surface-muted` con regla inferior; columnas internas separadas por reglas de 1 px, no por espacios.
- **Línea de tiempo:** regla graduada, forma de onda, barra estratigráfica de segmentos (botones planos), banda de probabilidad por celda apilada, playhead de 2 px con marca triangular.
- **Barras de probabilidad:** pista piedra, relleno del color de clase, marca vertical de tinta para el valor crudo cuando se compara con el suavizado.
- **Gráficos:** canvas propio (`web/app.js`), ejes en tinta, grilla en `--rule-faint`, series en tinta o color de clase, sin leyendas flotantes.
- **Botones:** primario relleno tinta; secundario fantasma con borde `--rule-soft` que pasa a tinta en hover; estado presionado invertido.

## Movimiento y superficies del navegador
- Transiciones sobrias (`100–200 ms ease`) en hover, selección y color de la etiqueta en vivo.
- Sin animaciones de fondo; el único indicador animado es el spinner del botón de procesar.
- `::selection`, scrollbar y `:focus-visible` (`2px solid #171717`) alineados a la paleta.

## Archivos
- `web/index.html` estructura, `web/style.css` tokens y componentes, `web/app.js` lógica y gráficos. Sin CDN ni fuentes externas.
