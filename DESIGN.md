# Design System — Segmentación de audio

## Dirección visual
- **Mundo:** Columna Estratigráfica (`seed:88ebdbb2b0e84bed`) — perfil técnico de laboratorio sobre papel marfil mate con reglas estructurales de `1px` y bloques planos de color para distinguir categorías de señal sin ruido visual.
- **Modo:** Operate (herramienta de análisis y demostración académica para la materia Muestreo).

## Paleta
- **Fondo principal (`--bg`):** `#FBFBF8` (marfil técnico mate)
- **Superficie elevada (`--surface`):** `#FFFFFF`
- **Superficie secundaria (`--surface-muted`):** `#F3F2EC`
- **Tinta principal (`--ink`):** `#171717`
- **Tinta secundaria (`--ink-muted`):** `#52524E`
- **Reglas estructurales:** `--rule: #171717` (ejes principales), `--rule-soft: #D6D5CE`, `--rule-faint: #E8E7E0`
- **Categorías de audio (relleno plano sin modulación):**
  - `musica`: `#2563EB`
  - `voz`: `#059669`
  - `ruido`: `#D97706`
  - `silencio`: `#94A3B8`
- **Estados:** Error `#B91C1C` sobre `#FEF2F2`; fila activa `#EFF6FF`.

## Tipografía y escala
- **UI y títulos:** `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`
- **Especificaciones técnicas y regla temporal:** `ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace`
- **Numerales:** `font-variant-numeric: tabular-nums` en todas las marcas de tiempo, duraciones, porcentajes y tamaños de archivo.

## Movimiento y superficies del navegador
- Transiciones rápidas y sobrias (`100ms–140ms ease`) en hover/selección.
- Sin animaciones infinitas de fondo; el indicador de carga usa un spinner compacto en el botón de acción y un riel de estado estático.
- `::selection`, scrollbar y `:focus-visible` (`2px solid #171717`) alineados a la paleta.
