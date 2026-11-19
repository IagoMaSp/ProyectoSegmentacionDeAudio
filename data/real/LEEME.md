# Grabaciones propias para la evaluación

Esta carpeta es para probar el sistema con audio que no salió de MUSAN, grabado
con el celular en condiciones reales. Los audios no se suben al repositorio (están
en `.gitignore`); solo este archivo y el CSV de ejemplo.

## Qué grabar

De 20 a 30 s por archivo, con la app de grabación del celular (Notas de voz en
iPhone, Grabadora en Android). Cualquier formato sirve: m4a, mp3, amr, ogg, wav.

| Archivo sugerido        | Qué grabar                                                                 |
| :---------------------- | :------------------------------------------------------------------------- |
| `voz_01.m4a`            | Hablar normal, a 30–50 cm del celular, en la pieza.                         |
| `voz_02.m4a`            | Otra persona, o leer en voz alta con la TV apagada.                         |
| `voz_03.m4a`            | Hablar con ruido de fondo (ventilador, calle por la ventana).               |
| `musica_01.m4a`         | Música por un parlante, con voz cantada.                                    |
| `musica_02.m4a`         | Música instrumental (sin voz).                                              |
| `ruido_01.m4a`          | Ruido ambiente fuerte: canilla abierta, secador, ventilador al máximo.      |
| `ruido_02.m4a`          | Calle o cocina (platos, campana).                                           |
| `silencio_01.m4a`       | La pieza en silencio, sin hablar ni moverse.                                |
| `silencio_02.m4a`       | Otra habitación en silencio, o de noche.                                    |
| `mezcla_01.m4a`         | Secuencia con varias clases (ver CSV abajo): 10 s silencio, 10 s voz, 10 s música, 10 s ruido. |

Tener en cuenta:

- No tocar el celular mientras graba: los golpes al micrófono son ruido, no silencio.
- Si la app tiene "reducción de ruido" o "mejorar voz", desactivarla.
- Para la mezcla conviene anotar los tiempos en el momento (cronómetro).

## Etiquetas

- **Por nombre:** si el archivo es de una sola clase, alcanza con que el nombre
  empiece por la clase: `voz_`, `musica_`, `ruido_` o `silencio_`.
- **Por intervalos:** para archivos con varias clases, crear `etiquetas.csv` con
  columnas `archivo,inicio,fin,clase` (segundos). Ver `etiquetas_ejemplo.csv`.
  Si un archivo está en el CSV, el CSV manda sobre el nombre.

## Correr la evaluación

```bash
.venv/bin/python -m scripts.eval_real
```

Imprime la línea de tiempo de cada archivo y deja las métricas en
`docs/resultados/eval_real.json` (se ven también en la pestaña Métricas de la
interfaz) y la figura `docs/figuras/real_lineas.png`.
