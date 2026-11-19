// AudioWorklet del modo en vivo: reenvia cada bloque de 128 muestras del microfono.
class Captura extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("captura", Captura);
