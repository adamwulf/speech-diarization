// Streaming sample-rate conversion: a windowed-sinc (Blackman) polyphase
// resampler that takes its input in blocks, so decoded audio can be converted
// to 16 kHz as it arrives instead of holding the full-rate signal in memory.

/** Sinc zero crossings on each side of the kernel, at the lower of the two rates. */
const ZERO_CROSSINGS = 16;
/** Pass-band edge, as a fraction of the lower rate's Nyquist frequency. */
const CUTOFF = 0.9;

function gcd(a, b) {
  while (b) [a, b] = [b, a % b];
  return a;
}

function sinc(x) {
  if (x === 0) return 1;
  const px = Math.PI * x;
  return Math.sin(px) / px;
}

/**
 * Converts mono PCM from `inputRate` to `outputRate` (both whole Hz).
 * push() returns the output that its input completes; flush() returns the
 * rest once the input has ended. For N input samples the total output is
 * ceil(N × outputRate / inputRate) samples, whatever the block sizes.
 * Equal rates pass the input through unchanged.
 */
export class StreamingResampler {
  #passthrough;
  #up; // output samples per `#down` input samples, in lowest terms
  #down;
  #half; // kernel taps on each side of the centre tap
  #taps;
  #kernels; // one normalized row of #taps coefficients per phase
  #history; // input samples from absolute index #historyStart onward
  #historyStart;
  #inputLength = 0;
  #outputLength = 0;
  #base = 0; // integer part of the next output's input position
  #phase = 0; // fractional part of that position, in steps of 1 / #up
  #flushed = false;

  constructor(inputRate, outputRate) {
    for (const rate of [inputRate, outputRate]) {
      if (!Number.isInteger(rate) || rate <= 0) throw new RangeError(`Invalid sample rate: ${rate}`);
    }
    this.#passthrough = inputRate === outputRate;
    if (this.#passthrough) return;
    const divisor = gcd(inputRate, outputRate);
    this.#up = outputRate / divisor;
    this.#down = inputRate / divisor;

    // The low-pass cutoff, as a fraction of the input Nyquist frequency.
    const cutoff = CUTOFF * Math.min(1, outputRate / inputRate);
    const width = ZERO_CROSSINGS / cutoff; // kernel half-width, in input samples
    this.#half = Math.ceil(width) + 1;
    this.#taps = 2 * this.#half + 1;
    this.#kernels = new Float32Array(this.#up * this.#taps);
    for (let phase = 0; phase < this.#up; phase++) {
      const row = phase * this.#taps;
      let sum = 0;
      for (let tap = 0; tap < this.#taps; tap++) {
        // Distance from the output position to input sample (base - half + tap).
        const distance = tap - this.#half - phase / this.#up;
        const window = Math.abs(distance) < width
          ? 0.42 + 0.5 * Math.cos((Math.PI * distance) / width) + 0.08 * Math.cos((2 * Math.PI * distance) / width)
          : 0;
        const value = cutoff * sinc(cutoff * distance) * window;
        this.#kernels[row + tap] = value;
        sum += value;
      }
      // Unity gain at DC for every phase.
      for (let tap = 0; tap < this.#taps; tap++) this.#kernels[row + tap] /= sum;
    }

    // Silence before the first sample, so the first outputs have full kernels.
    this.#history = new Float32Array(this.#half);
    this.#historyStart = -this.#half;
  }

  /** Adds input samples and returns every output sample they complete. */
  push(input) {
    if (this.#flushed) throw new Error('StreamingResampler: push() after flush()');
    if (this.#passthrough) return input.slice();
    this.#append(input);
    this.#inputLength += input.length;
    // An output needs #half input samples after its position.
    return this.#render(this.#inputLength - this.#half);
  }

  /** Ends the input and returns the remaining output samples. */
  flush() {
    if (this.#flushed) return new Float32Array(0);
    this.#flushed = true;
    if (this.#passthrough) return new Float32Array(0);
    // Silence after the last sample, so the last outputs have full kernels.
    this.#append(new Float32Array(this.#half + 1));
    return this.#render(this.#inputLength);
  }

  #append(input) {
    // Keep only the history that the next output still needs.
    const keepFrom = Math.max(this.#historyStart, this.#base - this.#half);
    const kept = this.#history.subarray(keepFrom - this.#historyStart);
    const next = new Float32Array(kept.length + input.length);
    next.set(kept);
    next.set(input, kept.length);
    this.#history = next;
    this.#historyStart = keepFrom;
  }

  /** Renders every output whose input position is before `limit`. */
  #render(limit) {
    const up = this.#up;
    const down = this.#down;
    // Output n sits at input position n × down / up.
    const total = limit > 0 ? Math.ceil((limit * up) / down) : 0;
    const count = Math.max(0, total - this.#outputLength);
    const output = new Float32Array(count);
    const kernels = this.#kernels;
    const taps = this.#taps;
    const history = this.#history;
    const offset = -this.#half - this.#historyStart;
    let base = this.#base;
    let phase = this.#phase;
    for (let i = 0; i < count; i++) {
      const first = base + offset;
      const row = phase * taps;
      let sum = 0;
      for (let tap = 0; tap < taps; tap++) sum += history[first + tap] * kernels[row + tap];
      output[i] = sum;
      phase += down;
      if (phase >= up) {
        const steps = Math.floor(phase / up);
        base += steps;
        phase -= steps * up;
      }
    }
    this.#base = base;
    this.#phase = phase;
    this.#outputLength += count;
    return output;
  }
}
