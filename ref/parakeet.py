import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

torch.set_num_threads(4)
torch.set_grad_enabled(False)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
SAMPLE_RATE, HOP, WIN, N_FFT, N_MELS = 16000, 160, 400, 512, 128
PREEMPHASIS, LOG_GUARD, NORM_EPS = 0.97, 2.0**-24, 1e-5
LAYERS, HIDDEN, HEADS, HEAD_DIM, CONV_KERNEL = 24, 1024, 8, 128, 9
SUB_CHANNELS, GROUP = 256, 128
VOCAB, BLANK, PAD, DURATIONS = 8193, 8192, 2, (0, 1, 2, 3, 4)
STEPS_PER_FRAME, MAX_TOKENS = 10, 4096
FRAME_SECONDS = 0.08
VAD_THRESHOLD, VAD_MIN_SPEECH, VAD_MIN_GAP = 0.5, 0.1, 0.1
SEGMENT_SECONDS, MIN_SEGMENT_SECONDS, MIN_PAUSE, PAUSE_EPS = 30.0, 1.0, 0.2, 1e-6
BLOCK_SAMPLES, MIN_TAIL_SAMPLES, MIN_SAMPLES = 120 * SAMPLE_RATE, SAMPLE_RATE // 2, 320
INT8_ACTIVATIONS = False


def load_wav(path):
    with wave.open(str(path)) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0)


def hann_window():
    n = np.arange(WIN, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (WIN - 1))
    pad = (N_FFT - WIN) // 2
    return torch.from_numpy(np.pad(window, (pad, pad)).astype(np.float32))


def mel_filterbank():
    def hz_to_mel(hz):
        return 15 + np.log(hz / 1000) * (27 / np.log(6.4)) if hz >= 1000 else 3 * hz / 200

    def mel_to_hz(mel):
        return np.where(mel >= 15, 1000 * np.exp((mel - 15) * (np.log(6.4) / 27)), 200 * mel / 3)

    edges = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2))
    bins = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    slopes = edges[None] - bins[:, None]
    delta = np.diff(edges)
    bank = np.maximum(0, np.minimum(-slopes[:, :-2] / delta[:-1], slopes[:, 2:] / delta[1:]))
    bank *= 2 / (edges[2:] - edges[:-2])[None]
    return torch.from_numpy(bank.astype(np.float32))


WINDOW, MEL_BANK = hann_window(), mel_filterbank()


def log_mel(pcm):
    """[samples] -> ([samples // HOP + 1, N_MELS], valid frames). The trailing frame is zeroed, not dropped."""
    x = torch.cat((pcm[:1], pcm[1:] - PREEMPHASIS * pcm[:-1]))
    frames = F.pad(x, (N_FFT // 2, N_FFT // 2)).unfold(0, N_FFT, HOP)
    spectrum = torch.fft.rfft(frames * WINDOW)
    power = spectrum.real.square() + spectrum.imag.square()
    mel = torch.log(power @ MEL_BANK + LOG_GUARD)
    valid = len(pcm) // HOP
    mean = mel[:valid].mean(0)
    std = mel[:valid].std(0)
    mel = (mel - mean) / (std + NORM_EPS)
    mel[valid:] = 0
    return mel, valid


def unpack_ternary(qweight, scales, in_features):
    digits = (torch.arange(256)[:, None] // torch.tensor([1, 3, 9, 27, 81])) % 3
    codes = digits[qweight.long()].reshape(qweight.shape[0], -1)[:, :in_features] - 1
    return codes.float() * scales.float().repeat_interleave(GROUP, dim=1)


def load_weights(model_dir):
    raw = load_file(str(model_dir / "model.safetensors"))
    manifest = json.loads((model_dir / "ternary.json").read_text())
    w = {}
    for module in manifest["quantized_modules"]:
        name = module["name"]
        w[name + ".weight"] = unpack_ternary(
            raw.pop(name + ".qweight"), raw.pop(name + ".scales"), module["in_features"]
        )
    for name, tensor in raw.items():
        w[name] = tensor.float() if tensor.is_floating_point() else tensor
    for i in range(LAYERS):
        p = f"encoder.layers.{i}.conv."
        scale = w[p + "norm.weight"] * torch.rsqrt(w[p + "norm.running_var"] + 1e-5)
        w[p + "depthwise_conv.bias"] = w[p + "norm.bias"] - w[p + "norm.running_mean"] * scale
        w[p + "depthwise_conv.weight"] = w[p + "depthwise_conv.weight"] * scale[:, None, None]
    for i in range(2):
        p = "decoder.lstm."
        w[f"decoder.cell{i}.weight"] = torch.cat((w[f"{p}weight_ih_l{i}"], w[f"{p}weight_hh_l{i}"]), dim=1)
        w[f"decoder.cell{i}.bias"] = w[f"{p}bias_ih_l{i}"] + w[f"{p}bias_hh_l{i}"]
    for name in ("proj", "out"):
        w[f"vad_head.{name}.weight"] = w[f"vad_head.{name}.weight"][:, :, 0]
    return w


def quantize_activations(x):
    """Photon's CPU kernel: dynamic int8 per (row, 128 column group), round half even of x * (127 / max|x|)."""
    groups = x.reshape(x.shape[0], -1, GROUP)
    peak = groups.abs().amax(-1, keepdim=True)
    codes = torch.round(groups * torch.where(peak > 0, 127 / peak, 0))
    return (codes * (peak / 127)).reshape(x.shape)


def ternary_linear(x, weight):
    return (quantize_activations(x) if INT8_ACTIVATIONS else x) @ weight.T


def layer_norm(x, w, name):
    return F.layer_norm(x, (HIDDEN,), w[name + ".weight"], w[name + ".bias"], 1e-5)


def subsample(mel, valid, w):
    """[frames, N_MELS] -> ([ceil(frames / 8), HIDDEN], valid frames).

    Rows past the valid length are zeroed after each stride 2 conv only, so the pointwise bias refills them
    and the next depthwise conv reads that row at its right edge. Truncating the input instead changes the
    last valid frame.
    """
    p = "encoder.subsampling."
    x = F.conv2d(mel[None, None], w[p + "layers.0.weight"], w[p + "layers.0.bias"], stride=2, padding=1)
    valid = (valid + 1) // 2
    x[:, :, valid:] = 0
    x = F.relu(x)
    for depthwise, pointwise in ((2, 3), (5, 6)):
        x = F.conv2d(
            x, w[f"{p}layers.{depthwise}.weight"], w[f"{p}layers.{depthwise}.bias"],
            stride=2, padding=1, groups=SUB_CHANNELS,
        )
        valid = (valid + 1) // 2
        x[:, :, valid:] = 0
        x = F.relu(F.conv2d(x, w[f"{p}layers.{pointwise}.weight"], w[f"{p}layers.{pointwise}.bias"]))
    x = x[0].permute(1, 0, 2).reshape(x.shape[2], -1)
    return x @ w[p + "linear.weight"].T + w[p + "linear.bias"], valid


def vad_probabilities(hidden, w):
    x = F.silu(hidden @ w["vad_head.proj.weight"].T + w["vad_head.proj.bias"])
    x = F.conv1d(x.T[None], w["vad_head.ctx.weight"], w["vad_head.ctx.bias"], padding=2)[0].T
    x = F.silu(x) @ w["vad_head.out.weight"].T + w["vad_head.out.bias"]
    return torch.sigmoid(x[:, 0])


def relative_positions(frames):
    """[2 * frames - 1, HIDDEN]; row r encodes relative offset frames - 1 - r, sin and cos interleaved."""
    inverse = 1 / (10000 ** (torch.arange(0, HIDDEN, 2, dtype=torch.float32) / HIDDEN))
    phase = torch.outer(torch.arange(frames - 1, -frames, -1).float(), inverse)
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


def attention_mix(q, k, v, r, bias_u, bias_v):
    """q, k, v [frames, HIDDEN], r [2 * frames - 1, HIDDEN] -> [frames, HIDDEN]."""
    frames = q.shape[0]
    q, k, v, r = (t.view(-1, HEADS, HEAD_DIM).transpose(0, 1) for t in (q, k, v, r))
    content = (q + bias_u[:, None]) @ k.transpose(1, 2)
    position = (q + bias_v[:, None]) @ r.transpose(1, 2)
    index = frames - 1 - torch.arange(frames)[:, None] + torch.arange(frames)[None]
    position = position.gather(2, index.expand(HEADS, -1, -1))
    scores = (content + position) * HEAD_DIM**-0.5
    return (scores.softmax(-1) @ v).transpose(0, 1).reshape(frames, HIDDEN)


def attention(x, positions, w, p):
    q, k, v = (ternary_linear(x, w[f"{p}{name}_proj.weight"]) for name in "qkv")
    r = ternary_linear(positions, w[p + "relative_k_proj.weight"])
    mixed = attention_mix(q, k, v, r, w[p + "bias_u"], w[p + "bias_v"])
    return ternary_linear(mixed, w[p + "o_proj.weight"])


def feed_forward(x, w, p):
    return ternary_linear(F.silu(ternary_linear(x, w[p + "linear1.weight"])), w[p + "linear2.weight"])


def depthwise(x, weight, bias):
    return F.silu(F.conv1d(x.T[None], weight, bias, padding=CONV_KERNEL // 2, groups=HIDDEN))[0].T


def conv_module(x, w, p):
    value, gate = ternary_linear(x, w[p + "pointwise_conv1.weight"]).chunk(2, dim=-1)
    x = depthwise(value * torch.sigmoid(gate), w[p + "depthwise_conv.weight"], w[p + "depthwise_conv.bias"])
    return ternary_linear(x, w[p + "pointwise_conv2.weight"])


def conformer_layer(x, positions, w, p):
    x = x + 0.5 * feed_forward(layer_norm(x, w, p + "norm_feed_forward1"), w, p + "feed_forward1.")
    x = x + attention(layer_norm(x, w, p + "norm_self_att"), positions, w, p + "self_attn.")
    x = x + conv_module(layer_norm(x, w, p + "norm_conv"), w, p + "conv.")
    x = x + 0.5 * feed_forward(layer_norm(x, w, p + "norm_feed_forward2"), w, p + "feed_forward2.")
    return layer_norm(x, w, p + "norm_out")


def predictor(token, state, w):
    x = w["decoder.embedding.weight"][token]
    hidden, cell = [], []
    for i, (h, c) in enumerate(zip(*state)):
        gates = torch.cat((x, h)) @ w[f"decoder.cell{i}.weight"].T + w[f"decoder.cell{i}.bias"]
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4)
        c = torch.sigmoid(forget_gate) * c + torch.sigmoid(input_gate) * torch.tanh(candidate)
        x = torch.sigmoid(output_gate) * torch.tanh(c)
        hidden.append(x)
        cell.append(c)
    x = x @ w["decoder.decoder_projector.weight"].T + w["decoder.decoder_projector.bias"]
    return x, (hidden, cell)


def tdt_greedy(encoded, w):
    """[frames, 640] -> rows of (frame, token, duration), one per joint evaluation, blanks included.

    The step budget is STEPS_PER_FRAME * frames for the whole segment. There is no per frame symbol cap:
    a non blank token with duration 0 stays on its frame.
    """
    zeros = [torch.zeros(encoded.shape[1])] * 2
    predicted, state = predictor(BLANK, (zeros, zeros), w)
    frame, steps, tokens = 0, [], 0
    while frame < len(encoded) and len(steps) < STEPS_PER_FRAME * len(encoded) and tokens < MAX_TOKENS:
        logits = F.relu(encoded[frame] + predicted) @ w["joint.head.weight"].T + w["joint.head.bias"]
        token = int(logits[:VOCAB].argmax())
        duration = DURATIONS[int(logits[VOCAB:].argmax())]
        if token == BLANK and duration == 0:
            duration = 1
        steps.append((frame, token, duration))
        frame += duration
        if token != BLANK:
            predicted, state = predictor(token, state, w)
            tokens += 1
    return steps


def speech_regions(probabilities, seconds):
    speaking = np.r_[False, probabilities >= VAD_THRESHOLD, False]
    edges = np.flatnonzero(speaking[1:] != speaking[:-1]) * FRAME_SECONDS
    regions = []
    for start, end in edges.reshape(-1, 2):
        if regions and start - regions[-1][1] < VAD_MIN_GAP:
            regions[-1][1] = end
        else:
            regions.append([start, end])
    return [(start, min(end, seconds)) for start, end in regions if end - start >= VAD_MIN_SPEECH]


def pauses_between(regions, seconds):
    pauses, previous = [], 0.0
    for start, end in sorted(regions):
        if start - previous >= MIN_PAUSE - PAUSE_EPS:
            pauses.append((previous, start))
        previous = max(previous, end)
    if seconds - previous >= MIN_PAUSE - PAUSE_EPS:
        pauses.append((previous, seconds))
    return pauses


def next_cut(pauses):
    inside = [p for p in pauses if p[0] >= MIN_SEGMENT_SECONDS and p[1] <= SEGMENT_SECONDS]
    if not inside:
        inside = [p for p in pauses if MIN_SEGMENT_SECONDS <= (p[0] + p[1]) / 2 <= SEGMENT_SECONDS]
    return (inside[-1][0] + inside[-1][1]) / 2 if inside else SEGMENT_SECONDS


def has_speech(regions, seconds):
    return any(end > 0 and start < seconds for start, end in regions)


class Parakeet:
    def __init__(self, model_dir=MODEL_DIR, dump_dir=None):
        self.w = load_weights(Path(model_dir))
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.dump_dir = Path(dump_dir) if dump_dir else None
        if self.dump_dir:
            self.dump_dir.mkdir(parents=True, exist_ok=True)

    def dump(self, name, value):
        if self.dump_dir:
            np.save(self.dump_dir / f"{name}.npy", np.asarray(value))

    def encode(self, pcm, tag):
        mel, valid = log_mel(pcm)
        self.dump(f"{tag}.mel", mel)
        x, valid = subsample(mel, valid, self.w)
        x = x[:valid]
        self.dump(f"{tag}.subsample", x)
        positions = relative_positions(valid)
        for i in range(LAYERS):
            x = conformer_layer(x, positions, self.w, f"encoder.layers.{i}.")
            self.dump(f"{tag}.layer{i:02d}", x)
        x = x @ self.w["encoder_projector.weight"].T + self.w["encoder_projector.bias"]
        self.dump(f"{tag}.encoded", x)
        return x

    def transcribe_segment(self, pcm, tag="seg000"):
        steps = tdt_greedy(self.encode(pcm, tag), self.w)
        self.dump(f"{tag}.steps", np.array(steps, dtype=np.int32).reshape(-1, 3))
        return self.tokenizer.decode([token for _, token, _ in steps if token not in (BLANK, PAD)]).strip()

    def block_speech(self, pcm, tag):
        mel, valid = log_mel(pcm)
        hidden, valid = subsample(mel, valid, self.w)
        probabilities = vad_probabilities(hidden, self.w)[:valid].numpy()
        self.dump(f"{tag}.probabilities", probabilities)
        return speech_regions(probabilities, len(pcm) / SAMPLE_RATE)

    def segments(self, pcm):
        """Sample ranges to transcribe. Audio up to 30 s is one segment and never reaches the VAD head."""
        cap = round(SEGMENT_SECONDS * SAMPLE_RATE)
        if len(pcm) <= cap:
            return [(0, len(pcm))]
        out, regions, start, read, block_index = [], [], 0, 0, 0
        while read < len(pcm):
            remaining = len(pcm) - read
            block = remaining if remaining <= BLOCK_SAMPLES else min(BLOCK_SAMPLES, remaining - MIN_TAIL_SAMPLES)
            offset = (read - start) / SAMPLE_RATE
            speech = self.block_speech(pcm[read : read + block], f"vad{block_index:03d}")
            regions += [(a + offset, b + offset) for a, b in speech]
            read += block
            block_index += 1
            while read - start > cap:
                cut = round(next_cut(pauses_between(regions, (read - start) / SAMPLE_RATE)) * SAMPLE_RATE)
                seconds = cut / SAMPLE_RATE
                if has_speech(regions, seconds):
                    out.append((start, start + cut))
                start += cut
                regions = [(max(a - seconds, 0.0), b - seconds) for a, b in regions if b > seconds]
        if read - start >= MIN_SAMPLES and has_speech(regions, (read - start) / SAMPLE_RATE):
            out.append((start, read))
        return out

    def transcribe(self, pcm):
        segments = self.segments(pcm)
        self.dump("segments", np.array(segments, dtype=np.int64).reshape(-1, 2))
        parts = [self.transcribe_segment(pcm[a:b], f"seg{i:03d}") for i, (a, b) in enumerate(segments)]
        return " ".join(part for part in parts if part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--model", default=MODEL_DIR)
    parser.add_argument("--dump", help="directory for .npy intermediates")
    parser.add_argument("--int8", action="store_true", help="quantize ternary layer inputs the way Photon's CPU path does")
    args = parser.parse_args()
    INT8_ACTIVATIONS = args.int8
    print(Parakeet(args.model, args.dump).transcribe(load_wav(args.wav)))
