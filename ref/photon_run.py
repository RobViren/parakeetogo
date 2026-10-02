import sys, time, wave
import moondream as md

with md.photon("moondream/parakeet-redux", device="cpu") as speech:
    speech.transcribe(audio=sys.argv[1])
    for path in sys.argv[1:]:
        w = wave.open(path)
        t = time.time()
        text = speech.transcribe(audio=path)["text"]
        print(f"{path}\t{w.getnframes() / w.getframerate():.2f}\t{time.time() - t:.3f}\t{text}", flush=True)
