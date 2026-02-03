from faster_whisper import WhisperModel

model = WhisperModel("./models/large-v3", device="cpu", compute_type="int8")

segments, info = model.transcribe("test.wav")

for s in segments:
    print(s.text)
