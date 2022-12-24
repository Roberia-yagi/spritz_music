import whisperx

device = "cuda:0"
audio_file = "./materials/music.mp3"

# transcribe with original whisper
model = whisperx.load_model("tiny", device)
result = model.transcribe(audio_file, fp16=False)

# load alignment model and metadata
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

# align whisper output
result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file, device)

print(result["segments"]) # before alignment

print(result_aligned["segments"]) # after alignment
print(result_aligned["word_segments"]) # after alignment