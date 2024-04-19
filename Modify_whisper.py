import pyaudio
import wave
import os
import tempfile
import threading
from faster_whisper import WhisperModel
from ct2_utils import CheckQuantizationSupport  # Assuming ct2_utils.py defines this class
import yaml
import requests
import io
from PIL import Image

# Model Parameters
model_name = "small.en"
quantization_type = "int8"
device_type = "cpu"
class VoiceRecorder:
    def __init__(self, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
        self.format, self.channels, self.rate, self.chunk = format, channels, rate, chunk
        self.is_recording, self.frames = False, []
        model_str = f"ctranslate2-4you/whisper-{model_name}-ct2-{quantization_type}"
        self.model = WhisperModel(model_str, device=device_type, compute_type=quantization_type, cpu_threads=26)
        print(f"Model updated to {model_name} with {quantization_type} quantization on {device_type} device") 

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.content
    
    def img_edit(self):
        response = requests.post("https://manjushri-instruct-pix-2-pix.hf.space/run/predict", json={
	"data": [
		f"data:{self.image};base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
		"Change the background to purple",
		7.5,
		5,
		276673670061981570,
		1.5,
	]
}).json()

        data = response["data"]
        

    def txt2img(self):
        self.API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        self.headers = {"Authorization": f"Bearer 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"}
        image_bytes = self.query({
	"inputs": f"{self.text}",
})
        self.image = Image.open(io.BytesIO(image_bytes))
        self.image.show()
        self.img_edit()

    def transcribe_audio(self, audio_file):
        segments, _ = self.model.transcribe(audio_file)
        self.text = "\n".join([segment.text for segment in segments])
        print(f"Transcription:\n{self.text}")
        self.txt2img()   

    def record_audio(self):
        print("Recording...")
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
            [self.frames.append(stream.read(self.chunk)) for _ in iter(lambda: self.is_recording, False)]
            stream.stop_stream()
            stream.close()
        finally:
            p.terminate()

    def save_audio(self):
        self.is_recording = False
        temp_filename = tempfile.mktemp(suffix=".wav")
        with wave.open(temp_filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.frames))
        self.transcribe_audio(temp_filename)
        os.remove(temp_filename)
        self.frames.clear()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            threading.Thread(target=self.record_audio).start()