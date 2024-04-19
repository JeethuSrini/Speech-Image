from ct2_utils import CheckQuantizationSupport
from Modify_whisper import VoiceRecorder  # Assuming you've defined this class in a file named voice_recorder.py
import requests
import io
from PIL import Image


if __name__ == "__main__":
    quantization_checker = CheckQuantizationSupport()
    cuda_available = quantization_checker.has_cuda_device()
    quantization_checker.update_supported_quantizations()

    recorder = VoiceRecorder()
    recorder.start_recording()  # Start recording
    input("Press Enter to stop recording...")  # Wait for user input to stop recording
    recorder.save_audio() 
    

