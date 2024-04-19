from flask import Flask, request, jsonify
from Modify_whisper import VoiceRecorder

app = Flask(__name__)

recorder = VoiceRecorder()
@app.route('/record', methods=['POST'])
def record_and_transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    # Save the uploaded audio data to a temporary file
    audio_data = request.files['audio'].read()
    temp_filename = tempfile.mktemp(suffix=".wav")
    with open(temp_filename, 'wb') as f:
        f.write(audio_data)

    try:
        # Perform transcription using the temporary file
        text = recorder.transcribe_audio(temp_filename)
        os.remove(temp_filename)  # Clean up temporary file

        return jsonify({'text': text})  # Return the transcribed text

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle errors