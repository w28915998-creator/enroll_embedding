#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
from scipy.spatial.distance import cosine
from speechbrain.pretrained import EncoderClassifier


# In[ ]:


# -----------------------------
# CONFIG
# -----------------------------
JSON_FILE = "enrolled_users.json"  # store embeddings here
MIN_DURATION = 10  # minimum recording time in seconds
SILENCE_THRESHOLD = 0.1  # RMS threshold to detect silence
SILENCE_DURATION = 2  # seconds of silence to auto-stop

# Load model once
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# -----------------------------
# Functions
# -----------------------------

def record_audio(min_duration=MIN_DURATION):
    """Record audio for at least min_duration and stop when user is silent."""
    print(f"🎙 Speak now (at least {min_duration} seconds). Recording will stop when you're silent...")

    samplerate = 16000
    channels = 1

    recording = []
    silence_time = 0
    start_time = time.time()

    stream = sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32')
    stream.start()

    try:
        while True:
            data, _ = stream.read(int(samplerate * 0.2))  # 0.2 sec chunks
            chunk = np.copy(data[:, 0])
            recording.append(chunk)

            # RMS to detect silence
            rms = np.sqrt(np.mean(chunk ** 2))
            if rms < SILENCE_THRESHOLD:
                silence_time += 0.2
            else:
                silence_time = 0

            duration = time.time() - start_time
            if duration >= min_duration and silence_time >= SILENCE_DURATION:
                break
    finally:
        stream.stop()
        stream.close()

    audio = np.concatenate(recording, axis=0)
    print(f"✅ Recorded {len(audio)/samplerate:.2f} seconds of audio.")
    return audio, samplerate

def extract_embedding(audio, samplerate):
    """Extract speaker embedding from audio array."""
    temp_path = "temp_audio.wav"
    sf.write(temp_path, audio, samplerate)

    signal, fs = torchaudio.load(temp_path)  # [channels, time]

    # Convert to mono if stereo
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    # Resample if needed
    if fs != samplerate:
        transform = torchaudio.transforms.Resample(orig_freq=fs, new_freq=samplerate)
        signal = transform(signal)

    # Now make it [batch, time]
    signal = signal.squeeze(0)            # [time]
    signal = signal.unsqueeze(0)          # [1, time] = batch size 1

    # Pass directly to encode_batch
    embedding_tensor = classifier.encode_batch(signal)
    embedding = embedding_tensor.squeeze(0).detach().cpu().numpy()

    os.remove(temp_path)
    return embedding.tolist()

def load_enrolled_data():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            return json.load(f)
    return {}

def save_enrolled_data(data):
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

def enroll_user(name):
    """Enroll a user with the given name"""
    try:
        audio, sr = record_audio(min_duration=MIN_DURATION)
        embedding = extract_embedding(audio, sr)
        enrolled_data = load_enrolled_data()
        
        # If name already exists, append to existing embeddings
        if name in enrolled_data:
            enrolled_data[name].append(embedding)
        else:
            enrolled_data[name] = [embedding]
            
        save_enrolled_data(enrolled_data)
        return True, f"User '{name}' enrolled successfully!"
    except Exception as e:
        return False, f"Error during enrollment: {str(e)}"

def check_user(threshold=0.3):
    """Check if a user is already enrolled"""
    try:
        enrolled_data = load_enrolled_data()
        if not enrolled_data:
            return False, "No users enrolled yet.", None, None

        audio, sr = record_audio(min_duration=MIN_DURATION)
        test_emb = np.array(extract_embedding(audio, sr)).flatten()

        best_name = None
        best_score = float('inf')

        for name, emb_list in enrolled_data.items():
            for emb in emb_list:
                emb_vec = np.array(emb).flatten()
                score = cosine(test_emb, emb_vec)
                if score < best_score:
                    best_score = score
                    best_name = name

        if best_score < threshold:
            return True, f"Match: {best_name} (distance={best_score:.3f})", best_name, best_score
        else:
            return False, f"No match under threshold. Closest guess: {best_name} (distance={best_score:.3f})", best_name, best_score
    except Exception as e:
        return False, f"Error during verification: {str(e)}", None, None

def get_enrolled_users():
    """Get list of all enrolled users"""
    enrolled_data = load_enrolled_data()
    return list(enrolled_data.keys())

