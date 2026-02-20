"""
A minimal FastAPI server for real-time Text-to-Speech using ZipVoice.

This server exposes a single `/tts` endpoint that accepts a JSON request
specifying a voice and text, and streams the resulting raw audio chunks
back to the client as quickly as possible.

Features:
-   Uses RealtimeTTS with the ZipVoiceEngine.
-   Loads two pre-configured voices at startup.
-   Handles one TTS request at a time to prevent resource contention.
-   Streams audio back for low-latency responses.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from RealtimeTTS import TextToAudioStream, ZipVoiceEngine, ZipVoiceVoice

# --- Configuration ---
ZIPVOICE_PROJECT_ROOT = os.getenv("ZIPVOICE_PROJECT_ROOT", "/opt/app-root/src/ZipVoice")

# IMPORTANT: Replace these with the actual paths and transcriptions for your voice prompts.
VOICE_ALPHA_WAV_PATH = os.getenv("VOICE_ALPHA_WAV_PATH", "reference1.wav")
VOICE_ALPHA_PROMPT_TEXT = os.getenv(
    "VOICE_ALPHA_PROMPT_TEXT",
    "Hi there! I'm really excited to try this out! I hope the speech sounds natural and warm - that's exactly what I'm going for!"
)


# --- Global State ---
# These will be initialized during the application lifespan startup.
engine: ZipVoiceEngine | None = None
stream: TextToAudioStream | None = None

# A semaphore to ensure only one TTS synthesis runs at a time, preventing GPU overload.
tts_semaphore = threading.Semaphore(1)


# --- Pydantic Model for the Request Body ---
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    voice: str = Field(..., min_length=1, description="The name of the voice to use (e.g., 'alpha-warm').")


# --- Core TTS Processing Logic (runs in a thread) ---
def process_tts_request(
    text_to_speak: str,
    selected_voice: ZipVoiceVoice,
    audio_queue: queue.Queue
):
    """
    Handles the actual TTS synthesis in a separate thread.
    """
    global engine, stream
    try:
        print(f"Thread started for voice '{selected_voice.prompt_wav_path}'. Synthesizing text...")

        # 1. Set the voice on the shared engine instance for this specific request.
        engine.set_voice(selected_voice)

        # 2. Define a callback to push audio chunks into the queue.
        def on_audio_chunk(chunk: bytes):
            audio_queue.put(chunk)

        # 3. Feed the text to the stream and start playing (synthesis).
        #    The `on_audio_chunk` callback will be fired for each chunk.
        stream.feed(text_to_speak)
        stream.play(
            on_audio_chunk=on_audio_chunk,
            comma_silence_duration=0.3,
            sentence_silence_duration=0.6,
            default_silence_duration=0.6,
            muted=True  # We don't want the server to play the audio, just generate it.
        )
        print("Synthesis finished.")

    except Exception as e:
        print(f"An error occurred during TTS processing: {e}")
    finally:
        # 4. Signal the end of the stream by putting None in the queue.
        audio_queue.put(None)
        # 5. Release the semaphore so the next request can be processed.
        tts_semaphore.release()
        print("Thread finished and semaphore released.")


# --- Audio Generator for Streaming Response ---
def audio_chunk_generator(audio_queue: queue.Queue):
    """
    Yields audio chunks from the queue as they become available.
    This function is used by the StreamingResponse.
    """
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break  # End of stream
        yield chunk
    print("Streaming response generator finished.")


# --- FastAPI Application Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown of the TTS engine.
    """
    global engine, stream

    print("--- Server Starting Up ---")

    if not ZIPVOICE_PROJECT_ROOT or not os.path.isdir(ZIPVOICE_PROJECT_ROOT):
        raise RuntimeError(
            "The 'ZIPVOICE_PROJECT_ROOT' environment variable is not set or is not a valid directory."
        )

    # Initialize the ZipVoiceEngine with a dummy voice (will be replaced per request)
    print("Initializing ZipVoiceEngine...")
    dummy_voice = ZipVoiceVoice(prompt_wav_path=VOICE_ALPHA_WAV_PATH, prompt_text=VOICE_ALPHA_PROMPT_TEXT)
    engine = ZipVoiceEngine(
        zipvoice_root=ZIPVOICE_PROJECT_ROOT,
        voice=dummy_voice,
        model_name="zipvoice",
        device="cuda" if "cuda" in os.getenv("DEVICE", "cuda") else "cpu" # Prefer CUDA
    )

    # Create the TextToAudioStream.
    stream = TextToAudioStream(engine, muted=True, level=logging.DEBUG)

    # Warm up the engine to reduce latency on the first request.
    print("Warming up the engine...")
    stream.feed("Server is now ready.").play(muted=True)

    print("--- Server Ready ---")

    yield  # The application is now running

    # --- Shutdown Logic ---
    print("--- Server Shutting Down ---")
    if engine:
        engine.shutdown()
        print("ZipVoiceEngine shut down successfully.")
    print("--- Shutdown Complete ---")


# --- FastAPI App and Endpoint ---
app = FastAPI(lifespan=lifespan)



# Simple in-memory cache for ZipVoiceVoice objects
_voice_cache = {}


def _list_voices_from_fs(voices_dir: str) -> Dict[str, Dict]:
    """
    Return a mapping of voice_name -> metadata found in the voices directory.
    The metadata currently includes whether the required files exist and a short
    preview of the prompt text if available.
    """
    voices = {}
    try:
        if os.path.isdir(voices_dir):
            for fname in os.listdir(voices_dir):
                if not fname.lower().endswith(".wav"):
                    continue
                name = os.path.splitext(fname)[0]
                wav_path = os.path.join(voices_dir, f"{name}.wav")
                txt_path = os.path.join(voices_dir, f"{name}.txt")
                exists = os.path.isfile(wav_path) and os.path.isfile(txt_path)
                preview = None
                if os.path.isfile(txt_path):
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            preview = f.read(200).strip()
                    except Exception:
                        preview = None
                voices[name] = {"exists": exists, "preview": preview}
    except Exception:
        # On any error, return what we have so far (or empty dict).
        pass
    return voices

@app.post("/api/c3BlZWNo")
async def create_speech(request: TTSRequest):
    """
    Accepts text and a voice name, and streams back raw PCM audio.
    Loads the requested voice from /opt/app-root/voices on-demand, with caching.
    """
    # 1. Check if the TTS engine is busy. If so, reject the request immediately.
    if not tts_semaphore.acquire(blocking=False):
        print("Request rejected: TTS service is busy.")
        raise HTTPException(
            status_code=503,
            detail="TTS service is busy. Please try again later.",
            headers={"Retry-After": "5"}
        )

    # 2. Load the requested voice from cache or disk
    voice_name = request.voice
    voices_dir = "/opt/app-root/voices"
    if voice_name in _voice_cache:
        selected_voice = _voice_cache[voice_name]
    else:
        wav_path = os.path.join(voices_dir, f"{voice_name}.wav")
        txt_path = os.path.join(voices_dir, f"{voice_name}.txt")
        if not os.path.isfile(wav_path) or not os.path.isfile(txt_path):
            tts_semaphore.release()
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{voice_name}' not found in {voices_dir}. Ensure both {voice_name}.wav and {voice_name}.txt exist."
            )
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
        selected_voice = ZipVoiceVoice(prompt_wav_path=wav_path, prompt_text=prompt_text)
        _voice_cache[voice_name] = selected_voice

    # 3. Create a queue to pass audio data from the processing thread to this endpoint.
    audio_queue = queue.Queue()

    # 4. Start the TTS processing in a separate thread.
    print(f"Received request for voice '{voice_name}'. Starting processing thread.")
    threading.Thread(
        target=process_tts_request,
        args=(request.text, selected_voice, audio_queue),
        daemon=True
    ).start()

    # 5. Return a streaming response that yields chunks from our generator.
    # ZipVoice outputs 16-bit PCM audio at a 24000 Hz sample rate.
    return StreamingResponse(
        audio_chunk_generator(audio_queue),
        media_type="audio/pcm; rate=24000; bit-depth=16; channels=1"
    )


@app.get("/api/voices")
async def list_voices():
    """Return a JSON object with all available voices.

    This merges voices discovered on disk under `/opt/app-root/voices` with
    any voices stored in the in-memory cache `_voice_cache`.
    """
    voices_dir = "/opt/app-root/voices"

    # 1) Voices from filesystem
    fs_voices = _list_voices_from_fs(voices_dir)

    # 2) Voices from cache
    cache_voices = {}
    for name, v in _voice_cache.items():
        # ZipVoiceVoice stores prompt_wav_path and prompt_text; provide a small preview
        preview = None
        try:
            if getattr(v, "prompt_text", None):
                preview = str(v.prompt_text)[:200]
        except Exception:
            preview = None
        cache_voices[name] = {"exists": True, "preview": preview}

    # 3) Merge (cache overrides filesystem entries)
    merged = fs_voices.copy()
    merged.update(cache_voices)

    # 4) Return as a simple mapping name -> metadata
    return {"voices": merged}


# --- Main Guard ---
if __name__ == "__main__":
    print("Starting FastAPI server for ZipVoice TTS...")
    if not ZIPVOICE_PROJECT_ROOT:
        print("\nERROR: The 'ZIPVOICE_PROJECT_ROOT' environment variable is not set.")
        print("Please set it to the path of your ZipVoice project clone before running.")
        exit(1)

    uvicorn.run(app, host="0.0.0.0", port=9086)