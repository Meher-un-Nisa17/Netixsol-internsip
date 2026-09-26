import sys
import os
from pathlib import Path
import time
import sounddevice as sd
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
from rag.rag_pipeline import RealEstateRAGPipeline
from utils.edge_tts_integration import EdgeVoiceSynthesizer
from deepgram import DeepgramClient

load_dotenv()

class DeepgramRealEstateVoiceAgent:
    def __init__(self):
        print("[System Init]: Loading Real Estate RAG Pipeline...")
        self.rag_pipeline = RealEstateRAGPipeline()
        
        print("[System Init]: Loading Edge-TTS Voice Engine (ur-PK-UzmaNeural)...")
        self.synthesizer = EdgeVoiceSynthesizer()
        
        print("[System Init]: Initializing Deepgram Nova-3 STT Client...")
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is missing from your environment variables (.env)!")
        self.deepgram = DeepgramClient(api_key=api_key)

    def play_audio(self, file_path: str):
        """Plays generated audio file out loud."""
        try:
            import soundfile as sf
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"[Audio Playback Error]: {e}")

    def listen_and_transcribe(self, duration=6, samplerate=16000) -> str:
        """Records from mic, packages into in-memory WAV bytes, and transcribes via Deepgram Nova-3 API."""
        print(f"\n👂 [Listening for up to {duration} seconds... Speak now]:")
        
        # Record audio block from microphone
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        
        # Package raw PCM buffer into an in-memory WAV container so Deepgram reads sample rates correctly
        import io
        import soundfile as sf
        
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, samplerate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        audio_bytes = wav_buffer.read()

        try:
            # Send packaged WAV bytes to Deepgram Nova-3 optimized for Urdu
            response = self.deepgram.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model="nova-3",
                language="ur",
                smart_format=True
            )
            
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript.strip()
        except Exception as e:
            print(f"[Deepgram STT Error]: {e}")
            return ""
        
    def run_live_agent(self):
        print("\n==================================================")
        print("🎙️ LIVE DEEPGRAM + EDGE-TTS AGENT (UZMA) READY")
        print("==================================================")
        
        output_dir = Path("data/audio_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initial Greeting
        greeting = "السلام علیکم! میں عظمیٰ بات کر رہی ہوں۔ آپ کس طرح کی پراپرٹی تلاش کر رہے ہیں"
        print(f"🤖 [Uzma]: {greeting}")
        
        welcome_path = output_dir / "welcome.wav"
        self.synthesizer.synthesize_speech(text=greeting, output_path=str(welcome_path))
        self.play_audio(str(welcome_path))

        try:
            while True:
                user_transcript = self.listen_and_transcribe(duration=7)
                
                if not user_transcript:
                    print("⚠️ [No speech detected or unclear, listening again...]")
                    continue
                
                print(f"👤 [Client Transcript]: '{user_transcript}'")
                
                if any(exit_word in user_transcript.lower() for exit_word in ["exit", "quit", "bye", "allah hafiz"]):
                    farewell = "Allah Hafiz! Phir milte hain."
                    print(f"🤖 [Uzma]: {farewell}")
                    farewell_path = output_dir / "farewell.wav"
                    self.synthesizer.synthesize_speech(text=farewell, output_path=str(farewell_path))
                    self.play_audio(str(farewell_path))
                    break

                # RAG Processing Turn
                start_time = time.time()
                text_response = self.rag_pipeline.handle_turn(user_transcript)
                print(f"🤖 [Uzma Response]: {text_response}")
                print(f"⏱️ [RAG Latency]: {time.time() - start_time:.2f}s")

                # Edge-TTS Response Generation
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                response_path = output_dir / f"response_{timestamp_str}.wav"
                
                self.synthesizer.synthesize_speech(
                    text=text_response,
                    output_path=str(response_path)
                )
                
                self.play_audio(str(response_path))

        except KeyboardInterrupt:
            print("\n\n[Session Terminated]. Allah Hafiz!")

if __name__ == "__main__":
    agent = DeepgramRealEstateVoiceAgent()
    agent.run_live_agent()