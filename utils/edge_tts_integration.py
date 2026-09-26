import asyncio
import edge_tts
import os
import concurrent.futures

class EdgeVoiceSynthesizer:
    def __init__(self, voice: str = "ur-PK-UzmaNeural"):
        self.voice = voice
        print(f"[Initializing Edge-TTS]: Loaded voice profile {self.voice}")

    async def synthesize_speech_async(self, text: str, output_path: str = "output_voice.wav") -> str:
        """Asynchronous synthesis safe for FastAPI and running event loops."""
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error during async Edge-TTS synthesis: {e}")
            return None

    def synthesize_speech(self, text: str, output_path: str = "output_voice.wav") -> str:
        """Synchronous wrapper for script-based runs (like Live_voice_agent.py)."""
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self.synthesize_speech_async(text, output_path))
                        return future.result()
            except Exception:
                pass
            return asyncio.run(self.synthesize_speech_async(text, output_path))
        except Exception as e:
            print(f"Error during Edge-TTS synthesis: {e}")
            return None