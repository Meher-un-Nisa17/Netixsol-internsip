import sys
import os
from pathlib import Path
import time

# Fix path to load root modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from rag.rag_pipeline import RealEstateRAGPipeline
from utils.edge_tts_integration import EdgeVoiceSynthesizer

load_dotenv()

class RealEstateVoicePipeline:
    def __init__(self):
        self.rag_pipeline = RealEstateRAGPipeline()
        try:
            print("[Initializing Edge-TTS Voice Pipeline]...")
            self.synthesizer = EdgeVoiceSynthesizer()
            self.tts_enabled = True
        except Exception as e:
            print(f"[Warning]: Failed to initialize Edge-TTS: {e}")
            self.tts_enabled = False

    def process_voice_interaction(self, transcript: str):
        start_time = time.time()
        print(f"\n[Client Input Text]: '{transcript}'")

        text_response = self.rag_pipeline.handle_turn(transcript)
        
        text_latency = time.time() - start_time
        print(f"[Female Sales Executive Response]: {text_response}")
        print(f"[RAG & Logic Latency]: {text_latency:.2f} seconds")

        if self.tts_enabled:
            print("[Generating Edge-TTS Neural Audio...] ", end="", flush=True)
            output_dir = Path("data/audio_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"sales_response_{timestamp_str}.wav"
            
            # Synthesize via Edge-TTS (ur-PK-UzmaNeural)
            generated_path = self.synthesizer.synthesize_speech(
                text=text_response,
                output_path=str(output_path)
            )
            
            if generated_path:
                print(f"Done! Saved to {generated_path}")
            else:
                print("Failed to generate audio stream.")
        else:
            print("[Warning]: TTS is disabled. Skipping audio generation.")

        total_latency = time.time() - start_time
        print(f"[Total Pipeline Latency]: {total_latency:.2f} seconds")
        return text_response

if __name__ == "__main__":
    voice_pipeline = RealEstateVoicePipeline()
    
    print("\n==================================================")
    print("STARTING INTERACTIVE RUN-TIME TEXT CHAT WITH UZMA")
    print("Type your message below (or type 'exit' to quit).")
    print("==================================================")
    
    print("\n👋 Uzma: السلام علیکم! میں عظمیٰ بات کر رہی ہوں۔ آپ کس طرح کی پراپرٹی تلاش کر رہے ہیں؟")
    
    while True:
        try:
            user_input = input("\n👤 You (Type message): ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "خروج"]:
                print("👋 Ending session. Goodbye!")
                break
                
            voice_pipeline.process_voice_interaction(user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive session. Goodbye!")
            break