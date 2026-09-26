import json
import logging
from rag.rag_pipeline import RealEstateRAGPipeline

logging.basicConfig(level=logging.INFO)

# Multi-turn end-to-end scenarios with Urdu inputs for 4 cities
END_TO_END_SCENARIOS = [
    {
        "city": "Lahore",
        "turns": [
            "مجھے لاہور میں ایک کنال کا گھر خریدنا ہے۔",
            "میرا بجٹ پچاس کروڑ روپے ہے۔",
            "جی ہاں، مجھے اس پراپرٹی کا وزٹ کرنا ہے۔",
            "میرا نام احمد رضا ہے۔",
            "مجھ سے اس نمبر پر رابطہ کریں 03009876543۔",
            "ملاقات کا وقت 2026-04-01 11:00 فکس کر لیں۔"
        ]
    },
    {
        "city": "Islamabad",
        "turns": [
            "اسلام آباد میں پانچ مرلے کا گھر خریدنا ہے۔",
            "بجٹ دو کروڑ ہے میرا۔",
            "اس کا وزٹ طے کر لیں۔",
            "نام ثناء مریم ہے۔",
            "فون نمبر 03011223344 ہے۔",
            "وقت 2026-04-02 14:00 پر رکھ لیں۔"
        ]
    },
    {
        "city": "Karachi",
        "turns": [
            "کراچی میں گھر خریدنا ہے۔",
            "قیمت تین کروڑ تک ہونی چاہیے۔",
            "وزٹ کے لیے وقت دینا ہے۔",
            "نام محمد علی ہے۔",
            "موبائل نمبر 03023344556 ہے۔",
            "ٹائم 2026-04-03 16:00 فکس کر دیں۔"
        ]
    },
    {
        "city": "Faisalabad",
        "turns": [
            "فیصل آباد میں گھر لینا ہے۔",
            "بجٹ ڈیڑھ کروڑ ہے۔",
            "پراپرٹی پسند ہے، وزٹ کرنا ہے۔",
            "عائشہ بی بی نام ہے۔",
            "نمبر یہ ہے 03034455667۔",
            "اپائنٹمنٹ 2026-04-04 12:00 پر بک کر دیں۔"
        ]
    }
]

def run_end_to_end_test():
    print("\n==================================================")
    print("RUNNING END-TO-END URDU AGENT PIPELINE TEST (4 CITIES)")
    print("==================================================")
    
    for scenario in END_TO_END_SCENARIOS:
        city_name = scenario["city"]
        print(f"\n================ CITY: {city_name} ================")
        agent = RealEstateRAGPipeline()
        
        for idx, user_msg in enumerate(scenario["turns"], 1):
            response = agent.handle_turn(user_msg)
            print(f"  [Turn {idx}] User: '{user_msg}'")
            print(f"  [Turn {idx}] Uzma: '{response}'")
            print(f"  [State]: City={agent.state.get('city')}, Budget={agent.state.get('max_budget')}, Visit={agent.state.get('wants_to_visit')}, Name={agent.state.get('client_name')}, Phone={agent.state.get('phone')}, Meeting={agent.state.get('meeting_time')}")

if __name__ == "__main__":
    run_end_to_end_test()