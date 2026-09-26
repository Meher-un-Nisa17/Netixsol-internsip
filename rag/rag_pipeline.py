import os
import json
import datetime
import re
import hashlib
from logging.handlers import RotatingFileHandler
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from json_repair import repair_json
import logging
from dotenv import load_dotenv
from openai import OpenAI

from rag.recommendation_engine import PropertyRecommendationEngine
from utils.employee_manager import EmployeeAssignmentManager
from utils.visit_db_manager import VisitDatabaseManager
from utils.n8n_integration import trigger_n8n_booking_workflow
from utils.crm_store import CRMLoggingStore
from utils.settings import BUSINESS_EMAIL

load_dotenv()

FAQ_CSV_PATH = "data/processed/faqs.csv"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma_db")

APP_LOG_PATH = os.getenv("APP_LOG_PATH", "conversation_audit.log")
os.makedirs(os.path.dirname(APP_LOG_PATH) or ".", exist_ok=True)
logging.basicConfig(
    handlers=[
        RotatingFileHandler(APP_LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"),
        logging.StreamHandler(),
    ],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

URDU_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
NUMBER_WORDS = {
    "ایک": 1, "اک": 1, "ایک": 1, "aik": 1, "ek": 1, "one": 1,
    "دو": 2, "do": 2, "two": 2, "تین": 3, "teen": 3, "three": 3,
    "چار": 4, "chaar": 4, "four": 4, "پانچ": 5, "paanch": 5, "five": 5,
    "چھ": 6, "chhay": 6, "سات": 7, "saat": 7, "آٹھ": 8, "aath": 8,
    "نو": 9, "nau": 9, "دس": 10, "das": 10, "ten": 10,
    "گیارہ": 11, "بارہ": 12, "تیرہ": 13, "چودہ": 14, "پندرہ": 15,
    "بیس": 20, "پچاس": 50, "pachaas": 50, "سو": 100, "sau": 100,
}

def _number_value(token: str):
    token = token.lower().replace(",", "").strip()
    if token == "ڈیڑھ" or token == "derh":
        return 1.5
    if token in NUMBER_WORDS:
        return NUMBER_WORDS[token]
    try:
        return float(token)
    except ValueError:
        return None

def _extract_budget(transcript: str):
    """Read common numeric and spoken UrduLish budget forms without treating phone/time as budget."""
    text = transcript.lower().translate(URDU_DIGITS)
    number_pattern = r"\d[\d,]*(?:\.\d+)?|ڈیڑھ|derh|" + "|".join(
        re.escape(word) for word in sorted(NUMBER_WORDS, key=len, reverse=True)
    )
    magnitudes = (
        (r"crores?|کروڑ", 10_000_000),
        (r"lakhs?|lacs?|لاکھ", 100_000),
        (r"thousand|hazar|ہزار", 1_000),
    )
    total = 0.0
    found_magnitude = False
    for magnitude_pattern, multiplier in magnitudes:
        pattern = rf"(?<!\w)({number_pattern})\s*(?:{magnitude_pattern})(?!\w)"
        for match in re.finditer(pattern, text):
            value = _number_value(match.group(1))
            if value is not None:
                total += value * multiplier
                found_magnitude = True
    if found_magnitude:
        return int(total)

    # For plain numeric budgets, bind the number to the budget marker. Do not
    # grab an earlier bedroom/Marla/Kanal count (for example, "5 Marla ...
    # budget 50000000"). Some callers also put the amount before "Budget".
    numeric = r"\d[\d,]*(?:\.\d+)?"
    marker = r"budget|pkr|rupees?|بجٹ|روپے"
    for marker_match in re.finditer(marker, text):
        after = text[marker_match.end():marker_match.end() + 60]
        after_match = re.search(rf"^\s*(?:is|of|up to|tak|hai|ہے|:|=)?\s*({numeric})", after)
        if after_match:
            return int(float(after_match.group(1).replace(",", "")))
        before = text[max(0, marker_match.start() - 60):marker_match.start()]
        before_matches = list(re.finditer(rf"(?<!\w)({numeric})(?!\w)", before))
        if before_matches:
            return int(float(before_matches[-1].group(1).replace(",", "")))
    numeric_values = re.findall(rf"(?<!\w)({numeric})(?!\w)", text)
    if len(numeric_values) == 1 and re.search(r"budget|pkr|rupees?|بجٹ|روپے", text):
        return int(float(numeric_values[0].replace(",", "")))
    return None


def _extract_meeting_time(transcript: str):
    """Parse explicit ISO or common UrduLish day/month/time phrases to local time."""
    text = transcript.lower().translate(URDU_DIGITS)
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})[ t](\d{1,2}:\d{2})(?::\d{2})?\b", text)
    if iso_match:
        try:
            parsed = datetime.datetime.fromisoformat(f"{iso_match.group(1)} {iso_match.group(2)}")
            return parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return None

    months = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
        "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    month_names = "|".join(sorted(months, key=len, reverse=True))
    pattern = (
        rf"\b(\d{{1,2}})\s+({month_names})\s+(?:(subha|subhe|morning|am|dupehar|dopahar|"
        rf"afternoon|shaam|sham|evening|raat|night|pm)\s+)?"
        rf"(\d{{1,2}})(?::(\d{{2}}))?\s*(?:baj(?:e|ay)?|o'clock)?\s*"
        rf"(am|pm)?\b"
    )
    match = re.search(pattern, text)
    if not match:
        return None
    day = int(match.group(1))
    month = months[match.group(2)]
    period = (match.group(3) or match.group(6) or "").lower()
    hour = int(match.group(4))
    minute = int(match.group(5) or 0)
    if hour > 23 or minute > 59:
        return None
    if period in {"subha", "subhe", "morning", "am"}:
        hour = 0 if hour == 12 else hour
    elif period in {"dupehar", "dopahar", "afternoon", "shaam", "sham", "evening", "raat", "night", "pm"}:
        hour = hour if hour == 12 else hour + 12
    elif 1 <= hour <= 11:
        # An unqualified 1–11 is ambiguous. Ask the caller instead of booking
        # the old time or silently guessing AM/PM.
        return None
    try:
        today = datetime.date.today()
        year = today.year
        parsed_date = datetime.date(year, month, day)
        if parsed_date < today:
            parsed_date = datetime.date(year + 1, month, day)
        return datetime.datetime.combine(parsed_date, datetime.time(hour, minute)).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return None

class RealEstateRAGPipeline:
    def __init__(self):
        self.recommender = PropertyRecommendationEngine()
        self.employee_manager = EmployeeAssignmentManager()
        self.visit_db = VisitDatabaseManager(os.getenv("VISIT_DB_PATH", "data/visits.db"))
        self.crm = CRMLoggingStore(os.getenv("CRM_DB_PATH", "crm_store.db"))
        
        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        if not ollama_api_key:
            raise RuntimeError("OLLAMA_API_KEY is missing from the environment")
        self.llm_client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1"),
            api_key=ollama_api_key,
        )
        # Ollama Cloud model names are configurable for accounts with different access.
        self.model_name = os.getenv("OLLAMA_MODEL", "gemma4:31b")

        # --- FAQ VECTOR STORE (semantic RAG over faqs.csv) ---
        self.faq_collection = None
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            self.faq_collection = chroma_client.get_or_create_collection(
                name="faqs_production_rag",
                embedding_function=embedding_fn
            )
            self._ensure_faq_index_populated()
        except Exception as e:
            logging.error(f"FAQ vector store init failed: {e}")

        # --- CONVERSATION / SLOT-FILLING STATE ---
        # NOTE: this used to be stranded as unreachable code inside
        # _retrieve_faq_context() (after its `return`), which meant
        # self.state / self.conversation_history were NEVER created and
        # every call to handle_turn() raised AttributeError. Fixed by
        # moving it into __init__ where it belongs.
        self.state = {
            "property_type": None,
            "city": None,
            "deal_type": None,
            "max_budget": None,
            "min_bedrooms": None,
            "target_area": None,
            "client_name": None,
            "phone": None,
            "meeting_time": None,
            "wants_to_visit": False,
            "wants_to_reschedule": False,
            "wants_to_cancel": False,
            "objection_raised": False,
            "last_event_id": None,
            "last_recommended_property": None,
            "last_client_message": ""
        }
        self.state["appointment_status"] = None
        self.state["booking_error"] = None

        current_date_str = datetime.date.today().strftime("%A, %B %d, %Y")

        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    "You are Uzma, a warm, professional, and persuasive female real estate sales executive in Pakistan. "
                    "You are speaking on a live phone call in urdu/hindi.\n\n"
                    f"CURRENT TIME CONTEXT: Today is {current_date_str}. All relative date references MUST be calculated dynamically relative to this exact date.\n\n"
                    "CRITICAL LANGUAGE RULE (MANDATORY):\n"
                    "- Write Urdu vocabulary in Urdu script only. Never transliterate Urdu words into Latin/Roman letters.\n"
                    "- When using an English word, keep it in English/Latin letters (e.g. Flat, Rent, Budget, PKR, Bedrooms, Visit, Appointment). Do not write English words in Urdu script.\n"
                    "- Mix Urdu and English naturally, but keep each word in its proper script. Example: 'جی، آپ کا Budget نوٹ کر لیا ہے۔ لاہور میں یہ Flat دستیاب ہے۔ کیا آپ Visit کرنا چاہیں گے؟'\n"
                    "- Every customer-facing spoken_response, including greetings, apologies, confirmations, and questions, must follow this Urdu-script plus English-words rule.\n"
                    "- Never ask for information already present in State Memory. Confirm the captured value and continue the property search.\n\n"
                    "- You MUST write your responses in Urdu script (اردو) even if question is in english script.\n"
                    "- Keep proper names, city names, real estate terms, and numbers in English when they are English words; write Urdu words in Urdu script.\n\n"
                    "ENTITY EXTRACTION & NUMERIC CONVERSION RULES (CRITICAL):\n"
                    "- Carefully listen to user inputs for property parameters (budget, bedrooms, city, deal type, property type).\n"
                    "- MARLA/KANAL DISTINCTION (CRITICAL): 'Marla' or 'Kanal' refers to plot/house area size (e.g., 'paanch marla' = 5 Marla property). NEVER map 'marla' or 'kanal' to `min_bedrooms`. Set `min_bedrooms` ONLY if the user explicitly mentions bedrooms/rooms (e.g., '3 bed'). Put marla/kanal size in `target_area` (e.g., '5 Marla').\n"
                    "- Convert spoken Pakistani numbers/budgets into precise numeric floats/integers:\n"
                    "  * Examples: 'chaar crore' -> max_budget: 40000000, 'pachaas hazar' -> max_budget: 50000, 'aik arab' -> max_budget: 1000000000.\n"
                    "  * Examples: 'chaar bed' / 'char rooms' -> min_bedrooms: 4.\n"
                    "- Once a parameter (like budget or location) is provided by the client, update it in the JSON keys and NEVER lose or overwrite it with null unless explicitly changed by the client.\n\n"
                    "STATE MEMORY GUARDRAIL (CRITICAL):\n"
                    "- ALWAYS check the provided 'State Memory' before asking any question.\n"
                    "- If a piece of information is ALREADY present in the State Memory, NEVER ask for it again.\n\n"
                    "OUTPUT FORMAT:\n"
                    "You MUST return your response strictly as a valid JSON object with EXACTLY these keys:\n"
                    "- 'property_type': string or null (House, Flat, Commercial, Plot)\n"
                    "- 'city': string or null\n"
                    "- 'deal_type': 'rent' or 'sale' or null\n"
                    "- 'max_budget': float or null\n"
                    "- 'min_bedrooms': int or null\n"
                    "- 'target_area': string or null\n"
                    "- 'client_name': string or null\n"
                    "- 'phone': string or null\n"
                    "- 'meeting_time': string 'YYYY-MM-DD HH:MM' or null\n"
                    "- 'wants_to_visit': boolean\n"
                    "- 'wants_to_reschedule': boolean\n"
                    "- 'wants_to_cancel': boolean\n"
                    "- 'objection_raised': boolean\n"
                    "- 'spoken_response': string (Your natural Urdu response with English real estate loanwords, under 3 sentences)."
                )
            }
        ]

    def _ensure_faq_index_populated(self):
        """Keep the Chroma FAQ index aligned with the currently deployed CSV."""
        if self.faq_collection is None:
            return
        try:
            if not os.path.exists(FAQ_CSV_PATH):
                logging.error(f"FAQ CSV not found at {FAQ_CSV_PATH}; FAQ RAG will have no data.")
                return
            with open(FAQ_CSV_PATH, "rb") as faq_file:
                source_hash = hashlib.sha256(faq_file.read()).hexdigest()
            collection_metadata = dict(self.faq_collection.metadata or {})
            if collection_metadata.get("source_sha256") == source_hash:
                return

            df = pd.read_csv(FAQ_CSV_PATH)
            ids = [f"faq_{i}" for i in range(len(df))]
            documents = df["question"].astype(str).tolist()
            metadatas = [
                {"answer": row["answer"], "category": row.get("category", "")}
                for _, row in df.iterrows()
            ]
            existing_ids = self.faq_collection.get(include=["metadatas"]).get("ids", [])
            stale_ids = sorted(set(existing_ids) - set(ids))
            if stale_ids:
                self.faq_collection.delete(ids=stale_ids)
            if ids:
                self.faq_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            collection_metadata["source_sha256"] = source_hash
            self.faq_collection.modify(metadata=collection_metadata)
            logging.info(f"FAQ vector store populated with {len(df)} entries from {FAQ_CSV_PATH}")
        except Exception as e:
            logging.error(f"Failed to populate FAQ vector store: {e}")

    def _retrieve_faq_context(self, query: str, n_results: int = 2, max_distance: float = 0.8) -> str:
        """Semantic search over the FAQ knowledge base; returns matched Q&A as grounding context."""
        if not self.faq_collection or not query:
            return ""
        try:
            results = self.faq_collection.query(query_texts=[query], n_results=n_results)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if results.get("distances") else [0] * len(docs)
            matches = []
            for doc, meta, dist in zip(docs, metas, distances):
                if dist is not None and dist > max_distance:
                    continue
                answer = meta.get("answer", "") if isinstance(meta, dict) else ""
                matches.append(f"Q: {doc}\nA: {answer}")
            if matches:
                return "RELEVANT COMPANY FAQ KNOWLEDGE:\n" + "\n---\n".join(matches)
        except Exception as e:
            logging.error(f"FAQ retrieval error: {e}")
        return ""

    def _get_current_funnel_stage(self) -> str:
        if self.state.get("wants_to_cancel") or self.state.get("wants_to_reschedule"):
            if not self.state.get("last_event_id") and not self.state.get("phone"):
                return "CANCELLATION/RESCHEDULING STAGE: The client wants to cancel or reschedule, but their phone number or event ID is missing. You MUST ask for their registered phone number or name politely. Do NOT ask about property types or cities."
            return "CANCELLATION/RESCHEDULING STAGE: Processing appointment modification."
        elif self.state.get("appointment_status") == "BOOKED" and not self.state.get("wants_to_visit"):
            return "STAGE FINAL: The booking is already successfully completed and the client said goodbye. Politely say Allah Hafiz and wrap up the call."
        elif not self.state.get("property_type") or not self.state.get("deal_type") or not self.state.get("city"):
            return "STAGE 1: Confirm property type, deal type, and city."
        elif not self.state.get("wants_to_visit"):
            return (
                "STAGE 2: You MUST explicitly read out the available property title, price, and bedrooms "
                "from the 'Database Context' to the client in your spoken response. "
                "Do NOT ask for generic details if a matching property is found. Pitch the specific property "
                "and ask if they would like to visit it."
            )
        elif not self.state.get("client_name") or not self.state.get("phone") or not self.state.get("meeting_time"):
            return "STAGE 3: Client agreed to a visit! You MUST collect their full Name, Phone Number, and preferred Date & Time."
        else:
            return "STAGE 4: All details collected. Confirm the visit."

    def handle_turn(self, transcript: str) -> str:
        logging.info("Received caller turn (%s characters).", len(transcript))
        self.conversation_history.append({"role": "user", "content": transcript})
        self.state["last_client_message"] = transcript
        lower_transcript = transcript.lower()
        explicit_preference_fields = set()
        normalized = lower_transcript.translate(URDU_DIGITS)
        city_terms = {
            "Islamabad": ("اسلام آباد", "اسلاماباد", "islamabad", "islamad", "islamabaad"),
            "Lahore": ("لاہور", "lahore"),
            "Karachi": ("کراچی", "karachi"),
            "Faisalabad": ("فیصل آباد", "فیصلاباد", "faisalabad", "faislabad"),
        }
        for city_name, terms in city_terms.items():
            if any(term in normalized for term in terms):
                self.state["city"] = city_name
                explicit_preference_fields.add("city")
                break

        if any(term in normalized for term in ("rent", "کرایہ", "کرائے", "رینٹ", "kiraya")):
            self.state["deal_type"] = "rent"
            explicit_preference_fields.add("deal_type")
        elif any(term in normalized for term in ("buy", "khareed", "خرید", "sale", "فروخت", "بیچنا")):
            self.state["deal_type"] = "sale"
            explicit_preference_fields.add("deal_type")
        elif "kharred" in normalized:
            self.state["deal_type"] = "sale"
            explicit_preference_fields.add("deal_type")

        if any(term in normalized for term in ("گھر", "ghar", "house")):
            self.state["property_type"] = "House"
            explicit_preference_fields.add("property_type")
        elif any(term in normalized for term in ("فلیٹ", "اپارٹمنٹ", "flat", "apartment")):
            self.state["property_type"] = "Flat"
            explicit_preference_fields.add("property_type")
        elif any(term in normalized for term in ("پلاٹ", "plot")):
            self.state["property_type"] = "Plot"
            explicit_preference_fields.add("property_type")
        elif any(term in normalized for term in ("کمرشل", "commercial")):
            self.state["property_type"] = "Commercial"
            explicit_preference_fields.add("property_type")

        # Capture plot/house size separately from bedroom count (e.g. "1 kanal ka ghar").
        area_pattern = r"(?<!\w)(\d+(?:\.\d+)?|" + "|".join(
            re.escape(word) for word in sorted(NUMBER_WORDS, key=len, reverse=True)
        ) + r")\s*(kanals?|کنال|marla|مرلہ|مرلے)(?!\w)"
        area_match = re.search(area_pattern, normalized)
        if area_match:
            area_number = _number_value(area_match.group(1))
            area_unit = "Kanal" if area_match.group(2).startswith(("kanal", "کنال")) else "Marla"
            if area_number is not None:
                self.state["target_area"] = f"{area_number:g} {area_unit}"
                explicit_preference_fields.add("target_area")

        bedroom_pattern = r"(\d+|" + "|".join(
            re.escape(word) for word in sorted(NUMBER_WORDS, key=len, reverse=True)
        ) + r")\s*(?:bed(?:\s*room)?s?|بیڈ(?:\s*روم)?|پیڈ(?:\s*روم)?|کمرے?)"
        bedroom_match = re.search(bedroom_pattern, normalized)
        if bedroom_match:
            bedroom_count = _number_value(bedroom_match.group(1))
            if bedroom_count is not None:
                self.state["min_bedrooms"] = int(bedroom_count)
                explicit_preference_fields.add("min_bedrooms")
        elif area_match:
            # A size count is not a bedroom count; keep the prior bedroom
            # preference (including None) instead of accepting an LLM guess.
            explicit_preference_fields.add("min_bedrooms")

        extracted_budget = _extract_budget(transcript)
        if extracted_budget is not None:
            self.state["max_budget"] = extracted_budget
            explicit_preference_fields.add("max_budget")
        if os.getenv("DEBUG_AGENT_EXTRACTION", "").lower() in {"1", "true", "yes"}:
            print("[Agent extraction]", {key: self.state.get(key) for key in (
                "city", "deal_type", "property_type", "min_bedrooms", "max_budget"
            )})

        # --- 1. DYNAMIC REGEX EXTRACTION (PHONE, DATE/TIME, NAME) ---
        phone_match = re.search(r'\b03\d{9}\b', transcript)
        if phone_match:
            self.state["phone"] = phone_match.group(0)

        extracted_meeting_time = _extract_meeting_time(transcript)
        if extracted_meeting_time:
            self.state["meeting_time"] = extracted_meeting_time
            explicit_preference_fields.add("meeting_time")

        name_patterns = [
            r"(?:mera\s+naam|my\s+name\s+is)\s+([A-Za-z\s]+?)(?=\s+hai|\s+ho|\s+hoon|\s+phone|\s+03|\s*,|\s*\.|$)",
            r"(?:main|i\s+am)\s+([A-Za-z\s]+?)(?=\s+baat|\s+bol|\s+hai|\s+hoon|\s*,|\s*\.|$)",
            r"\bnaam\s+([A-Za-z\s]+?)(?=\s+hai|\s+ho|\s+phone|\s*,|\s*\.|$)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 2 and extracted.lower() not in ["yeh", "kya", "hai", "ka", "g", "ji"]:
                    self.state["client_name"] = extracted.title()
                    break

        # --- 2. CRM LOGGING: USER TRANSCRIPT ---
        if self.state.get("phone"):
            self.crm.log_transcript_turn(self.state["phone"], "User", transcript)

        # --- 3. PRE-CHECK INTENT FROM TRANSCRIPT ---
        action_intent = None
        if any(k in lower_transcript for k in ["cancel", "rad", "nahi chahiy", "mansookh", "منسوخ", "کینسل"]):
            action_intent = "cancel"
            self.state["wants_to_cancel"] = True
            self.state["wants_to_visit"] = False
            self.state["wants_to_reschedule"] = False
        elif any(k in lower_transcript for k in ["reschedule", "change time", "doosra time", "badal", "tabdeel", "ری شیڈول"]):
            action_intent = "reschedule"
            self.state["wants_to_reschedule"] = True
            self.state["wants_to_visit"] = False
            self.state["wants_to_cancel"] = False
        elif any(k in lower_transcript for k in ["visit", "schedule", "book", "confirm", "dekhna", "daikhna", "fix kar", "spot visit", "وزٹ"]):
            if not self.state.get("wants_to_reschedule") and not self.state.get("wants_to_cancel"):
                action_intent = "book"
                self.state["wants_to_visit"] = True

        if self.state.get("booking_error") and extracted_meeting_time and not action_intent:
            action_intent = "book"
            self.state["wants_to_visit"] = True
        if action_intent == "book" and self.state.get("booking_error") and not extracted_meeting_time:
            self.state["meeting_time"] = None
            explicit_preference_fields.add("meeting_time")
        if action_intent == "book" and extracted_meeting_time and self.state.get("appointment_status") in {"BOOKED", "RESCHEDULED"}:
            action_intent = "reschedule"
            self.state["wants_to_reschedule"] = True
            self.state["wants_to_visit"] = False
            self.state["wants_to_cancel"] = False

        if action_intent == "reschedule" and not extracted_meeting_time:
            self.state["meeting_time"] = None
            explicit_preference_fields.add("meeting_time")
            self.state["booking_error"] = "A new appointment date and time are required."
            reply = "براہِ کرم اپنی Visit کے لیے نئی تاریخ اور وقت بتائیں۔"
            self.conversation_history.append({"role": "assistant", "content": reply})
            return reply

        # --- 4. OVERRIDE FOR CANCELLATION & RESCHEDULE (IF PHONE IS MISSING) ---
        if (self.state.get("wants_to_cancel") or self.state.get("wants_to_reschedule")) and not self.state.get("phone"):
            action_name = "کینسل" if self.state.get("wants_to_cancel") else "ری شیڈول"
            spoken_reply = f"اپنی اپائنٹمنٹ {action_name} کرنے کے لیے براہ کرم اپنا رجسٹرڈ فون نمبر بتا دیں۔"
            logging.info("Generated cancellation/reschedule information request.")
            return spoken_reply

        # --- 5. EXECUTE ACTIONS FOR CANCELLATION / RESCHEDULE ---
        if (self.state.get("wants_to_cancel") or self.state.get("wants_to_reschedule")) and self.state.get("phone"):
            was_cancelling = self.state.get("wants_to_cancel")
            was_rescheduling = self.state.get("wants_to_reschedule")

            self._process_calendar_actions()

            client = self.state.get("client_name") or "جناب"
            m_time = self.state.get("meeting_time", "")

            if was_cancelling and self.state.get("wants_to_cancel") is False and not self.state.get("booking_error"):
                spoken_reply = f"جی {client}! آپ کی اپائنٹمنٹ کامیابی سے کینسل کر دی گئی ہے۔ اس کی اطلاع ہماری ٹیم کو بھی میل کے ذریعے بھجوا دی گئی ہے۔"
            elif was_rescheduling and self.state.get("wants_to_reschedule") is False and not self.state.get("booking_error"):
                spoken_reply = f"جی {client}! آپ کی اپائنٹمنٹ کامیابی سے {m_time} پر ری شیڈول کر دی گئی ہے۔"
            else:
                error = str(self.state.get("booking_error") or "")
                if "No matching appointment" in error:
                    spoken_reply = "معذرت، اس نمبر پر کوئی شیڈولڈ Appointment نہیں ملی۔ براہِ کرم درست Phone Number بتائیں۔"
                elif was_rescheduling:
                    spoken_reply = "معذرت، نئی تاریخ Calendar میں محفوظ نہیں ہو سکی۔ براہِ کرم کوئی دوسرا خالی وقت بتائیں۔"
                elif was_cancelling:
                    spoken_reply = "معذرت، Appointment ابھی Calendar سے Cancel نہیں ہو سکی۔ براہِ کرم دوبارہ کوشش کریں۔"
                else:
                    spoken_reply = "معذرت، آپ کی درخواست مکمل نہیں ہو سکی۔ براہِ کرم دوبارہ کوشش کریں۔"

            if self.state.get("phone"):
                self.crm.log_transcript_turn(self.state["phone"], "Uzma (Assistant)", spoken_reply)

            logging.info("Completed an appointment update action.")
            return spoken_reply

        # --- 6. RECOMMENDER DATABASE CONTEXT WITH NORMALIZATION ---
        property_context = ""
        city_val = self.state.get("city")
        deal_val = self.state.get("deal_type")

        city_mapping = {"اسلام آباد": "Islamabad", "لاہور": "Lahore", "کراچی": "Karachi", "فیصل آباد": "Faisalabad", "Faislabad": "Faisalabad"}
        deal_mapping = {"rent": "rent", "sale": "sale", "خریدنا": "sale", "کرایہ": "rent"}

        norm_city = city_mapping.get(city_val, city_val) if city_val else None
        norm_deal = deal_mapping.get(str(deal_val).lower(), deal_val) if deal_val else None

        if explicit_preference_fields.intersection({"city", "deal_type", "property_type", "max_budget", "min_bedrooms", "target_area"}):
            self.state["last_recommendations"] = None
            self.state["last_recommended_property"] = None

        recommended_rows = []
        if norm_city and norm_deal and hasattr(self.recommender, "recommend"):
            try:
                recs = self.recommender.recommend(
                    city=norm_city,
                    deal_type=norm_deal,
                    property_type=self.state.get("property_type"),
                    max_budget=self.state.get("max_budget"),
                    min_bedrooms=self.state.get("min_bedrooms"),
                    target_area=self.state.get("target_area"),
                    top_n=3
                )
                if hasattr(recs, "empty") and not recs.empty:
                    props_list = []
                    for _, row in recs.iterrows():
                        area_str = f"{row.get('areaValue', '')} {row.get('areaUnit', '')}".strip() or "N/A"
                        recommended_rows.append({
                            "title": str(row.get("propertyTitle", "Property")),
                            "price": row.get("price"),
                            "beds": row.get("beds"),
                            "location": str(row.get("locationText", "N/A")),
                            "area_value": row.get("areaValue"),
                            "area_unit": str(row.get("areaUnit", "")),
                        })
                        props_list.append(f"- {row.get('propertyTitle')} | Price: {row.get('price'):,.0f} PKR | Beds: {row.get('beds')} | Size: {area_str} | Location: {row.get('locationText', 'N/A')}")
                    property_context = "MATCHED PROPERTIES:\n" + "\n".join(props_list)
                    self.state["last_recommendations"] = property_context
                    self.state["last_recommended_property"] = recommended_rows[0]
                else:
                    self.state["last_recommendations"] = None
                    self.state["last_recommended_property"] = None
            except Exception as rec_err:
                logging.error(f"Recommender Error: {rec_err}")

        if not property_context and self.state.get("last_recommendations"):
            lower_transcript = transcript.lower() if 'transcript' in locals() else ""
            if any(keyword in lower_transcript for keyword in ["option", "opt", "dikh", "batay", "bata", "show", "list", "آپشن"]):
                property_context = self.state["last_recommendations"]

        current_stage = self._get_current_funnel_stage()

        # --- SEMANTIC FAQ RAG ---
        faq_context = self._retrieve_faq_context(transcript)

        context_message = {
            "role": "system",
            "content": (
                f"State Memory: {json.dumps(self.state)} | "
                f"Database Context: {property_context} | "
                f"{faq_context} | "
                f"CURRENT SALES STAGE INSTRUCTION: {current_stage}. "
                "If FAQ knowledge above answers the client's question, use it as the factual basis for your "
                "spoken_response instead of guessing."
            )
        }

       # --- 7. LLM CALL & STATE SYNCHRONIZATION ---
       # --- LLM CALL & PARSING ---
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history + [context_message],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            
            # Use repair_json safely
            try:
                result = repair_json(raw_content, return_objects=True)
                if not isinstance(result, dict):
                    result = {"spoken_response": raw_content}
            except Exception as e:
                logging.error(f"JSON Repair Error: {e}")
                result = {"spoken_response": raw_content}

            # Safely extract spoken response with a fallback text
            spoken_reply = result.get("spoken_response")
            if not spoken_reply or not isinstance(spoken_reply, str):
                spoken_reply = "جی بالکل، میں آپ کی بات سمجھ رہی ہوں۔ بتائیں مزید کیا مدد کر سکتی ہوں؟"

            spoken_reply = spoken_reply.strip()

            # Safely update state variables from result dict
            for key in ["city", "deal_type", "property_type", "max_budget", "min_bedrooms", "target_area", "client_name", "phone", "meeting_time"]:
                if key in result and result[key] is not None and key not in explicit_preference_fields:
                    self.state[key] = result[key]

            for bool_field in ["wants_to_visit", "wants_to_reschedule", "wants_to_cancel", "objection_raised"]:
                if bool_field in result:
                    self.state[bool_field] = bool(result[bool_field])
            if action_intent == "book":
                self.state["wants_to_visit"] = True
                self.state["wants_to_reschedule"] = False
                self.state["wants_to_cancel"] = False
            elif action_intent == "reschedule":
                self.state["wants_to_visit"] = False
                self.state["wants_to_reschedule"] = True
                self.state["wants_to_cancel"] = False
            elif action_intent == "cancel":
                self.state["wants_to_visit"] = False
                self.state["wants_to_reschedule"] = False
                self.state["wants_to_cancel"] = True

            # Do not ask for budget again after it has been captured. When possible,
            # answer with a real listing from the recommender rather than a generic LLM reply.
            reply_lower = spoken_reply.lower()
            asks_for_budget = any(term in reply_lower for term in ("budget", "بجٹ", "مطلوبہ بجٹ"))
            budget = self.state.get("max_budget")
            if asks_for_budget and budget is not None:
                if recommended_rows:
                    listing = recommended_rows[0]
                    price = listing["price"]
                    price_text = f"PKR {float(price):,.0f}" if pd.notna(price) else "price details"
                    bed_text = f", {int(float(listing['beds']))} bedrooms" if pd.notna(listing["beds"]) else ""
                    spoken_reply = (
                        f"جی، آپ کا PKR {float(budget):,.0f} Budget نوٹ کر لیا ہے۔ "
                        f"{listing['title']}{bed_text}, {listing['location']} میں {price_text} کی دستیاب ہے۔ "
                        "کیا آپ اس Property کا Visit schedule کرنا چاہیں گے؟"
                    )
                else:
                    city = self.state.get("city") or "is area"
                    deal = self.state.get("deal_type") or "sale ya rent"
                    spoken_reply = (
                        f"جی، آپ کا PKR {float(budget):,.0f} Budget نوٹ کر لیا ہے۔ "
                        f"فی الحال {city} میں {deal} کے لیے اس Budget میں matching Property نہیں ملی۔ "
                        "کیا آپ Area یا Budget میں کچھ flexibility رکھ سکتے ہیں؟"
                    )

            # Execute a booking only after the caller has provided the required details.
            if self.state.get("wants_to_visit") and self.state.get("phone") and self.state.get("meeting_time"):
                outcome = self._process_calendar_actions()
                if outcome:
                    self.conversation_history.append({"role": "assistant", "content": outcome})
                    return outcome

            logging.info("Generated assistant response (%s characters).", len(spoken_reply))
            self.conversation_history.append({"role": "assistant", "content": raw_content})
            return spoken_reply

        except Exception as e:
            logging.error(f"Pipeline Error: {e}")
            return "\u0645\u0639\u0630\u0631\u062a\u060c \u0627\u0633 \u0648\u0642\u062a AI service \u0633\u06d2 \u0631\u0627\u0628\u0637\u06c1 \u0645\u06cc\u06ba \u0645\u0633\u0626\u0644\u06c1 \u06c1\u06d2\u06d4 \u0628\u0631\u0627\u06c1 \u06a9\u0631\u0645 \u062a\u06be\u0648\u0691\u06cc \u062f\u06cc\u0631 \u0628\u0639\u062f \u062f\u0648\u0628\u0627\u0631\u06c1 \u06a9\u0648\u0634\u0634 \u06a9\u0631\u06cc\u06ba\u06d4"
            return "جی بالکل، میں آپ کی بات سن رہی ہوں۔ کیا آپ اپنا مطلوبہ بجٹ بتانا چاہیں گے؟"
        
    def _process_calendar_actions(self):
        phone = self.state.get("phone")
        meeting_time = self.state.get("meeting_time")
        client_name = self.state.get("client_name") or "جناب"

        wants_cancel = self.state.get("wants_to_cancel", False)
        wants_reschedule = self.state.get("wants_to_reschedule", False)

        if phone and meeting_time and not wants_cancel and not wants_reschedule:
            self.state["wants_to_visit"] = True

        saved_visit = None
        if phone and self.visit_db and hasattr(self.visit_db, "find_visit_by_phone"):
            try:
                saved_visit = self.visit_db.find_visit_by_phone(phone)
                if saved_visit and saved_visit.get("event_id"):
                    self.state["last_event_id"] = saved_visit.get("event_id")
            except Exception as err:
                logging.error(f"Database lookup error: {err}")

        event_id = self.state.get("last_event_id")

        if wants_cancel:
            if not saved_visit or not event_id:
                self.state["booking_error"] = "No matching appointment was found."
                return
            try:
                result = trigger_n8n_booking_workflow(
                    intent="cancel",
                    client_name=client_name,
                    phone=phone,
                    city=saved_visit.get("city"),
                    target_area=saved_visit.get("target_area"),
                    meeting_time=saved_visit.get("meeting_time"),
                    event_id=event_id,
                    property_title=saved_visit.get("property_title"),
                    assigned_employee_email=(saved_visit.get("assigned_employee") or {}).get("email"),
                    assigned_employee_name=(saved_visit.get("assigned_employee") or {}).get("name"),
                )
            except Exception as err:
                logging.error(f"n8n cancellation error: {err}")
                result = {"status": "error"}
            if result.get("status") != "success":
                self.state["booking_error"] = result.get("message", "Calendar cancellation failed.")
                return

            if phone and self.visit_db and hasattr(self.visit_db, "remove_visit"):
                try:
                    self.visit_db.remove_visit(phone, event_id)
                except Exception as err:
                    logging.error(f"Database removal error: {err}")

            if phone:
                self.crm.log_appointment_event(
                    phone=phone,
                    action_type="CANCELLED",
                    event_id=event_id,
                    notes="Client requested appointment cancellation."
                )

            self.state["wants_to_cancel"] = False
            self.state["last_event_id"] = None
            self.state["appointment_status"] = "CANCELLED"
            self.state["booking_error"] = None
            return

        if wants_reschedule and meeting_time:
            if not saved_visit or not event_id:
                self.state["booking_error"] = "No matching appointment was found."
                return
            try:
                result = trigger_n8n_booking_workflow(
                    intent="reschedule",
                    client_name=client_name,
                    phone=phone,
                    meeting_time=meeting_time,
                    event_id=event_id,
                    property_title=saved_visit.get("property_title"),
                    assigned_employee_email=(saved_visit.get("assigned_employee") or {}).get("email"),
                    assigned_employee_name=(saved_visit.get("assigned_employee") or {}).get("name"),
                )
            except Exception as err:
                logging.error(f"n8n rescheduling error: {err}")
                result = {"status": "error"}
            if result.get("status") != "success":
                self.state["booking_error"] = result.get("message", "Calendar rescheduling failed.")
                return

            if saved_visit:
                saved_visit["meeting_time"] = meeting_time
                if self.visit_db and hasattr(self.visit_db, "save_visit"):
                    try:
                        self.visit_db.save_visit(saved_visit)
                    except Exception as err:
                        logging.error(f"Database save error during reschedule: {err}")

            if phone:
                self.crm.log_appointment_event(
                    phone=phone,
                    action_type="RESCHEDULED",
                    event_id=event_id,
                    meeting_time=meeting_time,
                    notes=f"Appointment rescheduled to {meeting_time}."
                )

            self.state["wants_to_reschedule"] = False
            self.state["appointment_status"] = "RESCHEDULED"
            self.state["booking_error"] = None
            self.state["meeting_time"] = None
            return

        if self.state.get("wants_to_visit") and phone and meeting_time:
            city = self.state.get("city")
            target_area = self.state.get("target_area")
            if not city:
                self.state["booking_error"] = "Please provide the city before booking."
                return "Booking کے لیے براہِ کرم Property کا City بتا دیں۔"
            selected_property = self.state.get("last_recommended_property") or {}
            property_title = selected_property.get("title") or f"{target_area} {city} Property Visit"

            emp = {"name": "Agent", "email": BUSINESS_EMAIL}
            if self.employee_manager and hasattr(self.employee_manager, "assign_best_employee"):
                try:
                    assigned = self.employee_manager.assign_best_employee(
                        city=city,
                        target_area=target_area,
                        meeting_time_str=meeting_time,
                        client_text=self.state.get("last_client_message", "")
                    )
                    if assigned:
                        emp = assigned
                except Exception as err:
                    logging.error(f"Employee assignment error: {err}")

            try:
                result = trigger_n8n_booking_workflow(
                    intent="book_visit",
                    client_name=client_name,
                    phone=phone,
                    city=city,
                    target_area=target_area,
                    meeting_time=meeting_time,
                    property_title=property_title,
                    assigned_employee_email=emp.get("email"),
                    assigned_employee_name=emp.get("name"),
                )
            except Exception as err:
                logging.error(f"n8n booking error: {err}")
                result = {"status": "error"}

            event_id = result.get("event_id")
            if result.get("status") != "success" or not event_id:
                self.state["booking_error"] = result.get("message", "Calendar could not confirm the requested slot.")
                self.state["wants_to_visit"] = False
                self.state["meeting_time"] = None
                return "معذرت، Calendar میں وقت confirm نہیں ہو سکا۔ براہِ کرم کوئی دوسرا وقت بتائیں۔"

            visit_record = {
                "client_name": client_name,
                "phone": phone,
                "property_title": property_title,
                "city": city,
                "target_area": target_area,
                "meeting_time": meeting_time,
                "assigned_employee": emp,
                "event_id": event_id
            }
            if not self.visit_db or not hasattr(self.visit_db, "save_visit"):
                self.state["booking_error"] = "Booking storage is unavailable."
                self.state["wants_to_visit"] = False
                return "معذرت، ابھی Booking محفوظ نہیں ہو سکی۔ براہِ کرم دوبارہ کوشش کریں۔"
            try:
                self.visit_db.save_visit(visit_record)
            except Exception as err:
                logging.error(f"Database save error: {err}")
                self.state["booking_error"] = "Booking record could not be saved."
                self.state["wants_to_visit"] = False
                return "معذرت، ابھی Booking محفوظ نہیں ہو سکی۔ براہِ کرم دوبارہ کوشش کریں۔"

            if phone:
                self.crm.log_appointment_event(
                    phone=phone,
                    action_type="BOOKED",
                    event_id=event_id,
                    meeting_time=meeting_time,
                    notes=f"New visit scheduled for {property_title} at {meeting_time}."
                )

            self.state["last_event_id"] = event_id
            self.state["wants_to_visit"] = False
            self.state["appointment_status"] = "BOOKED"
            self.state["booking_error"] = None
            self.state["meeting_time"] = None
            return f"جی {client_name}، {property_title} کا Property Visit {meeting_time} کے لیے Calendar میں Confirm ہو گیا ہے۔"
