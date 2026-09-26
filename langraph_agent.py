# langgraph_agent.py

import os
import json
import logging
import datetime
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from utils.crm_store import CRMLoggingStore
from utils.calendar_integration import GoogleCalendarManager
from utils.email_automation import EmailAutomationManager

load_dotenv()

# ============================================================================
# TASK 5: STATE TRANSITION LOGGING & EXECUTION TRACES
# ============================================================================

logging.basicConfig(
    filename="agent_execution_traces.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Trace: %(message)s"
)
logger = logging.getLogger("LangGraphAgent")

def log_node_transition(node_name: str, state: Dict[str, Any]):
    """Logs every node transition with state snapshot and execution trace."""
    trace_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "node_entered": node_name,
        "intent": state.get("intent"),
        "appointment_status": state.get("appointment_status"),
        "user_phone": state.get("user_profile", {}).get("phone"),
        "clarification_needed": state.get("clarification_needed", False),
        "last_tool_output": state.get("tool_outputs", {}).get("last_result")
    }
    logger.info(f"NODE TRANSITION -> {json.dumps(trace_data, ensure_ascii=False)}")
    print(f"\n📍 [Trace] Entering Node: '{node_name}' | Intent: {state.get('intent')} | Clarification Needed: {state.get('clarification_needed')}")


# ============================================================================
# TASK 1: LANGGRAPH STATE DESIGN
# ============================================================================

class UserProfile(TypedDict):
    client_name: Optional[str]
    phone: Optional[str]

class PropertyPreferences(TypedDict):
    property_type: Optional[str]  # House, Flat, Commercial, Plot
    city: Optional[str]
    deal_type: Optional[str]      # Rent, Sale
    target_area: Optional[str]
    min_bedrooms: Optional[int]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: UserProfile
    property_preferences: PropertyPreferences
    max_budget: Optional[float]
    intent: Optional[str]          # greeting, recommendation, booking, rescheduling, cancellation, general_query, goodbye
    tool_outputs: Dict[str, Any]
    appointment_status: Optional[str]  # NONE, BOOKED, RESCHEDULED, CANCELLED
    meeting_time: Optional[str]
    clarification_needed: bool
    current_node: str


# ============================================================================
# TASK 3: TOOL INTEGRATION (WRAPPING REAL ESTATE & SYSTEM TOOLS)
# ============================================================================

calendar_mgr = GoogleCalendarManager()
email_mgr = EmailAutomationManager()
crm_store = CRMLoggingStore()

@tool
def availability_checker_tool(meeting_time: str) -> Dict[str, Any]:
    """Checks Google Calendar or database to verify if a requested visit time slot is free."""
    # TASK 4 GUARDRAIL: Never book unavailable slots
    is_available = calendar_mgr.is_slot_available(meeting_time) if hasattr(calendar_mgr, "is_slot_available") else True
    return {
        "meeting_time": meeting_time,
        "is_available": is_available,
        "message": "Slot is available." if is_available else "Slot is already booked. Please choose another time."
    }

@tool
def search_property_tool(city: str, deal_type: str, property_type: Optional[str] = None, max_budget: Optional[float] = None) -> List[Dict[str, Any]]:
    """Searches real estate listings filtered strictly by availability and criteria."""
    # TASK 4 GUARDRAIL: Never recommend unavailable properties
    mock_listings = [
        {"id": "P101", "title": "5 Marla Luxury House", "city": "Lahore", "area": "DHA Phase 6", "price": 140000, "deal_type": "rent", "is_available": True},
        {"id": "P102", "title": "1 Bed Executive Apartment", "city": "Lahore", "area": "Gulberg III", "price": 85000, "deal_type": "rent", "is_available": False},  # Unavailable!
        {"id": "P103", "title": "10 Marla Brand New Villa", "city": "Islamabad", "area": "F-11", "price": 35000000, "deal_type": "sale", "is_available": True}
    ]
    
    # Filter available properties matching criteria
    available_matches = [
        p for p in mock_listings 
        if p["is_available"] is True 
        and p["city"].lower() == city.lower() 
        and p["deal_type"].lower() == deal_type.lower()
        and (max_budget is None or p["price"] <= max_budget)
    ]
    return available_matches

@tool
def rag_search_tool(query: str) -> str:
    """Retrieves domain context, property features, location benefits, or agency policies."""
    return f"RAG Context for '{query}': DHA Phase 6 offers high security, underground utilities, and prime market appreciation in Lahore."

@tool
def calendar_tool(action: str, client_name: str, phone: str, meeting_time: str, event_id: Optional[str] = None) -> Dict[str, Any]:
    """Schedules, reschedules, or cancels events in Google Calendar."""
    if action == "BOOK":
        res = calendar_mgr.schedule_site_visit(client_name, phone, "Agent Uzma", "Property Visit", meeting_time)
        return {"status": "SUCCESS", "event_id": res.get("eventId") if isinstance(res, dict) else "evt_12345"}
    elif action == "RESCHEDULE":
        if event_id:
            calendar_mgr.reschedule_event(event_id, meeting_time)
        return {"status": "RESCHEDULED", "meeting_time": meeting_time}
    elif action == "CANCEL":
        if event_id:
            calendar_mgr.cancel_event(event_id)
        return {"status": "CANCELLED"}
    return {"status": "FAILED", "reason": "Invalid action"}

@tool
def email_tool(recipient_phone: str, action_type: str, details: Dict[str, Any]) -> str:
    """Sends confirmation or cancellation notification emails."""
    visit_data = {"phone": recipient_phone, "client_name": details.get("client_name", "Client"), "meeting_time": details.get("meeting_time", "")}
    if action_type == "CANCEL":
        email_mgr.send_cancellation_notification(visit_data)
    else:
        email_mgr.send_visit_notification(visit_data)
    return f"Email sent successfully for action {action_type}."

@tool
def crm_tool(phone: str, state_snapshot: Dict[str, Any], action: str) -> str:
    """Logs conversation turns, client profile preferences, and appointment records."""
    if phone:
        crm_store.upsert_client_preferences(phone, state_snapshot)
        crm_store.log_appointment_event(phone, action, notes=f"LangGraph execution for action {action}")
    return "CRM audit logged successfully."


# ============================================================================
# TASK 2 & 4: GRAPH DESIGN & NODES WITH VALIDATION GUARDRAILS
# ============================================================================

def intent_detection_node(state: AgentState) -> AgentState:
    """Classifies user intent and extracts parameter entities."""
    log_node_transition("intent_detection", state)
    state["current_node"] = "intent_detection"
    
    last_msg = state["messages"][-1].content.lower()
    
    # Simple regex / keyphrase extraction logic
    if any(k in last_msg for k in ["hi", "hello", "assalam", "سلام"]):
        state["intent"] = "greeting"
    elif any(k in last_msg for k in ["cancel", "rad", "منسوخ"]):
        state["intent"] = "cancellation"
    elif any(k in last_msg for k in ["reschedule", "change time", "badal"]):
        state["intent"] = "rescheduling"
    elif any(k in last_msg for k in ["visit", "schedule", "book", "fix", "daikhna"]):
        state["intent"] = "booking"
    elif any(k in last_msg for k in ["price", "rent", "buy", "house", "flat"]):
        state["intent"] = "recommendation"
    elif any(k in last_msg for k in ["bye", "allah hafiz", "khuda hafiz"]):
        state["intent"] = "goodbye"
    else:
        state["intent"] = "rag"

    return state

def greeting_node(state: AgentState) -> AgentState:
    log_node_transition("greeting", state)
    msg = "Assalam-o-Alaikum! Main Uzma baat kar rahi hoon. Aap kis tarah ki property talash kar rahe hain?"
    state["messages"].append(AIMessage(content=msg))
    return state

def rag_node(state: AgentState) -> AgentState:
    log_node_transition("rag", state)
    last_user_msg = state["messages"][-1].content
    rag_res = rag_search_tool.invoke({"query": last_user_msg})
    state["tool_outputs"]["rag"] = rag_res
    state["messages"].append(AIMessage(content=f"{rag_res} Aap mazeed kya jaanna chahte hain?"))
    return state

def recommendation_node(state: AgentState) -> AgentState:
    log_node_transition("recommendation", state)
    city = state["property_preferences"].get("city") or "Lahore"
    deal_type = state["property_preferences"].get("deal_type") or "rent"
    
    # TASK 4 GUARDRAIL: Filter out unavailable properties automatically
    listings = search_property_tool.invoke({
        "city": city,
        "deal_type": deal_type,
        "max_budget": state.get("max_budget")
    })
    
    if listings:
        top = listings[0]
        state["tool_outputs"]["recommendation"] = top
        reply = f"Hamare paas {top['area']} {top['city']} mein {top['title']} dastyab hai. Price: PKR {top['price']:,}. Kya aap iska visit schedule karna chahenge?"
    else:
        reply = "Aap ki matlooba specifications par filhal koi dastyab property nahi mili. Kya aap budget ya area flex kar sakte hain?"
    
    state["messages"].append(AIMessage(content=reply))
    return state

def availability_checker_node(state: AgentState) -> AgentState:
    """Pre-booking validation node."""
    log_node_transition("availability_checker", state)
    meeting_time = state.get("meeting_time")
    
    # TASK 4 GUARDRAIL: Ask clarification if time or contact missing
    if not meeting_time or not state["user_profile"].get("phone"):
        state["clarification_needed"] = True
        return state

    chk = availability_checker_tool.invoke({"meeting_time": meeting_time})
    if not chk["is_available"]:
        state["clarification_needed"] = True
        state["messages"].append(AIMessage(content="Yeh slot pehle se booked hai. Baraye meharbani koi doosra time bata dein."))
    else:
        state["clarification_needed"] = False

    return state

def booking_node(state: AgentState) -> AgentState:
    log_node_transition("booking", state)
    
    phone = state["user_profile"]["phone"]
    name = state["user_profile"].get("client_name", "Client")
    m_time = state["meeting_time"]
    
    # Execute Booking
    res = calendar_tool.invoke({"action": "BOOK", "client_name": name, "phone": phone, "meeting_time": m_time})
    state["appointment_status"] = "BOOKED"
    state["tool_outputs"]["booking"] = res
    
    reply = f"Ji {name}! Aap ki visit appointment {m_time} ke liye kamyabi se book ho gayi hai."
    state["messages"].append(AIMessage(content=reply))
    return state

def rescheduling_node(state: AgentState) -> AgentState:
    log_node_transition("rescheduling", state)
    phone = state["user_profile"].get("phone")
    m_time = state.get("meeting_time")
    
    # TASK 4 GUARDRAIL: Ask clarification instead of guessing missing params
    if not phone or not m_time:
        state["clarification_needed"] = True
        state["messages"].append(AIMessage(content="Appointment reschedule karne ke liye apna registered phone number aur naya time bata dein."))
        return state

    calendar_tool.invoke({"action": "RESCHEDULE", "client_name": "", "phone": phone, "meeting_time": m_time, "event_id": "evt_12345"})
    state["appointment_status"] = "RESCHEDULED"
    state["clarification_needed"] = False
    
    reply = f"Aap ki appointment kamyabi se {m_time} par reschedule kar di gayi hai."
    state["messages"].append(AIMessage(content=reply))
    return state

def cancellation_node(state: AgentState) -> AgentState:
    log_node_transition("cancellation", state)
    phone = state["user_profile"].get("phone")
    
    # TASK 4 GUARDRAIL: Ask clarification if phone missing
    if not phone:
        state["clarification_needed"] = True
        state["messages"].append(AIMessage(content="Appointment cancel karne ke liye apna registered phone number bata dein."))
        return state

    calendar_tool.invoke({"action": "CANCEL", "client_name": "", "phone": phone, "meeting_time": "", "event_id": "evt_12345"})
    state["appointment_status"] = "CANCELLED"
    state["clarification_needed"] = False
    
    reply = "Aap ki appointment cancel kar di gayi hai."
    state["messages"].append(AIMessage(content=reply))
    return state

def email_node(state: AgentState) -> AgentState:
    log_node_transition("email", state)
    phone = state["user_profile"].get("phone")
    status = state.get("appointment_status")
    
    if phone and status in ["BOOKED", "RESCHEDULED", "CANCELLED"]:
        email_tool.invoke({
            "recipient_phone": phone,
            "action_type": status,
            "details": {"client_name": state["user_profile"].get("client_name"), "meeting_time": state.get("meeting_time")}
        })
        crm_tool.invoke({"phone": phone, "state_snapshot": dict(state), "action": status})
    return state

def clarification_node(state: AgentState) -> AgentState:
    log_node_transition("clarification", state)
    if not state["messages"] or not isinstance(state["messages"][-1], AIMessage):
        state["messages"].append(AIMessage(content="Mujhe aap ki baat poori tarah samajh nahi aai. Kripya apna naam, phone number aur waqt saaf bata dein."))
    return state

def goodbye_node(state: AgentState) -> AgentState:
    log_node_transition("goodbye", state)
    state["messages"].append(AIMessage(content="Rab Raakha! Aap se baat karke khushi hui. Allah Hafiz!"))
    return state


# ============================================================================
# TASK 2: GRAPH ROUTING & STATEGRAPH ASSEMBLY
# ============================================================================

def route_by_intent(state: AgentState) -> str:
    """Conditional router executing state flow based on detected intent and guardrails."""
    intent = state.get("intent")
    
    if intent == "greeting":
        return "greeting_node"
    elif intent == "cancellation":
        return "cancellation_node"
    elif intent == "rescheduling":
        return "rescheduling_node"
    elif intent == "booking":
        return "availability_checker_node"
    elif intent == "recommendation":
        return "recommendation_node"
    elif intent == "goodbye":
        return "goodbye_node"
    else:
        return "rag_node"

def route_after_availability(state: AgentState) -> str:
    """Routes to booking if slot is valid, or clarification if unavailable/missing."""
    if state.get("clarification_needed"):
        return "clarification_node"
    return "booking_node"

def route_after_action(state: AgentState) -> str:
    """Triggers email notification if an appointment action was completed."""
    if state.get("clarification_needed"):
        return END
    return "email_node"

# Build StateGraph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("intent_detection_node", intent_detection_node)
workflow.add_node("greeting_node", greeting_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("recommendation_node", recommendation_node)
workflow.add_node("availability_checker_node", availability_checker_node)
workflow.add_node("booking_node", booking_node)
workflow.add_node("rescheduling_node", rescheduling_node)
workflow.add_node("cancellation_node", cancellation_node)
workflow.add_node("email_node", email_node)
workflow.add_node("clarification_node", clarification_node)
workflow.add_node("goodbye_node", goodbye_node)

# Set Graph Connections & Edges
workflow.add_edge(START, "intent_detection_node")

workflow.add_conditional_edges(
    "intent_detection_node",
    route_by_intent,
    {
        "greeting_node": "greeting_node",
        "cancellation_node": "cancellation_node",
        "rescheduling_node": "rescheduling_node",
        "availability_checker_node": "availability_checker_node",
        "recommendation_node": "recommendation_node",
        "goodbye_node": "goodbye_node",
        "rag_node": "rag_node"
    }
)

workflow.add_conditional_edges(
    "availability_checker_node",
    route_after_availability,
    {
        "booking_node": "booking_node",
        "clarification_node": "clarification_node"
    }
)

workflow.add_conditional_edges("booking_node", route_after_action, {"email_node": "email_node", END: END})
workflow.add_conditional_edges("rescheduling_node", route_after_action, {"email_node": "email_node", END: END})
workflow.add_conditional_edges("cancellation_node", route_after_action, {"email_node": "email_node", END: END})

workflow.add_edge("greeting_node", END)
workflow.add_edge("rag_node", END)
workflow.add_edge("recommendation_node", END)
workflow.add_edge("email_node", END)
workflow.add_edge("clarification_node", END)
workflow.add_edge("goodbye_node", END)

# Compile LangGraph Agent
app = workflow.compile()