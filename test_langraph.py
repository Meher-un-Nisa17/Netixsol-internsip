# test_langgraph.py

from langraph_agent import app, AgentState
from langchain_core.messages import HumanMessage

def test_agent_graph():
    # Initial Agent State
    initial_state: AgentState = {
        "messages": [HumanMessage(content="Meri appointment cancel kar dein.")],
        "user_profile": {"client_name": "Ali Khan", "phone": "03001234567"},
        "property_preferences": {"city": "Lahore", "deal_type": "rent", "property_type": "House"},
        "max_budget": 150000.0,
        "intent": None,
        "tool_outputs": {},
        "appointment_status": "NONE",
        "meeting_time": "2026-10-10 15:00",
        "clarification_needed": False,
        "current_node": "START"
    }

    print("\n--- RUNNING LANGGRAPH AGENT ORCHESTRATION ---")
    output_state = app.invoke(initial_state)

    print("\n--- FINAL AGENT STATE SNAPSHOT ---")
    print(f"Detected Intent: {output_state.get('intent')}")
    print(f"Appointment Status: {output_state.get('appointment_status')}")
    print(f"Last Assistant Response: {output_state['messages'][-1].content}")

if __name__ == "__main__":
    test_agent_graph()