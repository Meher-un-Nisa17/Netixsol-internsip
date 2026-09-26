
import streamlit as st
import requests
import uuid


# ============================================================
# Configuration
# ============================================================

API_URL = "http://127.0.0.1:8000/chat"


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AFL Assistant",
    page_icon="🏉",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
    }

    .main-header h1 {
        font-size: 2.4rem;
        margin-bottom: 0.2rem;
    }

    .main-header p {
        color: #777;
        font-size: 1rem;
        margin-top: 0;
    }

    /* Prediction card */
    .prediction-card {
        border: 1px solid #ddd;
        border-radius: 14px;
        padding: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .prediction-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .prediction-winner {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0.4rem 0;
    }

    .prediction-probability {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Metadata */
    .metadata {
        font-size: 0.82rem;
        color: #777;
        margin-top: 0.5rem;
    }

    /* Quick question cards */
    .quick-question {
        padding: 0.5rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Session state
# ============================================================

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🏉 AFL Assistant</h1>
        <p>
            Your intelligent assistant for AFL teams, players,
            matches, statistics, history, rules, and predictions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("## 🏉 AFL Assistant")

    st.caption(
        "Ask questions about Australian Football League data "
        "and match predictions."
    )

    st.divider()

    st.markdown("### 💡 Quick Questions")

    quick_questions = [
        "What is Geelong's record against Essendon?",
        "How many wins did Geelong have against Essendon?",
        "Predict North Melbourne vs Fitzroy on 1983-04-04.",
        "What AFL teams are in the dataset?"
    ]

    for question in quick_questions:
        if st.button(
            question,
            use_container_width=True,
            key=f"quick_{question}"
        ):
            st.session_state.pending_question = question

    st.divider()

    st.markdown("### 🆔 Conversation")

    st.caption("Conversation ID")
    st.code(
        st.session_state.conversation_id,
        language="text"
    )

    if st.button(
        "🗑️ New Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())

        if "pending_question" in st.session_state:
            del st.session_state.pending_question

        st.rerun()

    st.divider()

    st.caption("Powered by LangGraph + FastAPI + Streamlit")


# ============================================================
# Display previous messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display prediction metadata if available
        if message["role"] == "assistant":

            prediction = message.get("prediction_metadata")
            metadata = message.get("metadata")

            if prediction:
                st.markdown("---")

                st.markdown("### 🏆 Match Prediction")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Match**")
                    st.write(
                        f"{prediction.get('home_team')} "
                        f"vs "
                        f"{prediction.get('away_team')}"
                    )

                    st.markdown("**Date**")
                    st.write(prediction.get("match_date"))

                with col2:
                    st.markdown("**Predicted Winner**")
                    st.write(
                        prediction.get("predicted_winner")
                    )

                    probability = prediction.get("probability")

                    if probability is not None:
                        st.markdown("**Model Probability**")
                        st.write(
                            f"{probability * 100:.1f}%"
                        )

            if metadata:
                st.caption(
                    f"Route: {metadata.get('route', 'N/A')} • "
                    f"Intent: {metadata.get('intent', 'N/A')} • "
                    f"Latency: {metadata.get('latency_ms', 'N/A')} ms"
                )


# ============================================================
# Determine current user message
# ============================================================

user_message = None

if "pending_question" in st.session_state:

    user_message = st.session_state.pending_question
    del st.session_state.pending_question

else:

    user_message = st.chat_input(
        "Ask an AFL question..."
    )


# ============================================================
# Process user request
# ============================================================

if user_message:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    with st.chat_message("user"):
        st.markdown(user_message)


    # --------------------------------------------------------
    # Call FastAPI
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "message": user_message,
                        "conversation_id":
                            st.session_state.conversation_id
                    },
                    timeout=30
                )

                # ------------------------------------------------
                # Successful API response
                # ------------------------------------------------

                if response.status_code == 200:

                    response_data = response.json()

                    assistant_response = response_data.get(
                        "response",
                        "No response returned."
                    )

                    st.markdown(assistant_response)


                    # ------------------------------------------------
                    # Prediction metadata
                    # ------------------------------------------------

                    prediction = response_data.get(
                        "prediction_metadata"
                    )

                    if prediction:

                        st.markdown("---")

                        st.markdown(
                            "### 🏆 Match Prediction"
                        )

                        col1, col2 = st.columns(2)

                        with col1:

                            st.markdown("**Match**")

                            st.write(
                                f"{prediction.get('home_team')} "
                                f"vs "
                                f"{prediction.get('away_team')}"
                            )

                            st.markdown("**Date**")

                            st.write(
                                prediction.get("match_date")
                            )

                        with col2:

                            st.markdown(
                                "**Predicted Winner**"
                            )

                            st.write(
                                prediction.get(
                                    "predicted_winner"
                                )
                            )

                            probability = prediction.get(
                                "probability"
                            )

                            if probability is not None:

                                st.markdown(
                                    "**Model Probability**"
                                )

                                st.write(
                                    f"{probability * 100:.1f}%"
                                )


                    # ------------------------------------------------
                    # Technical metadata
                    # ------------------------------------------------

                    route = response_data.get("route")
                    intent = response_data.get("intent")
                    latency = response_data.get("latency_ms")

                    st.caption(
                        f"Route: {route or 'N/A'} • "
                        f"Intent: {intent or 'N/A'} • "
                        f"Latency: "
                        f"{latency if latency is not None else 'N/A'} ms"
                    )


                    # ------------------------------------------------
                    # Save assistant response
                    # ------------------------------------------------

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "prediction_metadata": prediction,
                        "metadata": {
                            "route": route,
                            "intent": intent,
                            "latency_ms": latency
                        }
                    })


                # ------------------------------------------------
                # API returned an error
                # ------------------------------------------------

                else:

                    st.error(
                        f"API request failed "
                        f"(HTTP {response.status_code})."
                    )

                    st.caption(
                        "Please check that the FastAPI server "
                        "is running."
                    )


            # ----------------------------------------------------
            # Connection error
            # ----------------------------------------------------

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the AFL API."
                )

                st.caption(
                    "Make sure the FastAPI server is running "
                    "on port 8000."
                )


            # ----------------------------------------------------
            # Timeout
            # ----------------------------------------------------

            except requests.exceptions.Timeout:

                st.error(
                    "The AFL API took too long to respond."
                )

                st.caption(
                    "Please try the request again."
                )


            # ----------------------------------------------------
            # Other errors
            # ----------------------------------------------------

            except Exception as e:

                st.error(
                    "An unexpected error occurred."
                )

                st.caption(str(e))
