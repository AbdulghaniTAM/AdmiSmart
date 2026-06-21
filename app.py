import streamlit as st
from agent import agent_executor
from llm import flush_memory
from execution_logging import run_observed_agent

# Page configuration
st.set_page_config(
    page_title="AdmiSmart - Admission Agentic AI",
    page_icon="🎓",
    layout="centered"
)

# Custom Styling (Glassmorphism & Branding colors)
st.markdown("""
    <style>
    .main { background: #fdfdfd; }
    .stTextInput>div>div>input { border-radius: 8px; }
    .triage-badge {
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .badge-green { background-color: #d4edda; color: #155724; }
    .badge-yellow { background-color: #fff3cd; color: #856404; }
    .badge-red { background-color: #f8d7da; color: #721c24; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 AdmiSmart Admissions Agent")
st.write("Welcome to the Saudi Digital Academy Admission Support portal. Ask a question regarding deadlines, transcripts, or visa inquiries.")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display classification badge if available
        if "triage" in message:
            triage = message["triage"]
            if triage == "GREEN":
                st.markdown('<span class="triage-badge badge-green">🟢 Green: Auto-Resolved</span>', unsafe_allow_html=True)
            elif triage == "YELLOW":
                st.markdown('<span class="triage-badge badge-yellow">🟡 Yellow: Staff Review Queue</span>', unsafe_allow_html=True)
            elif triage == "RED":
                st.markdown('<span class="triage-badge badge-red">🔴 Red: Escalated to Senior Officers</span>', unsafe_allow_html=True)
        st.write(message["content"])

# React to user input
if user_query := st.chat_input("Ask AdmiSmart..."):
    # Reset chat history to ensure every question has its own context (like a new email response)
    st.session_state.messages = []
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        status_placeholder.info("Analyzing query and checking resources...")
        
        try:
            # Utilize the instrumented logging execution wrapper
            telemetry_log = run_observed_agent(user_query)
            agent_response = telemetry_log["final_output"]
            
            # Remove internal <think> blocks before displaying to the user
            import re
            agent_response = re.sub(r'<think>.*?</think>\n*', '', agent_response, flags=re.DOTALL).strip()
            
            # Derive triage status using your original badge rendering patterns
            triage_status = "GREEN"
            if "Staff Review Queue" in agent_response or "routed to the Staff" in agent_response:
                triage_status = "YELLOW"
            elif "Senior Admission Officers" in agent_response or "escalated" in agent_response:
                triage_status = "RED"
                
            status_placeholder.empty()
            
            # Render the triage status badge
            if triage_status == "GREEN":
                st.markdown('<span class="triage-badge badge-green">🟢 Green: Auto-Resolved</span>', unsafe_allow_html=True)
            elif triage_status == "YELLOW":
                st.markdown('<span class="triage-badge badge-yellow">🟡 Yellow: Staff Review Queue</span>', unsafe_allow_html=True)
            elif triage_status == "RED":
                st.markdown('<span class="triage-badge badge-red">🔴 Red: Escalated to Senior Officers</span>', unsafe_allow_html=True)
                
            response_placeholder.write(agent_response)
            
            # Optional developer expander to view trace data directly in the UI
            with st.expander(f"⚙️ System Trace Metrics [ID: {telemetry_log['trace_id']}]"):
                st.json(telemetry_log)
                
            st.session_state.messages.append({
                "role": "assistant",
                "content": agent_response,
                "triage": triage_status
            })
        except Exception as e:
            status_placeholder.empty()
            st.error(f"System Error Encountered: {str(e)}")
        finally:
            # Flush context window and empty GPU/ROCm cache
            flush_memory()

