import streamlit as st
from util import *
from groq import Groq
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up page configuration once
st.set_page_config(page_icon="🧑‍🏫", layout="wide", page_title="RAGenius")

# CSS optimizations - using a separate file would be even better for larger projects
st.markdown("""
<style>
.thinking-container {
    max-height: 300px;
    overflow-y: auto;
    border-radius: 0.5rem;
    padding: 1rem;
}

.streamlit-expanderHeader {
    background-color: transparent !important;
}

/* Optimized animations */
@keyframes shimmer {
    0% { background-position: -100% 0; }
    100% { background-position: 100% 0; }
}

.thinking-indicator {
    display: inline-block;
    font-weight: bold;
    position: relative;
    padding: 3px 8px;
    border-radius: 4px;
    margin-bottom: 8px;
}

.thinking-indicator span {
    background: linear-gradient(90deg, #FF4B4B, #FFA500, #FF4B4B);
    background-size: 200% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    animation: shimmer 1.5s infinite linear;
}

@keyframes dots {
    0% { content: "."; }
    33% { content: ".."; }
    66% { content: "..."; }
}

.thinking-dots::after {
    content: ".";
    animation: dots 1.2s infinite steps(1);
}
</style>
""", unsafe_allow_html=True)

# Cache API key to avoid repeated lookups
@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def main():
    # App header with brand styling
    st.markdown("<h4><span style='font-weight: 900; font-size:2rem; color: #FF4B4B;'>RAGenius:</span> Smart Answers for Smarter Learning.</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height: 3px; background: linear-gradient(90deg, red, orange, yellow);'></div>", unsafe_allow_html=True)
    
    # Get cached client
    client = get_client()

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "expander_states" not in st.session_state:
        st.session_state.expander_states = {}

    # Display chat history
    display_chat_history()

    # Handle new user input
    if prompt := st.chat_input("Enter your prompt here..."):
        handle_user_input(prompt, client)

def display_chat_history():
    """Display existing chat messages from history."""
    for i, message in enumerate(st.session_state.messages):
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant" and message.get("thinking"):
                expander_key = f"thinking_{i}"
                is_expanded = st.session_state.expander_states.get(expander_key, False)
                
                with st.expander("See thinking process", expanded=is_expanded):
                    st.markdown(f'<div class="thinking-container">{message["thinking"]}</div>', unsafe_allow_html=True)
            
            st.markdown(message["content"])

def handle_user_input(prompt, client):
    """Process user input and generate response."""
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)
    
    # Keep track of top of screen
    scroll_anchor = st.empty()
    
    # Initialize response variables
    full_content = ""
    final_thinking_content = ""
    
    try:
        # Prepare filtered messages for API
        filtered_messages = [
            {"role": m["role"], "content": remove_thinking(m["content"])}
            for m in st.session_state.messages
        ]
        
        # Create API request with optimized parameters
        chat_completion = client.chat.completions.create(
            model='qwen-qwq-32b',
            messages=filtered_messages,
            max_completion_tokens=128000,
            stream=True
        )
        
        # Process streaming response
        process_streaming_response(chat_completion, full_content, final_thinking_content)
        
    except Exception as e:
        logging.error(f"Error in chat completion: {str(e)}", exc_info=True)
        st.error(f"Error: {str(e)}", icon="🚨")
    
    # Add assistant response to history
    if full_content or final_thinking_content:  # Only add if we have content
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_content,
            "thinking": final_thinking_content if final_thinking_content else None
        })
    
    # Maintain scroll position
    scroll_anchor.markdown("")

def process_streaming_response(chat_completion, full_content, final_thinking_content):
    """Process streaming response from API with optimized rendering."""
    with st.chat_message("assistant", avatar="🤖"):
        # Set up UI containers
        thinking_indicator = st.empty()
        thinking_expander = st.expander("See thinking process", expanded=False)
        
        with thinking_expander:
            thinking_container = st.empty()
        
        response_placeholder = st.empty()
        
        is_thinking = False
        
        # Process each chunk from the stream
        for chunk_data in stream_Chat(chat_completion):
            content = chunk_data.get("content", "")
            thinking = chunk_data.get("thinking")
            
            # Handle thinking mode
            if thinking:
                if not is_thinking:  # Only set indicator when first entering thinking mode
                    is_thinking = True
                    thinking_indicator.markdown(
                        '<div class="thinking-indicator"><span>Thinking</span><span class="thinking-dots"></span></div>', 
                        unsafe_allow_html=True
                    )
                
                final_thinking_content = thinking
                thinking_container.markdown(f'<div class="thinking-container">{thinking}</div>', unsafe_allow_html=True)
            elif is_thinking and not thinking:
                is_thinking = False
                thinking_indicator.empty()
            
            # Update content if available
            if content and content.strip():
                full_content = content
                response_placeholder.markdown(full_content)
        
        # Clear thinking indicator when done
        thinking_indicator.empty()
        
        # Handle no content case
        if not full_content:
            response_placeholder.markdown("*No response content was generated.*")

if __name__ == "__main__":
    main()