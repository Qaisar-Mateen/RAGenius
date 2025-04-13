import streamlit as st
from util import *
from groq import Groq
import logging
import tempfile
import os

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

/* Upload area styling */
.upload-area {
    border: 2px dashed #FF4B4B;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    margin-bottom: 20px;
    background-color: rgba(255, 75, 75, 0.05);
}

.file-list {
    margin-top: 20px;
    border-left: 3px solid #FF4B4B;
    padding-left: 15px;
}

.start-button {
    background-color: #FF4B4B;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.start-button:hover {
    background-color: #FF6B6B;
}
</style>
""", unsafe_allow_html=True)

# Cache API key to avoid repeated lookups
@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def initialize_session_state():
    """Initialize all required session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "expander_states" not in st.session_state:
        st.session_state.expander_states = {}
        
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = []
        
    if "study_material_text" not in st.session_state:
        st.session_state.study_material_text = ""
        
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
        
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

def upload_interface():
    """Display the file upload interface for study materials."""
    st.markdown("<h2 style='text-align: center;'>Upload Your Study Materials</h2>", unsafe_allow_html=True)
    st.markdown(
        """<p style='text-align: center; font-size: 16px; margin-bottom: 25px;'>
        Upload your study materials (PDF, PowerPoint, or Word documents) to get started. 
        RAGenius will use these materials to provide smart answers to your questions.
        </p>""", 
        unsafe_allow_html=True
    )
    
    # File uploader
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='upload-area'>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose files", 
            type=["pdf", "docx", "pptx", "ppt"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process uploaded files
    if uploaded_files:
        st.markdown("<div class='file-list'>", unsafe_allow_html=True)
        st.subheader("Uploaded Files:")
        
        all_text = ""
        for uploaded_file in uploaded_files:
            # Check if file was already processed
            if any(file_info["name"] == uploaded_file.name for file_info in st.session_state.uploaded_files_info):
                st.info(f"✓ {uploaded_file.name} (already processed)")
                continue
                
            st.info(f"Processing {uploaded_file.name}...")
            
            # Process the file
            text, error = process_uploaded_file(uploaded_file, st.session_state.temp_dir)
            
            if error:
                st.error(error)
            else:
                # Store file info
                st.session_state.uploaded_files_info.append({
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "type": uploaded_file.type
                })
                all_text += f"\n\n--- Content from {uploaded_file.name} ---\n\n"
                all_text += text
                st.success(f"✓ {uploaded_file.name} processed successfully")
        
        if all_text:
            st.session_state.study_material_text += all_text
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show start button if files were uploaded
    if st.session_state.uploaded_files_info:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
            if st.button("Start Chatting", key="start_chat", use_container_width=True):
                st.session_state.setup_complete = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

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
        # Prepare system message with study material context
        system_message = {
            "role": "system",
            "content": f"You are RAGenius, a smart educational assistant. Use the study material provided below to answer the user's questions. If you don't know the answer based on the material, say so. Don't make up information.\n\nSTUDY MATERIAL:\n{st.session_state.study_material_text}\n\nIf the user asks about something not covered in the materials, let them know you can only answer based on the uploaded study materials."
        }
        
        # Prepare filtered messages for API
        filtered_messages = [system_message] + [
            {"role": m["role"], "content": remove_thinking(m["content"])}
            for m in st.session_state.messages
        ]
        
        # Create API request
        chat_completion = client.chat.completions.create(
            model='qwen-qwq-32b',
            messages=filtered_messages,
            max_completion_tokens=128000,
            stream=True
        )
        
        # Process streaming response
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
                    if not is_thinking:
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
        
    except Exception as e:
        logging.error(f"Error in chat completion: {str(e)}", exc_info=True)
        st.error(f"Error: {str(e)}", icon="🚨")
    
    # Add assistant response to history
    if full_content or final_thinking_content:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_content,
            "thinking": final_thinking_content if final_thinking_content else None
        })
    
    # Maintain scroll position
    scroll_anchor.markdown("")

def chat_interface():
    """Display the chat interface for interacting with study materials."""
    st.subheader("Chat with Your Study Materials")
    
    # Display info about uploaded files
    with st.expander("📚 Uploaded Study Materials", expanded=False):
        if st.session_state.uploaded_files_info:
            for file_info in st.session_state.uploaded_files_info:
                st.markdown(f"- **{file_info['name']}** ({round(file_info['size']/1024, 1)} KB)")
        else:
            st.warning("No study materials uploaded.")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your study materials..."):
        handle_user_input(prompt, get_client())

def main():
    # App header with brand styling
    st.markdown("<h4><span style='font-weight: 900; font-size:2rem; color: #FF4B4B;'>RAGenius:</span> Smart Answers for Smarter Learning.</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height: 3px; background: linear-gradient(90deg, red, orange, yellow);'></div>", unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Create two branches of the app based on setup status
    if not st.session_state.setup_complete:
        upload_interface()
    else:
        # If setup is complete, show the chat interface
        chat_interface()
        
        # Add option to reset and upload new materials
        with st.sidebar:
            st.header("Options")
            if st.button("Upload New Materials", use_container_width=True):
                # Reset session state
                st.session_state.setup_complete = False
                st.session_state.uploaded_files_info = []
                st.session_state.study_material_text = ""
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()