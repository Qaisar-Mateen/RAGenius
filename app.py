import streamlit as st
from util import *
from groq import Groq

st.set_page_config(page_icon="🧑‍🏫", layout="wide", page_title="RAGenius")

# Enhanced CSS to force expanders to be closed by default
st.markdown("""
<style>
.thinking-container {
    max-height: 300px;
    overflow-y: auto;
    border-radius: 0.5rem;
    padding: 1rem;
}

/* Force expanders to be closed by default */
.streamlit-expanderHeader {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

def main():

    # st.subheader("RAGenius: Smart Answers for Smarter Learning.", divider="rainbow", anchor=False)
    st.markdown("<h4><span style='font-weight: 900; font-size:2rem; color: #FF4B4B;'>RAGenius:</span> Smart Answers for Smarter Learning.</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height: 3px; background: linear-gradient(90deg, red, orange, yellow);'></div>", unsafe_allow_html=True)
    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"],
    )

    # Initialize message history and expander states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize expander states dictionary to track which expanders are open
    if "expander_states" not in st.session_state:
        st.session_state.expander_states = {}

    # Display chat messages from history on app rerun
    for i, message in enumerate(st.session_state.messages):
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            # For assistant messages, show thinking first then content
            if message["role"] == "assistant" and message.get("thinking"):
                # Create a unique key for each expander
                expander_key = f"thinking_{i}"
                # Default to closed unless explicitly opened by user
                is_expanded = st.session_state.expander_states.get(expander_key, False)
                
                with st.expander("See thinking process", expanded=is_expanded):
                    st.markdown(f'<div class="thinking-container">{message["thinking"]}</div>', unsafe_allow_html=True)
            
            # Display the message content
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your prompt here..."):
        # Create a container at the top to maintain scroll position
        scroll_anchor = st.empty()
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)

        full_content = ""
        final_thinking_content = ""
        
        # Fetch response from Groq API
        try:
            # Filter out thinking parts before sending messages to API
            filtered_messages = [
                {
                    "role": m["role"],
                    "content": remove_thinking(m["content"])
                }
                for m in st.session_state.messages
            ]
            
            chat_completion = client.chat.completions.create(
                model='qwen-qwq-32b',
                messages=filtered_messages,
                max_completion_tokens=128000,
                stream=True
            )

            # Stream the response and handle thinking parts
            with st.chat_message("assistant", avatar="🤖"):
                # Create a debug container at the top (hidden in production)
                debug_container = st.empty()
                
                # First create the thinking expander at the top
                thinking_expander = st.expander("See thinking process", expanded=False)
                with thinking_expander:
                    thinking_container = st.empty()
                
                # Then create a separate container for the actual response
                response_placeholder = st.empty()
                
                chunk_count = 0
                
                for chunk_data in stream_Chat(chat_completion):
                    chunk_count += 1
                    content = chunk_data.get("content", "")
                    thinking = chunk_data.get("thinking")
                    
                    # Always update full_content if we have content
                    if content and content.strip():
                        full_content = content
                        # Display the content
                        response_placeholder.markdown(full_content)
                        
                    # Display thinking content if available
                    if thinking:
                        final_thinking_content = thinking
                        thinking_container.markdown(f'<div class="thinking-container">{thinking}</div>', unsafe_allow_html=True)
                    
                    # Show debug info (can be commented out in production)
                    debug_container.markdown(f"Chunk #{chunk_count} - Content length: {len(content) if content else 0}, Thinking: {'Yes' if thinking else 'No'}")
                
                # Display a message if no content was received
                if not full_content:
                    response_placeholder.markdown("*No response content was generated.*")
                    debug_container.markdown(f"⚠️ No content received after {chunk_count} chunks")
                else:
                    debug_container.markdown(f"✅ Received {chunk_count} chunks. Final content length: {len(full_content)}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}", icon="🚨")
            import traceback
            st.error(traceback.format_exc())

        # Append the full response to session_state.messages with thinking part
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_content,
            "thinking": final_thinking_content if final_thinking_content else None
        })
        
        scroll_anchor.markdown("")
            
if __name__ == "__main__":
    main()