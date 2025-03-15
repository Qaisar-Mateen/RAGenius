import streamlit as st
from util import *
from groq import Groq

st.set_page_config(page_icon="🧑‍🏫", layout="wide", page_title="RAGenius")

st.markdown("""
<style>
/* Light theme (default) */
.thinking-container {
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid #f0f2f6;
    border-radius: 0.5rem;
    padding: 1rem;
    background-color: #f8f9fa;
}

/* Dark theme */
[data-theme="dark"] .thinking-container {
    border: 1px solid #333;
    background-color: #262730;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

def main():

    st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"],
    )

    # Initialize chat history and selected model
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("thinking"):
                with st.expander("See thinking process", expanded=False):
                    st.markdown(f'<div class="thinking-container">{message["thinking"]}</div>', unsafe_allow_html=True)


    if prompt := st.chat_input("Enter your prompt here..."):
        # Create a container at the top to maintain scroll position
        scroll_anchor = st.empty()
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)

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
                with st.expander("See thinking process", expanded=False) as thinking_expander:
                    thinking_container = st.empty()
                
                response_container = st.empty()
                
                full_content = ""
                thinking_content = ""
                
                for chunk_data in stream_Chat(chat_completion):
                    full_content = chunk_data["content"]
                    thinking = chunk_data["thinking"]
                    
                    if thinking is not None:
                        thinking_content = thinking
                        thinking_container.markdown(f'<div class="thinking-container">{thinking_content}</div>', unsafe_allow_html=True)
                    
                    response_container.markdown(full_content)
                
                # Final response
                response_container.markdown(full_content)
                
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages with thinking part
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_content,
            "thinking": thinking_content if thinking_content else None
        })
        
        # Ensure the page stays at the current position
        scroll_anchor.markdown("")
            
if __name__ == "__main__":
    main()