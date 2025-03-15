import streamlit as st
from util import *
from groq import Groq

st.set_page_config(page_icon="🧑‍🏫", layout="wide", page_title="RAGenius")

st.markdown("""
<style>
.thinking-container {
    max-height: 300px;
    overflow-y: auto;
    border-radius: 0.5rem;
    padding: 1rem;
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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            # For assistant messages, show thinking first then content
            if message["role"] == "assistant" and message.get("thinking"):
                with st.expander("See thinking process", expanded=False):
                    st.markdown(f'<div class="thinking-container">{message["thinking"]}</div>', unsafe_allow_html=True)
            
            # Display the message content
            st.markdown(message["content"])

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
                # First create elements for thinking content
                thinking_expander = st.expander("See thinking process", expanded=False)
                thinking_container = thinking_expander.empty()
                
                # Then create element for response content
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
        
        scroll_anchor.markdown("")
            
if __name__ == "__main__":
    main()