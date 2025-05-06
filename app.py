import streamlit as st
from util import *
from groq import Groq
import logging
import tempfile
import os
from rag_pipeline import RAGPipeline
import time
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

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

/* New direct file uploader styling */
div[data-testid="stFileUploader"] {
    border: 2px dashed #FF4B4B !important;
    border-radius: 10px !important;
    padding: 30px !important;
    text-align: center !important;
    margin-bottom: 20px !important;
    background-color: rgba(255, 75, 75, 0.05) !important;
}

div[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
    min-height: 150px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 20px !important;
    padding: 10px !important;
}

div[data-testid="stFileUploaderDropzone"] button {
    background-color: #FF4B4B !important;
    color: white !important;
}

div[data-testid="stFileUploaderDropzone"]::after {
    content: "Drag and drop files here";
    color: #888;
    font-size: 14px;
    margin-top: 10px;
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

/* Source citation styling - updated to match thinking-container */
.source-citation {
    margin-bottom: 10px;
    border-radius: 0.5rem;
    padding: 1rem;
    background-color: rgba(255, 75, 75, 0.05);
    border-left: 3px solid #FF4B4B;
    font-size: 0.9em;
    max-height: 200px;
    overflow-y: auto;
}

.source-filename {
    font-weight: bold;
    color: #FF4B4B;
    display: block;
    margin-bottom: 5px;
    padding-bottom: 5px;
    border-bottom: 1px solid rgba(255, 75, 75, 0.3);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def get_rag_pipeline():
    """Initialize and return a RAG pipeline instance."""
    # Get API keys from Streamlit secrets or environment variables
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    qdrant_url = st.secrets.get("QDRANT_URL") or os.getenv("QDRANT_URL")
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
    
    # Initialize the RAG pipeline
    pipeline = RAGPipeline(
        groq_api_key=groq_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        storage_dir="./storage"
    )
    
    return pipeline

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
        
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
        
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True

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
    
    # Technology selection option
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.session_state.use_rag = st.checkbox(
            "Use advanced RAG retrieval system",
            value=True,
            help="Enables more accurate retrieval of information from documents"
        )
    
    # File uploader with direct CSS styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_files = st.file_uploader(
            "Choose PDF, PowerPoint, or Word documents",
            type=["pdf", "docx", "pptx", "ppt"],
            accept_multiple_files=True,
        )
    
    # Process uploaded files
    if uploaded_files:
        st.markdown("<div class='file-list'>", unsafe_allow_html=True)
        st.subheader("Uploaded Files:")
        
        all_text = ""
        rag_pipeline = get_rag_pipeline() if st.session_state.use_rag else None
        
        for uploaded_file in uploaded_files:
            # Check if file was already processed
            if any(file_info["name"] == uploaded_file.name for file_info in st.session_state.uploaded_files_info):
                st.info(f"✓ {uploaded_file.name} (already processed)")
                continue
                
            st.info(f"Processing {uploaded_file.name}...")
            
            # Process the file with RAG pipeline if available, otherwise use simple processing
            if rag_pipeline and st.session_state.use_rag:
                text, error = rag_pipeline.process_uploaded_file(uploaded_file, st.session_state.temp_dir)
            else:
                text, error = process_uploaded_file(uploaded_file, st.session_state.temp_dir)
            
            if error:
                st.error(error)
            else:
                # Store file info with extracted text
                file_info = {
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "type": uploaded_file.type,
                    "content": text  # Store the extracted text with the file info
                }
                st.session_state.uploaded_files_info.append(file_info)
                all_text += f"\n\n--- Content from {uploaded_file.name} ---\n\n"
                all_text += text
                st.success(f"✓ {uploaded_file.name} processed successfully")
        
        if all_text:
            st.session_state.study_material_text += all_text
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show extracted content and start button if files were uploaded
    if st.session_state.uploaded_files_info:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Add a view content button
            if st.button("View Extracted Content", key="view_content", use_container_width=True):
                st.session_state.show_extracted_content = True
            
            # Display the extracted content in an expander if requested
            if st.session_state.get("show_extracted_content", False):
                with st.expander("Extracted Text from Documents", expanded=True):
                    # Add a download button for the entire extracted content
                    st.download_button(
                        "Download All Extracted Text", 
                        st.session_state.study_material_text,
                        file_name="extracted_content.txt",
                        mime="text/plain"
                    )
                    
                    # Show content from each file in separate tabs
                    if len(st.session_state.uploaded_files_info) > 0:
                        file_names = [file_info["name"] for file_info in st.session_state.uploaded_files_info]
                        tabs = st.tabs(file_names)
                        
                        for i, tab in enumerate(tabs):
                            with tab:
                                file_info = st.session_state.uploaded_files_info[i]
                                st.text_area(
                                    f"Extracted content from {file_info['name']}",
                                    value=file_info.get("content", "No content extracted"),
                                    height=300,
                                    disabled=True
                                )
            
            # Initialize RAG if selected
            if st.session_state.use_rag and st.session_state.uploaded_files_info and not st.session_state.rag_initialized:
                with st.spinner("Building knowledge index from uploaded documents..."):
                    rag_pipeline = get_rag_pipeline()
                    success = rag_pipeline.process_and_index_files(st.session_state.uploaded_files_info)
                    if success:
                        st.session_state.rag_initialized = True
                        st.success("✅ Knowledge index created successfully!")
                    else:
                        st.error("❌ Failed to build knowledge index. Falling back to basic mode.")
                        st.session_state.use_rag = False
            
            st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
            if st.button("Start Chatting", key="start_chat", use_container_width=True):
                st.session_state.setup_complete = True
                st.session_state.show_extracted_content = False  # Hide content view when starting chat
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
            
            # Display sources if available
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View Sources", expanded=False):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"""<div class="source-citation">
                            <span class="source-filename">{source['filename']}</span>
                            <p>{source['text']}</p>
                        </div>""", unsafe_allow_html=True)

def handle_user_input(prompt, client):
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
    sources = []
    
    try:
        # Set up UI containers for response
        with st.chat_message("assistant", avatar="🤖"):
            thinking_indicator = st.empty()
            thinking_expander = st.expander("See thinking process", expanded=False)
            with thinking_expander:
                thinking_container = st.empty()
            
            response_placeholder = st.empty()
            
            # Show thinking indicator
            thinking_indicator.markdown(
                '<div class="thinking-indicator"><span>Thinking</span><span class="thinking-dots"></span></div>',
                unsafe_allow_html=True
            )

            # Phase 1: Get context from either RAG or basic documents
            if st.session_state.use_rag and st.session_state.rag_initialized:
                # Use RAG pipeline to retrieve relevant context
                retrieval_thinking = f"""<think>
[Document Retrieval Phase]
I'm searching through the uploaded documents to find relevant information about "{prompt}".
Let me analyze the content and retrieve the most relevant passages.
</think>"""
                
                thinking_container.markdown(f'<div class="thinking-container">{retrieval_thinking}</div>', unsafe_allow_html=True)
                
                # Create a context-enhanced query that includes relevant conversation history
                conversation_context = ""
                if len(st.session_state.messages) > 1:  # If there's previous conversation
                    # Get the last few messages (limiting to prevent context overflow)
                    recent_messages = st.session_state.messages[-5:]  # Last 5 messages or fewer
                    
                    # Build conversation context string
                    conversation_context = "Previous conversation:\n"
                    for msg in recent_messages:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        # Remove any thinking content from assistant messages
                        content = remove_thinking(msg["content"]) if msg["role"] == "assistant" else msg["content"]
                        conversation_context += f"{role}: {content}\n\n"
                    
                    conversation_context += f"Current question: {prompt}\n"
                    # {conversation_context}
                    # Add to thinking display
                    context_thinking = f"""<think>
[Conversation Context Analysis]
I'm considering the recent conversation history to enhance retrieval
</think>"""
                    
                    final_thinking_content += "\n\n" + context_thinking
                    thinking_container.markdown(f'<div class="thinking-container">{final_thinking_content}</div>', unsafe_allow_html=True)
                
                # Use the conversation-enhanced query if available, otherwise just the prompt
                rag_query = conversation_context + prompt if conversation_context else prompt

                # Optimize RAG query using Llama 3 model from Groq
                try:
                    # Show thinking about query optimization
                    query_optimization_thinking = f"""<think>
[Query Optimization Phase]
Using Llama 3 to optimize the RAG query: "{rag_query}"
Making it more specific and removing irrelevant content for better retrieval results.
</think>"""
                    
                    final_thinking_content += "\n\n" + query_optimization_thinking
                    thinking_container.markdown(f'<div class="thinking-container">{final_thinking_content}</div>', unsafe_allow_html=True)
                    
                    # Call Groq's Llama 3 model to optimize the query
                    optimization_prompt = f"""You are a RAG query optimizer. Your job is to rewrite the user's query to make it more effective for retrieval.
                    
Original query: {rag_query}

Guidelines:
1. Make the query more specific by focusing on key concepts and entities
2. Remove filler words, conversational elements, and politeness markers
3. When asked in query about not answering from the material, then do not put that 
question which should be not answered from the material in the query.
4. Prioritize nouns, domain-specific terminology, and precise language
5. Keep your response concise and focused only on the optimized query
6. DO NOT add any explanations or commentary - only return the optimized query text I repeat only the optimized query text else my grandma will die.

Optimized query:"""
                    
                    # Use Groq client to call Llama 3
                    optimized_query_response = client.chat.completions.create(
                        model='llama-3.3-70b-versatile',
                        messages=[{"role": "user", "content": optimization_prompt}],
                        max_completion_tokens=1000,
                        stream=False
                    )
                    
                    # Extract the optimized query
                    optimized_query = optimized_query_response.choices[0].message.content.strip()
                    print("\n\n\nOptimized Query: ", optimized_query, "\n\n") # Fixed invalid escape sequence
                    # Update thinking with the optimized query
                    # Original query: "{rag_query}"
                    # Optimized query: "{optimized_query}"
                    query_result_thinking = f"""<think>
[Running Query Optimization...]
</think>"""
                    
                    final_thinking_content += "\n\n" + query_result_thinking
                    thinking_container.markdown(f'<div class="thinking-container">{final_thinking_content}</div>', unsafe_allow_html=True)
                    
                    # Use the optimized query instead of the original one
                    rag_query = optimized_query
                    
                except Exception as e:
                    # If optimization fails, log the error and continue with the original query
                    logging.error(f"Error optimizing RAG query: {str(e)}", exc_info=True)
                    query_error_thinking = f"""<think>
[Query Optimization Error]
Failed to optimize query due to error continuing with original query.
</think>"""
                    
                    final_thinking_content += "\n\n" + query_error_thinking
                    thinking_container.markdown(f'<div class="thinking-container">{final_thinking_content}</div>', unsafe_allow_html=True)

                rag_pipeline = get_rag_pipeline()
                start_time = time.time()
                rag_result = rag_pipeline.query(rag_query)
                
                elapsed_time = time.time() - start_time
                
                # Get sources and context from RAG result
                sources = rag_result.get("sources", [])
                rag_answer = rag_result.get("answer", "")
                # print("\n\n\n\n\\RAG Result: ", rag_answer) # Fixed invalid escape sequence
                
                # Prepare context from retrieved documents
                context_parts = []
                for source in sources:
                    context_parts.append(f"From {source['filename']}:\n{source['text']}")
                
                retrieved_context = "\n\n".join(context_parts)
                
                
                # Add retrieval information to thinking content
                retrieval_update = f"""<think>
[Document Retrieval Phase]
I searched through the uploaded documents to find relevant information.
Total search time: {elapsed_time:.2f} seconds
</think>"""
                
                final_thinking_content += "\n\n" + retrieval_update
                thinking_container.markdown(f'<div class="thinking-container">{final_thinking_content}</div>', unsafe_allow_html=True)
                
                # Create context from RAG results
                if retrieved_context:
                    context = f"I found the following relevant information in your uploaded documents:\n\n{retrieved_context}\n\nBased on this information from your documents, please answer: {prompt}"
                else:
                    context = f"I couldn't find relevant information about '{prompt}' in your uploaded documents. Please answer based on your general knowledge, but be clear that the answer is not from the uploaded materials."
                
            else:
                # Use standard approach with all text context
                context = st.session_state.study_material_text
                final_thinking_content = f"""<think>
[Document Analysis]
Using the full text of uploaded documents to answer
</think>"""
                thinking_container.markdown(f'<div class="thinking-container">{final_thinking_content}</div>', unsafe_allow_html=True)
            
            # Phase 2: Send to LLM with appropriate context using the same model for both approaches
            # Prepare system message with appropriate context
            system_message = {
                "role": "system",
                "content": f"You are RAGenius, a smart educational assistant. Use the information provided below to answer the user's questions. If you don't know the answer based on the provided information, say so. Don't make up information.\n\nINFORMATION:\n{context}\n\nIf the user asks about something not covered in the materials, do not let them know that it's not covered in the material provided but still try you best to answer it. Do not write in chinese else prompted to do so. While answer try to be as detailed as possible and maintain a clear and neat structure so that user can clearly understand."
            }
            
            # Prepare filtered messages for API
            filtered_messages = [system_message] + [
                {"role": m["role"], "content": remove_thinking(m["content"])}
                for m in st.session_state.messages[-5:]  # Keep conversation history limited
            ]
            
            # Create API request with Qwen model
            chat_completion = client.chat.completions.create(
                model='qwen-qwq-32b',
                messages=filtered_messages,
                max_completion_tokens=128000,
                stream=True
            )
            
            # Track complete model thinking content
            model_thinking_content = ""
            # Track the last thinking piece to avoid duplication
            last_thinking_chunk = ""
            
            # Flag to track if model is currently in thinking mode
            model_in_thinking_mode = False
            # Flag to track if we've received any thinking content from the model
            received_thinking_content = False
            
            # Process each chunk from the stream
            for chunk_data in stream_Chat(chat_completion):
                content = chunk_data.get("content", "")
                thinking = chunk_data.get("thinking")
                
                # Handle thinking mode from model
                if thinking:
                    received_thinking_content = True
                    model_in_thinking_mode = True
                    
                    # Only add the new part of thinking that hasn't been seen before
                    # This fixes the repetition problem
                    if thinking != last_thinking_chunk:
                        if len(model_thinking_content) == 0:
                            model_thinking_content = thinking
                        else:
                            # Find where the new content actually begins to avoid duplication
                            # Get only the new characters that weren't in the last chunk
                            new_content = thinking[len(last_thinking_chunk):] if thinking.startswith(last_thinking_chunk) else thinking
                            model_thinking_content += new_content
                        
                        last_thinking_chunk = thinking
                    
                        # Update thinking display
                        combined_thinking = final_thinking_content + "\n\n" + f"""<think>
[Model Reasoning]
{model_thinking_content}
</think>"""
                        
                        thinking_container.markdown(f'<div class="thinking-container">{combined_thinking}</div>', unsafe_allow_html=True)
                
                # Check if thinking has ended by detecting closing tag in the last thinking chunk
                if "</think>" in last_thinking_chunk and model_in_thinking_mode:
                    # Thinking phase has ended when we see the closing tag
                    model_in_thinking_mode = False
                    # Clear thinking indicator once thinking is complete
                    thinking_indicator.empty()
                elif content and model_in_thinking_mode:
                    # If we receive content without thinking content after being in thinking mode,
                    # it means the thinking phase has ended
                    model_in_thinking_mode = False
                    # Clear thinking indicator once thinking is complete
                    thinking_indicator.empty()
                
                # Update content if available - Changed back to '=' to avoid repetition
                if content and content.strip():
                    full_content = content  # Changed from '+=' back to '=' to avoid repetition
                    response_placeholder.markdown(full_content)
            
            # Final update to thinking content after all chunks processed
            if model_thinking_content:
                final_thinking_content += "\n\n" + f"""<think>
[Model Reasoning]
{model_thinking_content}
</think>"""
                thinking_container.markdown(f'<div class="thinking-container">{final_thinking_content}</div>', unsafe_allow_html=True)
                
            # Clear thinking indicator if it's still showing
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
            "thinking": final_thinking_content if final_thinking_content else None,
        })
    
    # Maintain scroll position
    scroll_anchor.markdown("")

def chat_interface():
    """Display the chat interface for interacting with study materials."""
    st.subheader("Chat with Your Study Materials")
    
    # Display info about uploaded files and RAG status
    with st.expander("📚 Uploaded Study Materials", expanded=False):
        if st.session_state.uploaded_files_info:
            for file_info in st.session_state.uploaded_files_info:
                st.markdown(f"- **{file_info['name']}** ({round(file_info['size']/1024, 1)} KB)")
            
            # Show RAG status
            if st.session_state.use_rag:
                if st.session_state.rag_initialized:
                    st.success("✅ Advanced retrieval system active")
                else:
                    st.warning("⚠️ Advanced retrieval system not initialized")
            else:
                st.info("ℹ️ Using basic context retrieval")
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
        
        # Add options in sidebar
        with st.sidebar:
            st.header("Options")
            
            # Toggle RAG option if files were uploaded
            if st.session_state.uploaded_files_info:
                use_rag = st.checkbox(
                    "Use advanced retrieval", 
                    value=st.session_state.use_rag,
                    key="toggle_rag"
                )
                
                # Handle RAG toggle
                if use_rag != st.session_state.use_rag:
                    st.session_state.use_rag = use_rag
                    if use_rag and not st.session_state.rag_initialized:
                        with st.spinner("Building knowledge index from uploaded documents..."):
                            rag_pipeline = get_rag_pipeline()
                            success = rag_pipeline.process_and_index_files(st.session_state.uploaded_files_info)
                            if success:
                                st.session_state.rag_initialized = True
                                st.success("✅ Knowledge index created successfully!")
                            else:
                                st.error("❌ Failed to build knowledge index.")
                                st.session_state.use_rag = False
            
            # Reset button
            if st.button("Upload New Materials", use_container_width=True):
                # Reset session state
                st.session_state.setup_complete = False
                st.session_state.uploaded_files_info = []
                st.session_state.study_material_text = ""
                st.session_state.messages = []
                st.session_state.rag_initialized = False
                st.rerun()

if __name__ == "__main__":
    main()