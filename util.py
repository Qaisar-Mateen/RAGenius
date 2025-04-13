from typing import Generator
import re
import time
import os
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter

def icon(emoji: str, st):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

def extract_thinking(content: str) -> tuple:
    """Extract thinking part and actual content from a message."""
    thinking_pattern = r'<think>(.*?)</think>'
    thinking_match = re.search(thinking_pattern, content, re.DOTALL)
    
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        # Remove thinking part from the content
        actual_content = re.sub(thinking_pattern, '', content, flags=re.DOTALL).strip()
        return thinking, actual_content
    return None, content

def remove_thinking(content: str) -> str:
    """Remove thinking part from content."""
    _, clean_content = extract_thinking(content)
    return clean_content

def stream_Chat(chat_completion) -> Generator[tuple, None, None]:
    """Yield chat response content from the Groq API response with thinking parts separated."""
    full_text = ""
    current_thinking = None
    previous_thinking = None
    current_content = ""
    previous_content = ""
    in_thinking_section = False
    
    for chunk in chat_completion:
        if not chunk.choices[0].delta.content:
            continue
            
        # Get the current chunk and add it to full text
        chunk_text = chunk.choices[0].delta.content
        full_text += chunk_text
        
        # More efficient handling of thinking sections
        think_start = full_text.find("<think>") 
        think_end = full_text.find("</think>")
        
        # Complete thinking section found
        if think_start >= 0 and think_end > think_start:
            in_thinking_section = False
            current_thinking = full_text[think_start+7:think_end].strip()
            
            # Get content before and after thinking section
            before_thinking = full_text[:think_start].strip()
            after_thinking = full_text[think_end+8:].strip()
            current_content = before_thinking + " " + after_thinking if before_thinking and after_thinking else before_thinking + after_thinking
            
            # Remove processed thinking section from full_text
            full_text = full_text[:think_start] + full_text[think_end+8:]
            
        # Partial thinking section (started but not completed)
        elif think_start >= 0:
            in_thinking_section = True
            current_thinking = full_text[think_start+7:].strip()
            current_content = full_text[:think_start].strip()
        # No thinking section
        elif not in_thinking_section:
            current_content = full_text.strip()
            current_thinking = None
        
        # Add a small delay for slower generation
        time.sleep(0.05)
        
        # Only yield if content has changed to avoid redundant updates
        if (current_thinking != previous_thinking) or (current_content != previous_content):
            previous_thinking = current_thinking
            previous_content = current_content
            
            yield {
                "thinking": current_thinking,
                "content": current_content,
                "chunk": chunk_text
            }

def extract_text_from_pdf(file_path):
    """Extract text from PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    """Extract text from Word document."""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_pptx(file_path):
    """Extract text from PowerPoint presentation."""
    ppt = Presentation(file_path)
    text = ""
    for slide in ppt.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def process_uploaded_file(uploaded_file, temp_dir):
    """Process an uploaded file and extract text."""
    # Create a temporary file path
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Save uploaded file to disk temporarily
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text based on file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    try:
        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            text = extract_text_from_docx(file_path)
        elif file_extension in ['.pptx', '.ppt']:
            text = extract_text_from_pptx(file_path)
        else:
            return None, f"Unsupported file format: {file_extension}"
        
        # Remove the temporary file
        os.remove(file_path)
        return text, None
    except Exception as e:
        # Remove the temporary file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        return None, f"Error processing {uploaded_file.name}: {str(e)}"

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """Split text into chunks for processing."""
    if not text:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    return text_splitter.split_text(text)