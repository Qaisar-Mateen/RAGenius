from typing import Generator
import re
import time

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