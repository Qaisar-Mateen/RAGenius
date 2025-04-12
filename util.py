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
    
    for chunk in chat_completion:
        if not chunk.choices[0].delta.content:
            continue
            
        # Get the current chunk and add it to full text
        chunk_text = chunk.choices[0].delta.content
        full_text += chunk_text
        
        # Extract thinking part if present
        think_pattern = r'<think>.*?</think>'
        think_matches = re.findall(think_pattern, full_text, re.DOTALL)
        
        if think_matches:
            # Get the last complete thinking block
            last_think = think_matches[-1]
            current_thinking = last_think[7:-8].strip()  # Remove <think> and </think> tags
            
            # Remove all thinking blocks from content
            current_content = re.sub(think_pattern, '', full_text, flags=re.DOTALL).strip()
        else:
            # Check if we have a partial thinking section in progress
            think_start = full_text.find("<think>")
            if think_start >= 0:
                # We have a thinking section in progress
                current_content = full_text[:think_start].strip()
                # Extract only the thinking part
                current_thinking = full_text[think_start+7:].strip()
            else:
                # No thinking section
                current_content = full_text.strip()
                current_thinking = None
        
        # Add a small delay for slower generation
        time.sleep(0.05)
        
        # Only yield if thinking content has changed to avoid repetition
        if current_thinking != previous_thinking or current_thinking is None:
            previous_thinking = current_thinking
            yield {
                "thinking": current_thinking,
                "content": current_content,
                "chunk": chunk_text
            }