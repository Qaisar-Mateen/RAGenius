from typing import Generator
import re

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
    accumulated_text = ""
    
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            chunk_content = chunk.choices[0].delta.content
            accumulated_text += chunk_content
            
            # Check if we have complete thinking tags
            thinking, current_content = extract_thinking(accumulated_text)
            
            yield {"thinking": thinking, "content": current_content, "chunk": chunk_content}