# app.py — Professionally You (Chennai tone, clean UI, tools)
# Run locally:  uv run python app.py

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
import json, os, requests
from datetime import datetime

# ============= Setup =============
load_dotenv(override=True)

# Try to get API key from environment (works both locally and on HF Spaces)
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment or HF Space secrets.")

client = OpenAI(api_key=openai_key)

PUSHOVER_USER = os.getenv("PUSHOVER_USER", "")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
NAME = "Aravind Bethapudi"

# ============= Send Notifications =============
def push(msg: str):
    """Send Pushover notification for leads/meetings"""
    if not (PUSHOVER_USER and PUSHOVER_TOKEN):
        return
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "user": PUSHOVER_USER,
                "token": PUSHOVER_TOKEN,
                "message": msg[:1024],
                "timestamp": int(datetime.now().timestamp())
            },
            timeout=8,
        )
    except:
        pass  # Don't break the app if notification fails

# ============= Load Your Profile Data =============
def load_personal_data():
    """Load summary.txt and Profile.pdf from 'me' folder"""
    summary, linkedin = "", ""
    
    # Load summary.txt
    try:
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            summary = f.read()
    except:
        pass
    
    # Load LinkedIn PDF
    try:
        reader = PdfReader("me/Profile.pdf")
        for page in reader.pages:
            text = page.extract_text()
            if text:
                linkedin += text + "\n"
    except:
        pass
    
    return summary, linkedin

SUMMARY, LINKEDIN = load_personal_data()

# ============= Tool Functions (what happens when AI calls them) =============
def record_user_details(email: str, name: str = "Not provided", notes: str = "None"):
    """Save lead information"""
    time = datetime.now().strftime("%Y-%m-%d %H:%M")
    push(f"📧 NEW LEAD [{time}]\nName: {name}\nEmail: {email}\nNotes: {notes}")
    return {"status": "success", "message": "Details recorded! I'll be in touch soon."}

def record_unknown_question(question: str):
    """Log questions I couldn't answer"""
    time = datetime.now().strftime("%Y-%m-%d %H:%M")
    push(f"❓ UNKNOWN [{time}]\n{question}")
    return {"status": "logged", "message": "Noted! I'll research this."}

def request_meeting(name: str, email: str, preferred_time: str = "Anytime", topic: str = "General"):
    """Schedule a meeting"""
    time = datetime.now().strftime("%Y-%m-%d %H:%M")
    push(f"📅 MEETING [{time}]\nName: {name}\nEmail: {email}\nTime: {preferred_time}\nTopic: {topic}")
    return {"status": "scheduled", "message": f"Meeting request received! You'll get a calendar invite at {email}."}

# ============= Tool Definitions (tells AI when/how to use them) =============
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Use when someone wants to connect or shows interest",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Their email"},
                    "name": {"type": "string", "description": "Their name"},
                    "notes": {"type": "string", "description": "Any context about their interest"}
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Use ONLY when you truly don't know the answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question you couldn't answer"}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_meeting",
            "description": "Use when someone wants to schedule a call or meeting",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Their name"},
                    "email": {"type": "string", "description": "Their email"},
                    "preferred_time": {"type": "string", "description": "When they want to meet"},
                    "topic": {"type": "string", "description": "What they want to discuss"}
                },
                "required": ["name", "email"]
            }
        }
    }
]

# ============= Execute Tool Calls =============
def handle_tool_calls(tool_calls):
    """Run the tool functions and return results"""
    results = []
    for call in tool_calls:
        tool_name = call.function.name
        args = json.loads(call.function.arguments)
        
        # Call the actual Python function
        if tool_name == "record_user_details":
            output = record_user_details(**args)
        elif tool_name == "record_unknown_question":
            output = record_unknown_question(**args)
        elif tool_name == "request_meeting":
            output = request_meeting(**args)
        else:
            output = {"status": "error", "message": f"Unknown tool: {tool_name}"}
        
        results.append({
            "role": "tool",
            "content": json.dumps(output),
            "tool_call_id": call.id
        })
    return results

# ============= Build System Prompt =============
def build_system_prompt(style: str, domain: str):
    """Create instructions for the AI based on style and domain"""
    
    # Style instructions
    if style == "Storytelling – Chennai":
        tone = "Warm, Chennai-flavored: confident, humble, story-first with crisp takeaways. Professional but relatable."
    else:
        tone = "Concise, professional, impact-oriented. Lead with results."
    
    # Domain context
    domain_text = f"\n**Focus on {domain} context**\n" if domain else ""
    
    # Main instructions
    prompt = f"""You are {NAME}'s AI assistant. Answer questions about his work, projects, skills, and experience.

**Style**: {tone}

**Your tasks**:
- Answer questions using the profile data below
- When someone shows interest, use record_user_details to capture their info
- When someone wants a meeting, use request_meeting
- If you truly don't know something, use record_unknown_question (don't overuse this)
- Be helpful, professional, and engaging

{domain_text}"""
    
    # Add profile data if available
    if SUMMARY or LINKEDIN:
        prompt += f"\n\n## {NAME}'s Profile\n\n"
        if SUMMARY:
            prompt += f"**Summary**:\n{SUMMARY}\n\n"
        if LINKEDIN:
            prompt += f"**LinkedIn**:\n{LINKEDIN[:5000]}\n"
    else:
        prompt += "\n(No profile data loaded - answer from general knowledge)\n"
    
    return prompt

# ============= Main Chat Function =============
def chat_fn(message, history, style, domain):
    """Process user message and return AI response"""
    
    # Build conversation history
    messages = [{"role": "system", "content": build_system_prompt(style, domain)}]
    
    # Add previous messages
    for msg in history:
        if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    # Call OpenAI with tool support (may loop if tools are called)
    max_loops = 5
    for _ in range(max_loops):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            temperature=0.7,
        )
        
        choice = response.choices[0]
        
        # If AI wants to use tools, execute them and continue
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)  # Add AI's tool request
            messages.extend(handle_tool_calls(choice.message.tool_calls))  # Add tool results
            continue
        
        # Normal response - return it
        return choice.message.content or "No response"
    
    return "Sorry, something went wrong. Please try again."

# ============= UI Styling =============
CSS = """
/* Clean light theme */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* Hero section */
#hero {
    padding: 40px;
    border-radius: 16px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-bottom: 32px;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.25);
}

#title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 12px;
    letter-spacing: -0.5px;
}

#subtitle {
    font-size: 16px;
    line-height: 1.6;
    opacity: 0.95;
}

/* Labels */
label {
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #1f2937 !important;
    margin-bottom: 8px !important;
}

/* Radio buttons */
.gr-radio {
    padding: 4px !important;
}

/* Dropdown */
.gr-dropdown select {
    border: 2px solid #e5e7eb !important;
    border-radius: 8px !important;
    padding: 10px !important;
    font-size: 15px !important;
}

/* Chat container */
.chatbot {
    border: 2px solid #e5e7eb !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

/* Input box */
textarea {
    border: 2px solid #e5e7eb !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    padding: 14px !important;
}

textarea:focus {
    border-color: #667eea !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Buttons */
.gr-button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

.gr-button-secondary {
    background: white !important;
    border: 2px solid #e5e7eb !important;
    color: #374151 !important;
}

.gr-button-secondary:hover {
    background: #f9fafb !important;
    border-color: #d1d5db !important;
}

/* Example buttons */
button[class*="example"] {
    background: white !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 10px !important;
    color: #374151 !important;
    padding: 12px 20px !important;
    transition: all 0.2s !important;
}

button[class*="example"]:hover {
    background: #f9fafb !important;
    border-color: #667eea !important;
    color: #667eea !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15) !important;
}

/* Footer */
#footer {
    text-align: center;
    padding: 24px;
    color: #6b7280;
    font-size: 14px;
    border-top: 2px solid #e5e7eb;
    margin-top: 32px;
}

#footer strong {
    color: #374151;
}

/* Message bubbles */
.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

.bot-message {
    background: #f3f4f6 !important;
    color: #1f2937 !important;
}
"""

# ============= Build UI =============
with gr.Blocks(css=CSS, theme=gr.themes.Soft(), fill_height=True) as demo:
    
    # Header
    gr.Markdown(
        """<div id="hero">
            <div id="title">👨‍💻 Professionally You — AI Agent</div>
            <div id="subtitle">
                Ask about projects, technical skills, and experience. The agent automatically captures leads and schedules meetings.
            </div>
        </div>"""
    )
    
    # Controls
    with gr.Row():
        with gr.Column(scale=1):
            style = gr.Radio(
                ["Professional – Concise", "Storytelling – Chennai"],
                value="Storytelling – Chennai",
                label="🎯 Response Style",
                info="Choose how the agent communicates"
            )
        with gr.Column(scale=1):
            domain = gr.Dropdown(
                ["", "Healthcare", "E-commerce", "FinTech", "IoT / Edge", "Agentic AI"],
                value="",
                label="🏢 Domain Context",
                info="Optional: Tailor responses to your industry"
            )
    
    # Chat Interface
    gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        additional_inputs=[style, domain],
        chatbot=gr.Chatbot(
            type="messages",
            height=520,
            show_copy_button=True,
            avatar_images=(None, "👨‍💻"),
            placeholder="Ask me anything about Aravind's work, skills, or availability...",
            show_label=False
        ),
        textbox=gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=7
        ),
        examples=[
            ["What are your key technical skills?"],
            ["Tell me about your most impactful project"],
            ["Can we schedule a call to discuss collaboration?"],
            ["What's your experience with AI and machine learning?"]
        ]
    )
    
    # Footer with status
    status = "✅ Loaded" if (SUMMARY or LINKEDIN) else "⚠️ Not found"
    gr.Markdown(
        f"""<div id="footer">
            <strong>Profile data:</strong> {status} | 
            <strong>Notifications:</strong> {"✅ Enabled" if (PUSHOVER_USER and PUSHOVER_TOKEN) else "⚠️ Disabled"}<br>
            Built with OpenAI GPT-4o-mini + Gradio
        </div>"""
    )

# ============= Launch =============
if __name__ == "__main__":
    import sys
    print("=" * 50, flush=True)
    print("🚀 Starting Professionally You AI Agent...", flush=True)
    print("=" * 50, flush=True)
    print(f"Profile data loaded: {'✅ Yes' if (SUMMARY or LINKEDIN) else '❌ No'}", flush=True)
    print(f"Pushover enabled: {'✅ Yes' if (PUSHOVER_USER and PUSHOVER_TOKEN) else '❌ No'}", flush=True)
    print("=" * 50, flush=True)
    sys.stdout.flush()
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        quiet=False  # Show all startup messages
    )