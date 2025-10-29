import gradio as gr
import os
from dotenv import load_dotenv
from run import initialize_agent, ask

# Load environment variables from .env file
load_dotenv()

# Global agent instance
agent = None

def initialize_agent_global():
    """Initialize the global agent."""
    global agent
    try:
        if os.getenv("OPENAI_API_KEY"):
            agent = initialize_agent()
            return "✅ Agent initialized successfully!"
        else:
            agent = None
            return "❌ OpenAI API Key not found"
    except Exception as e:
        agent = None
        return f"❌ Failed to initialize agent: {str(e)}"

def chat_with_agent(message, history):
    """Handle chat interaction with the agent."""
    global agent
    
    if not agent:
        return "Agent not initialized. Please check your API key configuration.", history
    
    if not message.strip():
        return "", history
    
    try:
        # Get response from agent
        response = ask(agent, message)
        
        # Update chat history
        history.append([message, response])
        
        return "", history
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        history.append([message, error_msg])
        return "", history

def clear_chat():
    """Clear the chat history."""
    return []

def refresh_agent_status():
    """Refresh the agent and return status."""
    return initialize_agent_global()

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize the agent
    status = initialize_agent_global()
    
    with gr.Blocks(title="AI Agent Chat") as demo:
        
        # Header
        gr.Markdown("# 🤖 AI Agent Chat")
        gr.Markdown("Chat with an AI agent that's designed to be polite but unhelpful!")
        
        # Status display
        status_text = gr.Textbox(
            value=status,
            label="Status",
            interactive=False
        )
        
        # Chat interface
        chatbot = gr.Chatbot(label="Chat History", height=400)
        
        # Input area
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me anything...",
                label="Your Message",
                scale=4
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Control buttons
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            refresh_btn = gr.Button("🔄 Refresh Agent", variant="secondary")
        
        # Instructions
        gr.Markdown("""
        ### Instructions
        This AI agent is designed to be polite but unhelpful. 
        It will tell you that it cannot answer your questions.
        
        Try asking it anything to see how it responds!
        """)
        
        # Event handlers
        send_btn.click(
            chat_with_agent,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            chat_with_agent,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot]
        )
        
        refresh_btn.click(
            refresh_agent_status,
            outputs=[status_text]
        )
    
    return demo


def main():
    """Main function to launch the Gradio app."""
    print("🚀 Starting Gradio AI Agent Chat...")
    
    # Create the interface
    demo = create_gradio_interface()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
