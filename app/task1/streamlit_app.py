import streamlit as st
import os
from typing import Optional
from dotenv import load_dotenv
from run import initialize_agent, ask

# Load environment variables from .env file
load_dotenv()


def initialize_session_state():
    """Initialize session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = False


def check_api_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="AI Agent Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def display_sidebar():
    """Display the sidebar with configuration options."""
    with st.sidebar:
        st.title("🤖 AI Agent")
        st.markdown("---")
        
        # API Key status
        if check_api_key():
            st.success("✅ OpenAI API Key is set")
            st.session_state.api_key_set = True
        else:
            st.error("❌ OpenAI API Key not found")
            st.markdown("Please set your `OPENAI_API_KEY` environment variable or add it to a `.env` file.")
            st.session_state.api_key_set = False
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # Instructions
        st.markdown("### Instructions")
        st.markdown("""
        This AI agent is designed to be polite but unhelpful. 
        It will tell you that it cannot answer your questions.
        
        Try asking it anything to see how it responds!
        """)


def initialize_agent_if_needed():
    """Initialize the agent if API key is available and agent not yet initialized."""
    if st.session_state.api_key_set and st.session_state.agent is None:
        try:
            with st.spinner("Initializing AI agent..."):
                st.session_state.agent = initialize_agent()
            st.success("Agent initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.session_state.agent = None


def display_chat_messages():
    """Display chat messages in the main area."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input():
    """Handle user input and generate responses."""
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        if st.session_state.agent:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = ask(st.session_state.agent, prompt)
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            error_msg = "Agent not initialized. Please check your API key configuration."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    """Main Streamlit application."""
    setup_page()
    initialize_session_state()
    display_sidebar()
    
    # Main content area
    st.title("🤖 AI Agent Chat")
    st.markdown("Chat with an AI agent that's designed to be polite but unhelpful!")
    
    # Initialize agent if possible
    initialize_agent_if_needed()
    
    # Display chat messages
    display_chat_messages()
    
    # Handle user input
    handle_user_input()


if __name__ == "__main__":
    main()
