import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def load_system_prompt() -> str:
    template_path = Path(__file__).parent / "prompt_templates" / "system_prompt.jinja2"
    return template_path.read_text(encoding="utf-8")


def build_agent():
    system_prompt = load_system_prompt()

    # Build a simple chat prompt with a system message enforcing our policy
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Model: uses OPENAI_API_KEY from env; user can swap to other chat models
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = prompt | model
    return chain


def ask(chain, user_input: str) -> str:
    result = chain.invoke({"input": user_input})
    return result.content if hasattr(result, "content") else str(result)


def initialize_agent():
    """Initialize the agent with proper environment setup."""
    # Load environment from .env if present
    load_dotenv()

    # Ensure API key present; fail fast with a polite message
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    return build_agent()


def main():
    """CLI interface for the agent."""
    chain = initialize_agent()

    # Simple REPL
    print("Agent ready. Type your question (Ctrl+C to exit).")
    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            response = ask(chain, user_input)
            print(response)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()


