import os
from pathlib import Path
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# Load environment variables
load_dotenv()


# Prompt loading functions
def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{prompt_name}.jinja2"
    return prompt_path.read_text(encoding="utf-8")


# Define the state schema
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    search_needed: bool
    search_results: str
    final_response: str


def create_router_agent():
    """Router agent that decides whether web search is needed."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system_prompt = load_prompt("router_prompt")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return prompt | model


def create_search_agent():
    """Agent that performs web search and summarizes findings."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system_prompt = load_prompt("search_agent_prompt")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Initialize search tool
    search_tool = DuckDuckGoSearchRun()
    
    # Bind tools to the model
    agent = prompt | model.bind_tools([search_tool])
    
    return agent


def create_response_agent():
    """Agent that provides the final response to the user."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system_prompt = load_prompt("response_agent_prompt")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return prompt | model


def router_node(state: AgentState) -> AgentState:
    """Node that routes to search or direct response."""
    router = create_router_agent()
    
    # Get the last user message
    last_message = state["messages"][-1]
    
    # Route decision
    response = router.invoke({"messages": [last_message]})
    decision = response.content.strip().upper()
    
    # Update state
    state["search_needed"] = "SEARCH_NEEDED" in decision
    state["messages"].append(AIMessage(content=f"Router decision: {decision}"))
    
    return state


def search_node(state: AgentState) -> AgentState:
    """Node that performs web search and summarizes results."""
    search_agent = create_search_agent()
    
    # Get the original user question
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg
            break
    
    if not user_message:
        state["search_results"] = "No user question found."
        return state
    
    # Invoke search agent
    response = search_agent.invoke({"messages": [user_message]})
    
    # Add the agent's response to messages
    state["messages"].append(response)
    
    # Check if tool calls were made
    tool_calls = getattr(response, "tool_calls", None) or []
    
    if tool_calls:
        # Execute search tool
        search_tool = DuckDuckGoSearchRun()
        search_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            if "duckduckgo" in tool_name.lower() or "search" in tool_name.lower():
                # Extract query from tool call args
                args = tool_call.get("args", {})
                query = args.get("query") if isinstance(args, dict) else str(user_message.content)
                if not query:
                    query = str(user_message.content)
                try:
                    # DuckDuckGoSearchRun expects a string input
                    result = search_tool.invoke(query)
                    search_results.append(f"Query: {query}\nResults: {result}")
                except Exception as e:
                    search_results.append(f"Query: {query}\nError: {str(e)}")
        
        if search_results:
            # Summarize search results
            summarizer_system_prompt = load_prompt("summarizer_prompt")
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", summarizer_system_prompt),
                ("human", f"User question: {user_message.content}\n\nSearch results:\n{chr(10).join(search_results)}"),
            ])
            
            summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            summary = summarizer.invoke(summary_prompt.format_messages())
            
            state["search_results"] = summary.content if hasattr(summary, "content") else str(summary)
        else:
            state["search_results"] = "Search tool was called but no results were obtained."
    else:
        # If no tool calls, try direct search with user query
        search_tool = DuckDuckGoSearchRun()
        try:
            result = search_tool.invoke(str(user_message.content))
            state["search_results"] = result
        except Exception as e:
            state["search_results"] = f"Search failed: {str(e)}"
    
    return state


def response_node(state: AgentState) -> AgentState:
    """Node that generates the final response."""
    response_agent = create_response_agent()
    
    # Build context for response
    messages_to_send = []
    
    # Add original user message
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            messages_to_send.append(msg)
            break
    
    # Add search results if available
    if state.get("search_results"):
        messages_to_send.append(AIMessage(
            content=f"Here is information from web search: {state['search_results']}"
        ))
    
    # Generate response
    response = response_agent.invoke({"messages": messages_to_send})
    
    final_response = response.content if hasattr(response, "content") else str(response)
    state["final_response"] = final_response
    state["messages"].append(AIMessage(content=final_response))
    
    return state


def should_search(state: AgentState) -> Literal["search", "direct_response"]:
    """Conditional edge function to route based on search decision."""
    return "search" if state.get("search_needed", False) else "direct_response"


def build_workflow():
    """Build the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("search", search_node)
    workflow.add_node("response", response_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        should_search,
        {
            "search": "search",
            "direct_response": "response"
        }
    )
    
    # Add edges from search to response
    workflow.add_edge("search", "response")
    
    # Add edge from response to end
    workflow.add_edge("response", END)
    
    return workflow.compile()


def initialize_agent():
    """Initialize the multi-agent workflow with proper environment setup."""
    # Ensure API key is present
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    
    return build_workflow()


def ask(workflow, user_input: str) -> str:
    """Ask a question to the multi-agent workflow."""
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "search_needed": False,
        "search_results": "",
        "final_response": ""
    }
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    return result.get("final_response", "No response generated.")


def main():
    """CLI interface for the multi-agent workflow."""
    workflow = initialize_agent()
    
    print("Multi-Agent Workflow ready. Type your question (Ctrl+C to exit).")
    print("The agent will decide whether to search the web or provide a direct answer.\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
            
            print("\nProcessing...")
            response = ask(workflow, user_input)
            print(f"\n{response}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()

