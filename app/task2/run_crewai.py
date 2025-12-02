import os
from pathlib import Path
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv()


# Prompt loading functions
def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{prompt_name}.jinja2"
    return prompt_path.read_text(encoding="utf-8")


def create_router_agent():
    """Router agent that decides whether web search is needed."""
    system_prompt = load_prompt("router_prompt")
    
    return Agent(
        role="Router Agent",
        goal="Analyze user questions and determine if web search is needed",
        backstory=system_prompt,
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    )


def create_search_agent():
    """Agent that performs web search and summarizes findings."""
    system_prompt = load_prompt("search_agent_prompt")
    search_tool = ScrapeWebsiteTool()
    
    return Agent(
        role="Research Agent",
        goal="Search the web and summarize findings to answer user questions",
        backstory=system_prompt,
        tools=[search_tool],
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    )


def create_response_agent():
    """Agent that provides the final response to the user."""
    system_prompt = load_prompt("response_agent_prompt")
    
    return Agent(
        role="Response Agent",
        goal="Provide clear, accurate, and helpful responses to user questions",
        backstory=system_prompt,
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    )


def build_workflow():
    """Build the CrewAI workflow."""
    router_agent = create_router_agent()
    search_agent = create_search_agent()
    response_agent = create_response_agent()
    
    return {
        "router": router_agent,
        "search": search_agent,
        "response": response_agent
    }


def initialize_agent():
    """Initialize the multi-agent workflow with proper environment setup."""
    # Ensure API keys are present
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    
    # Note: ScrapeWebsiteTool can scrape websites to extract information
    # No API key required for basic usage
    
    return build_workflow()


def ask(workflow, user_input: str) -> str:
    """Ask a question to the multi-agent workflow."""
    router_agent = workflow["router"]
    search_agent = workflow["search"]
    response_agent = workflow["response"]
    
    # Step 1: Router decides if search is needed
    router_task = Task(
        description=f"Analyze this question and determine if it requires web search. "
                   f"Respond with ONLY 'SEARCH_NEEDED' or 'NO_SEARCH': {user_input}",
        agent=router_agent,
        expected_output="Either 'SEARCH_NEEDED' or 'NO_SEARCH'"
    )
    
    router_crew = Crew(
        agents=[router_agent],
        tasks=[router_task],
        process=Process.sequential,
        verbose=True
    )
    
    router_result = router_crew.kickoff()
    # Extract the result from CrewAI output
    decision = str(router_result.raw if hasattr(router_result, 'raw') else router_result).strip().upper()
    search_needed = "SEARCH_NEEDED" in decision
    
    # Step 2: If search needed, perform search
    search_results = ""
    if search_needed:
        summarizer_prompt = load_prompt("summarizer_prompt")
        
        search_task = Task(
            description=f"Search the web for information to answer this question: {user_input}. "
                       f"After searching, summarize the findings using this instruction: {summarizer_prompt}",
            agent=search_agent,
            expected_output="A concise summary of search results that answers the user's question"
        )
        
        search_crew = Crew(
            agents=[search_agent],
            tasks=[search_task],
            process=Process.sequential,
            verbose=True
        )
        
        search_result = search_crew.kickoff()
        # Extract the result from CrewAI output
        search_results = str(search_result.raw if hasattr(search_result, 'raw') else search_result)
    
    # Step 3: Generate final response
    if search_results:
        response_description = (
            f"Answer this user question: {user_input}\n\n"
            f"Here is information from web search: {search_results}\n\n"
            f"Use the search results to inform your answer. Provide a clear, helpful response."
        )
    else:
        response_description = (
            f"Answer this user question: {user_input}\n\n"
            f"Use your general knowledge to provide a clear, helpful response."
        )
    
    response_task = Task(
        description=response_description,
        agent=response_agent,
        expected_output="A clear, accurate, and helpful response to the user's question"
    )
    
    response_crew = Crew(
        agents=[response_agent],
        tasks=[response_task],
        process=Process.sequential,
        verbose=True
    )
    
    final_response = response_crew.kickoff()
    # Extract the result from CrewAI output
    return str(final_response.raw if hasattr(final_response, 'raw') else final_response)


def main():
    """CLI interface for the multi-agent workflow."""
    workflow = initialize_agent()
    
    print("Multi-Agent Workflow (CrewAI) ready. Type your question (Ctrl+C to exit).")
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

