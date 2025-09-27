import os
from dotenv import load_dotenv
# We now use the standard LangChain OpenAI wrapper, but point it at DeepSeek's URL
from langchain_openai import ChatOpenAI 
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import retrieve_campaign_brief, calculate_correlation 


# Define the global list of all available tools
TOOLS = [
    retrieve_campaign_brief, 
    calculate_correlation
]

# 1. Load Environment Variables
load_dotenv()

# 2. Configure the LLM
try:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file.")

    # Initialize the LLM
    llm = ChatOpenAI( 
        model="deepseek-chat",  
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com/v1", 
        temperature=0.3
    )
    print("Deepseek LLM initialized successfully.")

except Exception as e:
    print(f"Error initializing LLM. Check API key and package: {e}")
    llm = None 

# Define all available tools for agents that need them
TOOLS = [retrieve_campaign_brief, calculate_correlation]

def create_agent(llm, tools: list, system_prompt: str) -> AgentExecutor:
    """Helper function to create a LangChain agent executor."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

# --- AGENT NODE FUNCTIONS ---

def pre_processing_agent_node(state: dict) -> dict:
    """Node for the Pre-Processing Agent."""
    system_prompt = "You are a Data Quality and Pre-Processing Agent. Clean, normalize, and validate survey data. Summarize the cleaned output."
    # Create and execute agent
    agent = create_agent(llm, [retrieve_campaign_brief], system_prompt)
    raw_data = state.get("survey_data_raw", "Sample raw data: Q1=5, Q2=7, Q3=99 (outlier), Q4=2...")
    result = agent.invoke({"input": f"Pre-process and validate this survey data: {raw_data}", "chat_history": []})
    return {"pre_processing_output": result["output"]}

def risk_auditor_agent_node(state: dict) -> dict:
    """Node for the Risk Auditor Agent (uses RAG tool)."""
    system_prompt = "You are an AI Risk Auditor. Flag potential ethical, cultural, or regulatory risks in ads. Use the 'retrieve_campaign_brief' tool for context (RQ4)."
    agent = create_agent(llm, [retrieve_campaign_brief], system_prompt)
    simulated_responses = state.get("simulated_responses", "Simulated responses suggest some users find claims aggressive.")
    input_task = f"Audit the ad based on these simulated responses: {simulated_responses} and the campaign brief."
    result = agent.invoke({"input": input_task, "chat_history": []})
    return {"risk_audit_flags": result["output"]}

def multi_construct_respondent_agent_node(state: dict) -> dict:
    """Node for the Multi-Construct Respondent Agent (Simulated Survey Agent)."""
    system_prompt = "You are a sophisticated 'Simulated Consumer' Agent. Evaluate the ad stimuli across: Attention, Clarity, Persuasiveness, Brand Fit, and Emotional Response (Phase A, RQ1). Provide scores (0-100) and justification."
    # Create and execute agent
    agent = create_agent(llm, [retrieve_campaign_brief], system_prompt) 
    corrected_data = state.get("corrected_data", "Data is clean. Focus: Test Ad Variant A - 'The Mountain' theme.")
    input_task = f"Evaluate Ad Variant A based on the following context/data: {corrected_data}. Provide scores and a diagnostic."
    result = agent.invoke({"input": input_task, "chat_history": []})
    return {"simulated_responses": result["output"]}

def creative_alignment_agent_node(state: dict) -> dict:
    """Node for the Creative Alignment Agent (uses RAG tool)."""
    system_prompt = "You are a Creative Strategy Consultant Agent. Evaluate the alignment between the simulated ad responses and the Campaign Brief. Use the 'retrieve_campaign_brief' tool."
    agent = create_agent(llm, [retrieve_campaign_brief], system_prompt)
    simulated_responses = state.get("simulated_responses", "Simulated responses.")
    input_task = f"Assess creative success based on simulated responses: {simulated_responses}. Retrieve and use the Campaign Brief. Summarize the gap analysis."
    result = agent.invoke({"input": input_task, "chat_history": []})
    return {"gap_analysis_output": result["output"]}

def analytics_agent_node(state: dict) -> dict:
    """Node for the Analytics Agent (uses correlation tool)."""
    system_prompt = "You are a Senior Data Analyst Agent. Compare AI-generated evaluations with human consumer responses to fulfill RQ1 and RQ2. Use the 'calculate_correlation' tool. Generate an 'Analytics Dashboard' summary."
    agent = create_agent(llm, [calculate_correlation], system_prompt)
    ai_data = state.get("simulated_responses", "AI data is ready.")
    human_data = state.get("corrected_data", "Human data is ready.")
    input_task = f"Perform statistical analysis (use the tool) to compare AI data: '{ai_data}' with Human data: '{human_data}'. Generate a summary."
    result = agent.invoke({"input": input_task, "chat_history": []})
    return {"analytics_dashboard": result["output"]}