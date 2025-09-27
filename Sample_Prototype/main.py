from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from agents import (
    pre_processing_agent_node, 
    risk_auditor_agent_node, 
    multi_construct_respondent_agent_node, 
    creative_alignment_agent_node,
    analytics_agent_node
)

# --- 1. DEFINE THE GRAPH STATE (THE BLACKBOARD) ---
class PreTestingState(TypedDict):
    """Represents the shared state of the workflow."""
    survey_data_raw: str | None
    pre_processing_output: str | None
    corrected_data: str | None
    simulated_responses: str | None
    campaign_brief: str | None
    risk_audit_flags: str | None
    creative_brief_available: bool # Decision from 'Human Check 2'
    gap_analysis_output: str | None
    analytics_dashboard: str | None
    final_report_draft: str | None
    final_report: str | None
    chat_history: Annotated[List[BaseMessage], add_messages]


# --- 2. HUMAN CHECK FUNCTIONS (NODES) ---

def human_check_1_verify(state: PreTestingState) -> dict:
    """Human Check 1: Verify Inputs. Assumes human accepts/corrects the data."""
    print("\n--- PAUSE: Human Check 1 (Verify Inputs) ---")
    
    # We use a blocking input() for a simple pause in the terminal
    input(">>> PRESS ENTER to simulate human verification of Pre-Processing Output. <<<")
    
    pre_processed_data = state.get("pre_processing_output", "No data found.")
    
    # Human verifies and provides the 'Corrected Pre-Processing Output'
    corrected_data = f"Human Verified & Corrected Data: {pre_processed_data}" 
    
    return {"corrected_data": corrected_data}


def human_check_2_brief(state: PreTestingState) -> str:
    """Human Check 2: Creative Brief Decision. Branching decision point."""
    print("\n--- PAUSE: Human Check 2 (Creative Brief Decision) ---")
    
    # Simple terminal decision point (Y/N)
    decision = input(">>> Is the Creative Brief available? (Type 'yes' or 'no'): ").lower().strip()

    if decision == 'yes':
        print("Decision: Brief available. Proceeding to Creative Alignment Agent.")
        # CRITICAL FIX: Return a dict containing the transition key
        return {"creative_brief_status": "brief_available"} 
    else:
        print("Decision: No brief. Skipping to Analytics Agent.")
        # CRITICAL FIX: Return a dict containing the transition key
        return {"creative_brief_status": "no_brief"}


def human_check_3_report(state: PreTestingState) -> dict:
    """Human Check 3: Results before Report. Human reviews dashboard and finalizes."""
    print("\n--- PAUSE: Human Check 3 (Review Dashboard and Draft Report) ---")
    
    # We use a blocking input() for a simple pause in the terminal
    input(">>> PRESS ENTER to simulate human review of Analytics Dashboard and finalize the report. <<<")
    
    analytics_dashboard = state.get("analytics_dashboard", "No dashboard data.")

    # The human review results in the final report draft
    final_report = (
        f"FINAL REPORT (Human Approved): Based on the analysis summarized in the Dashboard: '{analytics_dashboard}', "
        "the AI system provides strong directional insights. Recommendations are finalized."
    )
    
    return {"final_report": final_report}


# --- 3. BUILD THE GRAPH ---

workflow = StateGraph(PreTestingState)

# 3a. Define Nodes (Steps)
workflow.add_node("preprocess", pre_processing_agent_node)
workflow.add_node("human_check_1", human_check_1_verify)
workflow.add_node("multi_respondent", multi_construct_respondent_agent_node)
workflow.add_node("risk_auditor", risk_auditor_agent_node)
workflow.add_node("human_check_2", human_check_2_brief)
workflow.add_node("creative_alignment", creative_alignment_agent_node)
workflow.add_node("analytics", analytics_agent_node)
workflow.add_node("human_check_3", human_check_3_report)

# 3b. Define Edges (Flow)
workflow.set_entry_point("preprocess")

# Data Path
workflow.add_edge("preprocess", "human_check_1")
workflow.add_edge("human_check_1", "multi_respondent")

# AI Simulation Path
workflow.add_edge("multi_respondent", "risk_auditor")
workflow.add_edge("risk_auditor", "human_check_2") 

# Branching Logic (Human Check 2)
workflow.add_conditional_edges(
    "human_check_2", 
    # 2. Conditional Logic (The Lambda Function - one callable object)
    lambda state: state['creative_brief_status'],
    # 3. Mapping Dictionary (The Dict)
    {
        "brief_available": "creative_alignment",
        "no_brief": "analytics" 
    }
)

# Alignment Path (if brief was available)
workflow.add_edge("creative_alignment", "analytics") 

# Final Steps
workflow.add_edge("analytics", "human_check_3")
workflow.add_edge("human_check_3", END)


# 4. Compile the Graph
app = workflow.compile()
print("LangGraph pipeline compiled successfully.")

# --- 5. EXECUTION ---

# Sample raw input data 
initial_state = PreTestingState(
    survey_data_raw="[Raw Survey] Ad A: Persuasion 5/7, Clarity 6/7. Ad B: Persuasion 4/7, Clarity 7/7 (High confidence). N=485. Q1=Missing for 10 users.",
    simulated_responses=None,
    pre_processing_output=None,
    corrected_data=None,
    campaign_brief="Need to check if the 'Eco-Refresh' campaign brief is available.",
    risk_audit_flags=None,
    creative_brief_available=False,
    gap_analysis_output=None,
    analytics_dashboard=None,
    final_report_draft=None,
    final_report=None,
    chat_history=[]
)

print("\n--- STARTING AGENT WORKFLOW ---")
try:
    # Run the graph
    final_state = app.invoke(initial_state)
    
    print("\n--- WORKFLOW COMPLETE ---")
    print("\nFINAL REPORT GENERATED:")
    print(final_state["final_report"])

except Exception as e:
    print(f"\n--- WORKFLOW ERROR --- \n{e}")