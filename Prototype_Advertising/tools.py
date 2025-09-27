from langchain_core.tools import tool
import os


# --- MOCK DATA ---
# Simulating the 'Campaign brief' retrieval for the RAG component
MOCK_CAMPAIGN_BRIEF = """
Campaign Title: 'Eco-Refresh' Launch
Product: New biodegradable water bottle line.
Target Audience: Eco-conscious millennials (Age 25-35) in urban areas.
Key Message: 'Hydrate your life, not the planet.'
Risk Areas to Avoid: Overly aggressive "greenwashing" claims,
stereotyping of the target audience as "hippies."
"""

# --- TOOLS ---
@tool
def retrieve_campaign_brief(query: str) -> str:
    """
    Retrieves the campaign brief and associated background documents 
    using a RAG mechanism based on the query. This provides context 
    to the Risk Auditor Agent and Creative Alignment Agent.
    """
    if "campaign" in query.lower() or "brief" in query.lower():
        print("Tool executed: Retrieving Campaign Brief from RAG...")
        return MOCK_CAMPAIGN_BRIEF
    else:
        return "No relevant campaign brief found for the query."

@tool
def calculate_correlation(data_summary: str) -> str:
    """
    Simulates calculating Pearson/Spearman correlation between two sets of ratings (AI vs. Human).
    Input should be a summary of the data to correlate.
    """
    print(f"Tool executed: Calculating correlation for: {data_summary}")
    # Simulates the result from your research objective (RQ1)
    return "Result: Moderate positive correlation (r=0.45) found between AI and Human Persuasion scores."