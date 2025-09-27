# 🤖 Agentic Workflow for Analytical Validation (LangGraph)
This project implements an Agentic Workflow using LangGraph to automate complex, multi-stage analytical processes. The architecture orchestrates specialized LLM agents for functions like data handling, contextual retrieval, and validation, demonstrating a robust framework for building trustworthy, auditable analytical systems.

### 🔑 Setup and Execution

#### 1. Installation
Clone the repository and install dependencies:

##### Example setup command
pip install -r requirements.txt

#### 2. API Key Configuration
The project uses the DeepSeek API for LLM execution. Secure your key using a .env file:

- Edit the file named .env in the root directory.

- Add your API key using the exact variable name below:

##### .env file content
DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY_HERE"

#### 3. Run the System
Start the LangGraph pipeline:

python main.py


