# KnowMD0

## Overview

## Setup Instructions

### Prerequisites
Ensure your system is equipped with the following:
- Python 3.6 or newer
- Essential Python packages (installable via pip):
  - `langchain`
  - `chainlit`
  - `sentence-transformers`
  - `faiss`
  - `PyPDF2` (for loading PDF documents)

### Setting Up Your Environment
1. **Create a Python Virtual Environment** (Recommended):
   - Initialize the environment: 
     ```
     python -m venv venv
     ```
   - Activate the environment:
     - On Unix or MacOS: 
       ```
       source venv/bin/activate
       ```
     - On Windows: 
       ```
       venv\Scripts\activate
       ```

2. **Install Required Packages**:
   - Install all dependencies from the `requirements.txt` file:
     ```
     pip install -r requirements.txt
     ```

After completing these steps, you'll be ready to start using the Langchain Medical Bot.

## Usage
To use the Openchat llm model Chatbot, ensure that the required data sources are available in the specified 'data' directory. This data can be in the file format of pdf, txt, or xlsx. Run the `ingest.py` script first to process the data and create the vector database. Once the database is ready, open Git Bash within your folder, and input/execute the following: `chainlit run model.py -w` to start the chatbot and interact with your files.

![CPUMedicalChatbot Interface](https://github.com/VidhyaVarshanyJS/KnowMD0/blob/master/others/Architecture.png)

