# AI Research Summary Agent (v2)

## 1. Overview

The AI Research Summary Agent is a Python-based tool designed to automate the process of generating comprehensive research summaries on a given topic. It leverages local Large Language Models (LLMs) via an LM Studio API endpoint, web scraping with Selenium, and information retrieval from DuckDuckGo and YouTube to gather, filter, summarize, and synthesize information.

The agent aims to produce well-structured, informative summaries by:
- Breaking down a topic into key research aspects.
- Searching the web and YouTube for relevant content for each aspect.
- Scraping textual content from websites and fetching YouTube video transcripts.
- Evaluating the quality of gathered content using an LLM to filter out spam, promotional material, and potentially recycled or superficial information.
- Summarizing the high-quality content for each aspect using an LLM.
- Synthesizing all summarized information into a final, coherent research summary in Markdown format.

## 2. Features

-   **Topic Decomposition:** Uses an LLM to break down a user-provided topic into key researchable aspects.
-   **Multi-Source Information Retrieval:**
    -   Searches the web using DuckDuckGo.
    -   Searches YouTube for relevant videos and fetches their transcripts.
-   **Dynamic Web Scraping:** Employs Selenium to scrape text from JavaScript-heavy websites.
-   **LLM-Powered Content Evaluation:** Assesses the quality, relevance, and potential recency of scraped content before processing.
-   **LLM-Powered Summarization:** Generates concise summaries of individual content pieces.
-   **LLM-Powered Synthesis:** Compiles all summarized information into a final, structured research document.
-   **External Configuration:** Key parameters (API endpoints, model names, search settings, LLM parameters) are managed via an external `config.json` file.
-   **LLM Response Caching:** Implements file-based and in-memory caching for LLM responses to save time and resources on repeated queries.
-   **Basic Source Prioritization:** Favors web URLs containing keywords typically associated with authoritative sources (e.g., .edu, .gov, .org).
-   **Markdown Output:** Saves the final research summary as an `.md` file.

## 3. Requirements

-   Python 3.10+
-   LM Studio (or any OpenAI-compatible API endpoint for a local LLM) running with a loaded model.
-   Google Chrome browser installed (for Selenium).

## 4. Setup and Installation

1.  **Clone/Download the Script:**
    git clone the project using ```git clone https://github.com/entity0326/AI-Research-Agent```

2.  **Install Python Dependencies:**
    Open your terminal or command prompt, navigate to the project directory, and install the required libraries:
    ```
    pip install -r requirements.txt
    ```

3.  **Set up LM Studio (or other LLM API):**
    -   Ensure LM Studio is installed and running.
    -   Load a suitable instruction-following model (e.g., Qwen2, Mixtral, Llama).
    -   Start the local server in LM Studio and note the API endpoint URL (e.g., `http://localhost:1234/v1/chat/completions`) and the model identifier.

4.  **WSL (Windows Subsystem for Linux) Specific Setup:**
    If you are running this script within WSL and Selenium/ChromeDriver fails with a "status code 127" or similar, you likely need to install Chrome and its dependencies within your WSL distribution:
    ```bash
    sudo apt-get update && sudo apt-get install -y libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 libgtk-3-0 libasound2 lsb-release xdg-utils wget fonts-liberation
    ```
    Also, if LM Studio is running on your Windows host and the script is in WSL, ensure the `lm_studio_api_endpoint` in `config.json` uses your Windows host's IP address (e.g., `http://<your-windows-ip>:1234/v1/chat/completions`) instead of `localhost`.
