# Needed for making HTTP requests to LM Studio and websites
import requests
# Needed for parsing HTML content from websites
from bs4 import BeautifulSoup
# For searching with DuckDuckGo (install with: pip install duckduckgo-search)
from duckduckgo_search import DDGS
# For searching YouTube (install with: pip install youtube-search-python)
from youtube_search import YoutubeSearch
# For getting YouTube transcripts (install with: pip install youtube-transcript-api)
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# For Selenium (install with: pip install selenium webdriver-manager)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time # For adding delays if needed
import re # For parsing LLM quality check response
import json # For loading config
import os # For creating cache directory
import hashlib # For creating cache filenames

# --- Global Configuration ---
CONFIG = {}
LLM_CACHE = {} # In-memory cache for the current session

def load_config(config_path="config.json"):
    """Loads configuration from a JSON file."""
    global CONFIG
    try:
        with open(config_path, 'r') as f:
            CONFIG = json.load(f)
        print("✅ Configuration loaded successfully.")
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Configuration file '{config_path}' not found. Please create it.")
        exit()
    except json.JSONDecodeError:
        print(f"❌ CRITICAL ERROR: Configuration file '{config_path}' is not valid JSON.")
        exit()
    
    # Create cache directory if it doesn't exist and caching is enabled
    if CONFIG.get("cache_llm_responses") and CONFIG.get("cache_directory"):
        if not os.path.exists(CONFIG["cache_directory"]):
            try:
                os.makedirs(CONFIG["cache_directory"])
                print(f"✅ Cache directory '{CONFIG['cache_directory']}' created.")
            except Exception as e:
                print(f"⚠️ Warning: Could not create cache directory '{CONFIG['cache_directory']}': {e}")
                CONFIG["cache_llm_responses"] = False # Disable caching if dir creation fails


# --- Helper Functions ---

def get_cache_filepath(prompt_messages_tuple, temperature, max_tokens):
    """Generates a unique filename for caching based on prompt content."""
    if not CONFIG.get("cache_llm_responses") or not CONFIG.get("cache_directory"):
        return None

    # Create a stable string representation of the messages
    # Convert list of dicts to tuple of tuples of sorted items to make it hashable
    stable_messages_repr = tuple(tuple(sorted(msg.items())) for msg in prompt_messages_tuple)
    
    # Combine with other params for uniqueness
    cache_key_data = (stable_messages_repr, temperature, max_tokens, CONFIG.get("lm_studio_model_name"))
    
    # Hash the stable representation
    hasher = hashlib.md5()
    hasher.update(str(cache_key_data).encode('utf-8'))
    filename = hasher.hexdigest() + ".json"
    return os.path.join(CONFIG["cache_directory"], filename)

def query_llm(prompt_messages, temperature=0.7, max_tokens=32000):
    """
    Sends a list of messages to the local LLM via LM Studio and gets a response.
    Uses caching if enabled in config.
    """
    # Convert prompt_messages to a tuple of tuples for use as a dictionary key and for hashing
    prompt_messages_tuple = tuple(tuple(sorted(msg.items())) for msg in prompt_messages)
    
    cache_filepath = get_cache_filepath(prompt_messages, temperature, max_tokens) # Pass original list for hashing

    # Check in-memory cache first
    if CONFIG.get("cache_llm_responses"):
        cache_key_in_memory = (prompt_messages_tuple, temperature, max_tokens, CONFIG.get("lm_studio_model_name"))
        if cache_key_in_memory in LLM_CACHE:
            print(f"✅ LLM Response (In-Memory Cache Hit) for Temp: {temperature}, MaxTokens: {max_tokens}")
            return LLM_CACHE[cache_key_in_memory]
        
        # Check file cache
        if cache_filepath and os.path.exists(cache_filepath):
            try:
                with open(cache_filepath, 'r') as f:
                    cached_response = json.load(f)
                print(f"✅ LLM Response (File Cache Hit) for Temp: {temperature}, MaxTokens: {max_tokens}")
                LLM_CACHE[cache_key_in_memory] = cached_response.get("content") # Store in-memory
                return cached_response.get("content")
            except Exception as e:
                print(f"⚠️ Warning: Could not read from cache file {cache_filepath}: {e}")

    payload = {
        "model": CONFIG.get("lm_studio_model_name"),
        "messages": prompt_messages, # Original list of dicts for the API
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    print(f"\n🤖 Querying LLM ({CONFIG.get('lm_studio_model_name')} Temp: {temperature} MaxTokens: {max_tokens})...")
    try:
        response = requests.post(CONFIG.get("lm_studio_api_endpoint"), json=payload, headers={"Content-Type": "application/json"}, timeout=300) 
        response.raise_for_status()
        response_json = response.json()
        content = None
        if 'choices' in response_json and response_json['choices']:
            if 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                content = response_json['choices'][0]['message']['content'].strip()
            elif 'text' in response_json['choices'][0]: 
                content = response_json['choices'][0]['text'].strip()
        
        if content:
            print("✅ LLM Response Received.")
            if CONFIG.get("cache_llm_responses") and cache_filepath:
                try:
                    # Store the original prompt_messages (list of dicts) in the cache file for readability/debugging
                    with open(cache_filepath, 'w') as f:
                        json.dump({"content": content, "prompt": prompt_messages, "temp": temperature, "max_tokens": max_tokens}, f, indent=4)
                    # Also store in in-memory cache
                    LLM_CACHE[cache_key_in_memory] = content
                    print(f"   💾 LLM Response saved to cache: {cache_filepath}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not write to cache file {cache_filepath}: {e}")
            return content
        else:
            print("⚠️ LLM response did not contain expected content structure.")
            print(f"LLM Response JSON: {response_json}")
            return None

    except requests.exceptions.Timeout:
        print(f"❌ Error: Timeout while querying LLM at {CONFIG.get('lm_studio_api_endpoint')}.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Could not connect or query LLM at {CONFIG.get('lm_studio_api_endpoint')}: {e}")
        if 'response' in locals() and response is not None:
            print(f"LLM Response Status: {response.status_code}")
            print(f"LLM Response Text: {response.text}")
        return None
    except (KeyError, IndexError, TypeError) as e:
        print(f"❌ Error: Could not parse LLM response: {e}")
        if 'response_json' in locals(): print(f"LLM Response JSON: {response_json}")
        elif 'response' in locals() and response is not None: print(f"LLM Response Text: {response.text}")
        return None

def evaluate_content_quality_llm(text_content, research_aspect, topic, source_url):
    """
    Uses LLM to evaluate if the content is high quality, not spam/promotional, or recycled.
    Returns a tuple: (is_high_quality (bool), justification (str))
    """
    print(f"🔎 Evaluating content quality for: {source_url}")
    # Enhanced prompt for recency and quality
    prompt_messages = [
        {"role": "system", "content": (
            "You are a meticulous Content Quality Analyst. Your task is to evaluate the provided text based on its relevance, "
            "informativeness, depth, and potential issues like being spam, heavily promotional, clickbait, or outdated/recycled "
            "content presented with a misleadingly recent title/framing. The primary goal is to determine if this content is suitable for a serious research summary.\n"
            "Consider the source URL if it provides clues (though focus mainly on the text).\n"
            "Specifically check for:\n"
            "1. Depth of Information: Is it superficial, or does it provide substantial details, data, or analysis?\n"
            "2. Recency/Timeliness: Does the content offer specific examples, data, or discussions relevant to recent developments (if the topic implies it), or does it feel like generic advice that could have been written years ago, regardless of the stated publication date (if any)? Look for phrases that might indicate outdatedness despite a new title.\n"
            "3. Promotional Tone: Is it overly focused on selling a product/service rather than providing objective information?\n"
            "4. Spam/Clickbait Characteristics: Does it exhibit signs of low-quality, engagement-farming content?\n"
            "Respond with a verdict and a brief justification. Format your response EXACTLY as follows:\n"
            "Verdict: [GOOD/BAD]\n"
            "Reason: [Your brief justification here, addressing the points above, especially recency and depth]"
        )},
        {"role": "user", "content": (
            f"Topic of Research: '{topic}'\n"
            f"Specific Aspect: '{research_aspect}'\n"
            f"Source URL: {source_url}\n\n"
            f"Text Content to Evaluate (first ~4000 chars):\n---\n{text_content[:4000]}...\n---\n\n"
            "Is this content high quality, genuinely informative, and seemingly current for the aspect and topic, or does it show signs of being low-quality (spam, promotional, recycled, superficial)? "
            "Provide your verdict and reason in the specified format."
        )}
    ]
    
    response = query_llm(prompt_messages, 
                         temperature=CONFIG.get("llm_temperature_content_evaluation", 0.1), 
                         max_tokens=CONFIG.get("llm_max_tokens_content_evaluation", 350))
    
    if not response:
        print("   ⚠️ LLM did not respond for quality check. Assuming BAD quality by default.")
        return False, "LLM did not respond for quality check."

    verdict_match = re.search(r"Verdict:\s*(GOOD|BAD)", response, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
    
    verdict_str = None
    reason_str = "No reason extracted."

    if verdict_match:
        verdict_str = verdict_match.group(1).upper()
    if reason_match:
        reason_str = reason_match.group(1).strip()

    if verdict_str == "GOOD":
        print(f"   ✅ Content quality assessed as GOOD. Reason: {reason_str}")
        return True, reason_str
    elif verdict_str == "BAD":
        print(f"   ❌ Content quality assessed as BAD. Reason: {reason_str}")
        return False, reason_str
    else:
        print(f"   ⚠️ Could not parse LLM quality verdict. Response: '{response}'. Assuming BAD quality.")
        return False, f"Could not parse LLM quality verdict. Raw response: {response}"


def search_web_ddg(query, num_results):
    """
    Performs a web search using DuckDuckGo and returns URLs.
    Sorts results to prefer URLs with keywords from preferred_domain_keywords.
    """
    print(f"🔎 Searching web (DuckDuckGo) for: '{query}' (max {num_results} results)")
    urls = []
    try:
        # Fetch slightly more results to have a better pool for prioritization
        ddg_results_to_fetch = num_results * 2 
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=ddg_results_to_fetch, region='wt-wt') 
            if results:
                for result in results:
                    if result and 'href' in result:
                        urls.append(result['href'])
        
        # Basic source prioritization based on keywords in URL
        preferred_keywords = CONFIG.get("preferred_domain_keywords", [])
        if preferred_keywords:
            # Score URLs: 1 if a preferred keyword is present, 0 otherwise. Sort descending by score.
            # This brings URLs with preferred keywords to the front.
            urls.sort(key=lambda u: any(keyword in u.lower() for keyword in preferred_keywords), reverse=True)
        
        # Return the requested number of results after prioritization
        final_urls = urls[:num_results]
        print(f"Found and prioritized {len(final_urls)} web results via DuckDuckGo (out of {len(urls)} fetched).")
        return final_urls
    except Exception as e:
        print(f"❌ Error during DuckDuckGo web search for '{query}': {e}")
    return urls[:num_results] # Fallback if sorting fails or no preferred keywords

def search_youtube(query, max_results):
    """
    Performs a YouTube search and returns video data (title, id, url).
    """
    print(f"📺 Searching YouTube for: '{query}' (max {max_results} results)")
    video_data = [] 
    try:
        results = YoutubeSearch(query, max_results=max_results).to_dict()
        for video in results:
            if video and 'id' in video and 'title' in video:
                video_data.append({
                    "title": video['title'], 
                    "id": video['id'], 
                    "url": f"https://www.youtube.com/watch?v={video['id']}"
                })
        print(f"Found {len(video_data)} YouTube results.")
    except Exception as e:
        print(f"❌ Error during YouTube search for '{query}': {e}")
    return video_data

def scrape_website_text_selenium(url, char_limit, wait_time):
    """
    Scrapes the main textual content from a webpage using Selenium to handle JavaScript.
    """
    print(f"📄 Scraping website with Selenium: {url}")
    
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"user-agent={CONFIG.get('user_agent')}") # Use user-agent from config

    driver = None
    try:
        print("   Setting up Selenium WebDriver...")
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        print("   WebDriver setup complete.")

        driver.get(url)
        
        print(f"   Waiting up to {wait_time} seconds for page elements to load...")
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        page_source = driver.page_source
        print("   Page source retrieved via Selenium.")
        
        soup = BeautifulSoup(page_source, 'html.parser')
        
        for element_type in ["script", "style", "nav", "footer", "header", "aside", "form", "button", "img", "svg", "iframe", "link", "meta", "noscript", "figure", "figcaption"]:
            for element in soup.find_all(element_type):
                element.decompose()
        
        main_content_selectors = ['main', 'article', '[role="main"]', '.post-content', '.entry-content', '.td-post-content', '.content', '.page-content', '.article-body', '.story-content']
        content_text = ""
        found_main = False
        for selector in main_content_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                content_text = main_element.get_text(separator='\n', strip=True)
                found_main = True
                print(f"   Found main content using selector: {selector}")
                break
        
        if not found_main: 
            if soup.body:
                content_text = soup.body.get_text(separator='\n', strip=True)
                print("   Used body content as fallback.")
            else: 
                content_text = soup.get_text(separator='\n', strip=True)
                print("   Used all text content as fallback (no body tag).")

        lines = (line.strip() for line in content_text.splitlines())
        meaningful_lines = [line for line in lines if sum(c.isalnum() for c in line) > 30 and len(line.split()) > 4] 
        text = '\n'.join(meaningful_lines)
        
        print(f"Scraped {len(text)} characters from {url} (after cleaning with Selenium).")
        return text[:char_limit]
        
    except Exception as e:
        print(f"❌ Error scraping {url} with Selenium: {e}")
        return None
    finally:
        if driver:
            print("   Closing Selenium WebDriver.")
            driver.quit()


def get_youtube_transcript(video_id, char_limit):
    """
    Fetches and returns the transcript for a YouTube video.
    Handles FetchedTranscript object correctly.
    """
    print(f"📜 Getting transcript for YouTube video ID: {video_id}")
    try:
        transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_obj = None # This will be a Transcript object, not FetchedTranscript yet
        
        preferred_langs = ['en', 'en-US', 'en-GB']
        try:
            transcript_obj = transcript_list_obj.find_manually_created_transcript(preferred_langs)
            print("   Found manual English transcript.")
        except NoTranscriptFound:
            try:
                transcript_obj = transcript_list_obj.find_generated_transcript(preferred_langs)
                print("   Found generated English transcript.")
            except NoTranscriptFound:
                print("   No English transcript found. Trying first available...")
                try:
                    # Iterate through all available transcripts if English is not found
                    # The transcript_list_obj is iterable and yields Transcript objects
                    first_available = next(iter(transcript_list_obj), None)
                    if first_available:
                        transcript_obj = first_available
                        print(f"   Using transcript in language: {transcript_obj.language} (code: {transcript_obj.language_code})")
                    else:
                        print(f"   ❌ No transcripts at all found for video ID {video_id} after listing.")
                        return None
                except Exception as e_iter: 
                    print(f"   ❌ Error while trying to get first available transcript: {e_iter}")
                    return None
        
        if not transcript_obj: # Should be caught by the logic above, but as a safeguard
            print(f"   ❌ No usable transcript object found for video ID {video_id}.")
            return None

        # Fetch the actual transcript data. This returns a FetchedTranscript object or raises an error.
        fetched_transcript_data = transcript_obj.fetch() # This is the FetchedTranscript object
        
        text_segments = []
        # The FetchedTranscript object is directly iterable, yielding dicts.
        # No need to check isinstance(fetched_transcript_data, list)
        try:
            for entry in fetched_transcript_data: 
                if isinstance(entry, dict) and 'text' in entry:
                    text_segments.append(entry['text'])
                else:
                    # This case might handle if an element within the fetched data isn't the expected dict
                    print(f"   ⚠️ WARNING: Skipping unexpected entry type in transcript data: {type(entry)} - {str(entry)[:100]}")
            full_transcript_text = " ".join(text_segments)
        except TypeError: # If fetched_transcript_data is not iterable for some unexpected reason
             print(f"   ❌ ERROR: Fetched transcript data (type: {type(fetched_transcript_data)}) is not iterable as expected.")
             return None
        
        if not full_transcript_text.strip():
            print(f"   ⚠️ WARNING: Extracted transcript text is empty for video ID {video_id}.")
            return None

        print(f"Fetched transcript of {len(full_transcript_text)} characters for {video_id}.")
        return full_transcript_text[:char_limit]

    except TranscriptsDisabled:
        print(f"   ❌ Transcripts are disabled for video ID {video_id}.")
        return None
    except NoTranscriptFound: # This can be raised by list_transcripts or find_..._transcript
        print(f"   ❌ No transcript could be found for video ID {video_id} (overall).")
        return None
    except Exception as e: # Catch-all for other unexpected errors
        print(f"❌ Error getting transcript for video ID {video_id}: {type(e).__name__} - {e}")
        return None

# --- Main Program Logic ---

def generate_research_summary(topic):
    """
    Main function to orchestrate the research summary generation process.
    """
    print(f"\n✨ Starting research summary generation for topic: '{topic}' ✨")

    aspect_prompt_messages = [
        {"role": "system", "content": "You are an expert research analyst. Your task is to break down a complex topic into a list of key aspects or sub-topics that would be essential for a comprehensive research summary. Provide only the list of aspect names, each on a new line. Do not add any preamble or explanation. Ensure each aspect name is concise and clear."},
        {"role": "user", "content": f"Topic: '{topic}'. Generate a list of key aspects for a research summary on this topic."}
    ]
    print("\n🧠 Asking LLM to generate key research aspects...")
    aspect_list_str = query_llm(aspect_prompt_messages, 
                                temperature=CONFIG.get("llm_temperature_aspect_generation", 0.4), 
                                max_tokens=CONFIG.get("llm_max_tokens_aspect_generation", 600))

    if not aspect_list_str:
        print("❌ Could not generate key aspects from LLM. Exiting.")
        return
    
    key_aspects = [aspect.strip() for aspect in aspect_list_str.split('\n') if aspect.strip() and len(aspect.strip()) > 3]
    if not key_aspects:
        print("❌ LLM returned an empty or invalid aspect list. Exiting.")
        print(f"LLM raw output for aspects: '{aspect_list_str}'")
        return

    print(f"\n🔑 Generated Key Aspects for '{topic}':")
    for i, aspect in enumerate(key_aspects): print(f"  {i+1}. {aspect}")

    research_content_by_aspect = {}
    processed_urls = set() 

    for aspect_name in key_aspects:
        print(f"\n--- Processing Aspect: '{aspect_name}' ---")
        all_summaries_for_aspect = []
        content_found_for_aspect = False

        search_query_web = f"in-depth information or analysis on '{aspect_name}' regarding '{topic}'"
        search_query_youtube = f"detailed explanation or discussion '{aspect_name}' for '{topic}'"

        web_urls = search_web_ddg(search_query_web, num_results=CONFIG.get("search_web_ddg_num_results", 3)) 
        youtube_videos = search_youtube(search_query_youtube, max_results=CONFIG.get("search_youtube_num_results", 2))

        # Consolidate sources and prioritize web URLs that seem more authoritative
        sources_to_process = []
        for url in web_urls:
            sources_to_process.append({"type": "web", "url": url, "title": None}) 
        for video in youtube_videos:
            sources_to_process.append({"type": "youtube", "url": video['url'], "id": video['id'], "title": video['title']})
        
        # Simple reordering: process preferred web URLs first
        preferred_keywords = CONFIG.get("preferred_domain_keywords", [])
        if preferred_keywords:
            # Sort so that items matching preferred keywords come first
            # True (1) for match, False (0) for no match. Sort descending.
            sources_to_process.sort(key=lambda s: (s['type'] == 'web' and any(keyword in s['url'].lower() for keyword in preferred_keywords)), reverse=True)


        for source_item in sources_to_process:
            source_url = source_item["url"]
            if source_url in processed_urls:
                print(f"   ⏩ Skipping already processed URL: {source_url}")
                continue
            
            text_to_evaluate = None
            source_type_for_summary = ""

            if source_item["type"] == "web":
                scraped_text = scrape_website_text_selenium(source_url, 
                                                            char_limit=CONFIG.get("scrape_char_limit", 30000), 
                                                            wait_time=CONFIG.get("scrape_wait_time", 15))
                processed_urls.add(source_url)
                if scraped_text and len(scraped_text) > 200:
                    text_to_evaluate = scraped_text
                    source_type_for_summary = f"Web Article - Scraped: {source_url}"
                elif scraped_text:
                    print(f"   ⚠️ Scraped text from {source_url} is too short to summarize meaningfully (Length: {len(scraped_text)}).")
                else:
                    print(f"   ⚠️ No text scraped from {source_url} using Selenium.")

            elif source_item["type"] == "youtube":
                video_title = source_item["title"]
                video_id = source_item["id"]
                transcript = get_youtube_transcript(video_id, char_limit=CONFIG.get("transcript_char_limit", 25000))
                processed_urls.add(source_url) 
                if transcript and len(transcript) > 200:
                    text_to_evaluate = transcript
                    source_type_for_summary = f"YouTube Video: {video_title} - {source_url}"
                elif transcript:
                    print(f"   ⚠️ Transcript for {video_title} is too short to summarize meaningfully (Length: {len(transcript)}).")
                else:
                    print(f"   ⚠️ No transcript retrieved for {video_title}.")

            if text_to_evaluate:
                is_good, justification = evaluate_content_quality_llm(text_to_evaluate, aspect_name, topic, source_url)
                if not is_good:
                    print(f"   🗑️ Discarding content from {source_url} due to low quality. Justification: {justification}")
                    continue

                print(f"   Summarizing content from: {source_url}")
                summary_prompt_messages = [
                    {"role": "system", "content": "You are an expert information extractor. Your task is to summarize the provided text, focusing on key findings, data, arguments, evidence, and important conclusions relevant to the research aspect. Prioritize specific, verifiable information. Be concise, factual, and informative. Avoid fluff, opinions unless clearly stated as such in the source, and general introductions/conclusions. Output only the summary."},
                    {"role": "user", "content": f"The following text is about '{aspect_name}' in the context of research on '{topic}'. Please summarize its key information and findings. Text:\n\n---\n{text_to_evaluate}\n---"}
                ]
                summary = query_llm(summary_prompt_messages, 
                                    temperature=CONFIG.get("llm_temperature_summarization", 0.2), 
                                    max_tokens=CONFIG.get("llm_max_tokens_summarization", 1500)) 
                if summary:
                    all_summaries_for_aspect.append(f"Source ({source_type_for_summary}):\n{summary}\n")
                    content_found_for_aspect = True
                else: print(f"   ⚠️ Could not summarize text from {source_url}")
        
        if content_found_for_aspect:
            research_content_by_aspect[aspect_name] = "\n\n".join(all_summaries_for_aspect)
            print(f"   ✅ High-quality content gathered and summarized for aspect '{aspect_name}'.")
        else:
            research_content_by_aspect[aspect_name] = f"No high-quality content found or summarized for the aspect '{aspect_name}' after searching, filtering, and processing."
            print(f"   ⚠️ No high-quality content for aspect '{aspect_name}'.")

    print("\n\n🔬 Synthesizing Full Research Summary from Gathered Content...")
    final_summary_prompt_parts = [
        f"You are an expert research analyst and writer. Your primary task is to synthesize the provided research findings into a comprehensive, insightful, and well-structured research summary on the topic: '{topic}'.",
        "The summary should be rich in information, clearly explain complex points, and provide a holistic overview of the topic based on the gathered data from high-quality sources.",
        "Structure the summary logically. Start with a brief introduction. Then, for each key aspect, integrate the summarized findings from reputable sources. Ensure smooth transitions.",
        "If information for an aspect is sparse or notes 'No high-quality content found', briefly acknowledge this lack of vetted information for that aspect and suggest it as an area needing further investigation with reliable sources.",
        "Conclude with an overall synthesis or concluding remarks. Maintain an objective and analytical tone. Prioritize information that is specific, well-supported, and appears current.",
        "Format the output clearly using Markdown (e.g., headings for the main topic and sub-headings for key aspects).",
        "\nHere is the summarized content for each research aspect (filtered for quality):\n"
    ]
    for i, aspect_name in enumerate(key_aspects):
        content = research_content_by_aspect.get(aspect_name, "No high-quality content was gathered for this aspect.")
        final_summary_prompt_parts.append(f"\n--- Research Aspect {i+1}: {aspect_name} ---\n{content}\n---\n")
    
    final_summary_prompt_messages = [
        {"role": "system", "content": "You are an expert research analyst and writer. Synthesize provided research notes (which have been pre-filtered for quality) into a comprehensive, insightful, and well-structured research summary. If notes for an aspect indicate no high-quality content was found, acknowledge this. Use Markdown for formatting."},
        {"role": "user", "content": "\n".join(final_summary_prompt_parts)}
    ]
    print("🧠 Asking LLM to generate the final research summary...")
    final_summary = query_llm(final_summary_prompt_messages, 
                              temperature=CONFIG.get("llm_temperature_final_synthesis", 0.5), 
                              max_tokens=CONFIG.get("llm_max_tokens_final_synthesis", 8192)) 

    if final_summary:
        print("\n\n🎉 --- Generated Research Summary --- 🎉")
        print(final_summary)
        safe_topic_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in topic).rstrip()
        filename = f"{safe_topic_name.replace(' ', '_')}_research_summary_v2.md"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# Research Summary (v2 - Filtered & Configured): {topic}\n\n{final_summary}")
            print(f"\n💾 Research summary saved to {filename}")
        except Exception as e: print(f"❌ Error saving research summary to file: {e}")
    else: print("❌ Could not generate the final research summary from LLM.")

# --- Run the program ---
if __name__ == "__main__":
    load_config() # Load configuration at the start

    print("--- AI Research Summary Generator (v2 - with Config, Cache, Recency Filter, Transcript Fix) ---")
    print("This script uses a local LLM (via LM Studio) to generate a research summary")
    print("by searching, scraping, filtering for quality, and synthesizing content.")
    print(f"Using LLM Model: {CONFIG.get('lm_studio_model_name', 'NOT SET IN CONFIG')}")
    print(f"LLM API Endpoint: {CONFIG.get('lm_studio_api_endpoint', 'NOT SET IN CONFIG')}")
    if CONFIG.get("cache_llm_responses"):
        print(f"LLM Caching: ENABLED (Directory: {CONFIG.get('cache_directory')})")
    else:
        print("LLM Caching: DISABLED")

    print("Ensure LM Studio is running and configured correctly in this script via 'config.json'.")
    print("Make sure you have installed the required libraries: pip install requests beautifulsoup4 duckduckgo-search youtube-search-python youtube-transcript-api selenium webdriver-manager")
    print("You also need Google Chrome installed for Selenium to control.")
    print("\nIMPORTANT FOR WSL (Windows Subsystem for Linux) USERS:")
    print("If Selenium/ChromeDriver fails with 'status code 127', you likely need to install Chrome dependencies in WSL:")
    print("sudo apt-get update && sudo apt-get install -y libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 libgtk-3-0 libasound2 lsb-release xdg-utils wget fonts-liberation")
    print("Also, ensure LM_STUDIO_API_ENDPOINT uses your Windows host IP if LM Studio is on Windows and this script in WSL.\n")
    print("---------------------------\n")

    if not CONFIG.get("lm_studio_model_name") or CONFIG.get("lm_studio_model_name") == "your-local-model-identifier": 
        print("🚨 CRITICAL ERROR: Please set 'lm_studio_model_name' in 'config.json'")
        print("   to the identifier of the model you have loaded in LM Studio.")
        exit()
    
    if not CONFIG.get("lm_studio_api_endpoint") or "1234/v1/chat/completions" not in CONFIG.get("lm_studio_api_endpoint"): 
         print(f"🚨 WARNING: 'lm_studio_api_endpoint' ('{CONFIG.get('lm_studio_api_endpoint')}') in 'config.json' might not be correctly configured.")
    elif "localhost:1234" in CONFIG.get("lm_studio_api_endpoint") and not CONFIG.get("lm_studio_api_endpoint").startswith("http://") and not CONFIG.get("lm_studio_api_endpoint").startswith("https://"):
        print(f"🚨 WARNING: 'lm_studio_api_endpoint' ('{CONFIG.get('lm_studio_api_endpoint')}') in 'config.json' might be missing 'http://' or 'https://'.")

    research_topic_input = input("➡️ Enter the topic for the research summary you want to create (e.g., 'The Impact of AI on Climate Change Solutions'): ")
    if research_topic_input and research_topic_input.strip():
        generate_research_summary(research_topic_input.strip())
    else:
        print("No topic entered. Exiting.")

