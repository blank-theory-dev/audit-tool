# --- Core Imports ---
import streamlit as st
import json
import requests
from urllib.parse import urlparse, urljoin, quote_plus
from bs4 import BeautifulSoup, Comment # Added Comment to potentially ignore commented out sections
import datetime
import traceback
import io
import asyncio
import webbrowser
import re
import threading

# Assuming meta_ads_audit.py is in the same directory and correctly implemented
try:
    from meta_ads_audit import meta_ads_audit
    META_ADS_AUDIT_AVAILABLE = True
except ImportError:
    META_ADS_AUDIT_AVAILABLE = False

# --- Streamlit App UI Config (MUST BE FIRST st command) ---
st.set_page_config(page_title="Gemini Product Page UX Auditor", layout="wide")

# --- Playwright Import & Check ---
try:
    from playwright.async_api import async_playwright, Error as PlaywrightError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    st.error("❌ Playwright library not found! Please run: pip install playwright && playwright install")
    st.warning("⚠️ Automatic screenshotting will be disabled.")
    PLAYWRIGHT_AVAILABLE = False
    import os
    if os.name != 'nt': # Avoid running system commands on Windows if not necessary or if they might fail
        st.info("Attempting to install Playwright browsers and dependencies...")
        try:
            os.system('playwright install')
            os.system('playwright install-deps')
            st.info("Playwright installation commands executed.")
        except Exception as install_e:
            st.error(f"Error running playwright install commands: {install_e}")

# --- Image Handling ---
try:
    from PIL import Image
    PIL_AVAILABLE = True
    import google.ai.generativelanguage as glm
except ImportError:
    st.warning("⚠️ Pillow or google.ai.generativelanguage parts not available. Screenshot analysis disabled.")
    PIL_AVAILABLE = False

# --- Google API Imports & Check ---
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_LIBS_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Google API libraries not found. Google Doc export disabled.")
    GOOGLE_LIBS_AVAILABLE = False
    class HttpError(Exception): pass

# --- Gemini API Import & Check ---
try:
    import google.generativeai as genai
    GEMINI_LIBS_AVAILABLE = True
except ImportError:
    st.error("❌ Google Generative AI library not found! Please run: pip install google-generativeai")
    st.warning("⚠️ Gemini analysis features will be disabled.")
    GEMINI_LIBS_AVAILABLE = False

# --- Apify Client Check ---
try:
    from apify_client import ApifyClient
    APIFY_CLIENT_AVAILABLE = True
except ImportError:
    st.error("❌ Apify Client library not found! Please run: `pip install apify-client`")
    APIFY_CLIENT_AVAILABLE = False

# --- Configuration ---
GOOGLE_SERVICE_ACCOUNT_FILE = "service_account_key.json" # Replace with your actual path if not in root
WHATCMS_API_KEY = "w3xz6q7bamb7zixn1skvj2ei8wkz2xafrrjszv5fkk8yscm4019cim6wtgxuk13y20u2wu" # Replace if needed
GEMINI_API_KEY = "AIzaSyDAfqg0tqIkVAE_DV4vd6OOjJz_pXdnHso" # Replace with your actual key # <<-- NOTE: This key was truncated in the prompt, ensure your actual key is here

# --- API URLs & Settings ---
WHATCMS_API_URL = "https://whatcms.org/API/Tech"
GOOGLE_DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.file']
GOOGLE_DOCS_SCOPES = ['https://www.googleapis.com/auth/documents']
GOOGLE_ALL_SCOPES = GOOGLE_DRIVE_SCOPES + GOOGLE_DOCS_SCOPES
REQUESTS_TIMEOUT = 20
PLAYWRIGHT_TIMEOUT = 45000 # Default timeout for page navigation
SCROLL_DELAY_MS = 2000
MAX_SCROLLS = 5
GEMINI_MULTIMODAL_MODEL_NAME = "gemini-2.0-flash" # Using the previous stable flash model

# --- Prompts ---
COMBINED_ANALYSIS_SUMMARY_PROMPT = """
Analyze the following sequence of screenshots for a product page (domain: '{domain}'). Screenshots are provided for both Desktop and Mobile viewports, captured at different scroll positions (labeled D0, D1... M0, M1...).

Based *only* on these images, perform a visual UX critique and generate a structured report following the EXACT format below. Synthesize findings across all images for each point.

**Output Format (Use Markdown):**

Overall Summary:
[Provide a concise 1-2 paragraph summary. Include overall impression (e.g., clean, professional), key strengths (e.g., imagery, clear info), the most critical weakness observed (e.g., sticky CTA issue), and the primary opportunity for improvement based on the visual evidence.]

--- Product Page Findings: ---
[List 5-7 key findings synthesized from analyzing *all* screenshots (desktop and mobile). For each finding, use the precise sub-headings and formatting shown below. Focus on actionable UX/UI issues or strengths related to conversion, usability, layout, and trust, considering the product page context and differences between desktop/mobile.]

**Issue/Opportunity:**
[Describe the specific UX issue or strength observed across the screenshots. Be precise. e.g., "Primary 'Add to Cart' button is not sticky on desktop or mobile." or "Persistent Modal obscures view on page load." or "Image zoom functionality is broken on mobile." or "Footer lacks clear trust signals or delivery information." or "Lack of a sticky add to cart button creates friction."]
**Impact:**
[Explain the likely positive or negative effect on the user experience or conversion rate. e.g., "Users reviewing details must scroll back up, adding friction..."]
**Recommendation:**
[Suggest a specific, actionable improvement or way to maintain the strength. e.g., "Implement a sticky 'Add to Cart' button fixed at the bottom..." or "Ensure the modal can be easily dismissed and does not reappear aggressively." or "Fix image zoom for mobile devices." or "Add clear links to shipping, returns, and contact in the footer."]

**(Repeat the Issue/Opportunity, Impact, Recommendation structure for each finding)**

--- Key Recommendations Summary: ---
[List the top 3-5 most impactful, actionable recommendations derived from the findings above, as a numbered list.]
1.  [Recommendation 1]
2.  [Recommendation 2]
3.  [Recommendation 3]
4.  [Recommendation 4 (Optional)]
5.  [Recommendation 5 (Optional)]

**Analysis Guidance:**
*   Pay close attention to sticky/floating elements (nav, CTAs, chat) and their behavior/visibility/overlap across scrolls on both desktop and mobile.
*   Evaluate clarity, layout, consistency, CTAs, product info presentation, and overall trust signals.
*   Compare the desktop vs. mobile experience where relevant differences are observed.
*   If you observe issues related to modals, sticky elements (like 'sticky add to cart' or 'persistent CTA'), image zoom, navigation difficulties, or footer content, please describe them clearly under "Issue/Opportunity" and provide a "Recommendation".
"""

AD_IMAGE_ANALYSIS_PROMPT = """
Critically analyze the following ad creative image in conjunction with the provided ad text and headline. Provide a structured assessment focusing on the points below.

**Output Format (Use Markdown):**

--- Ad Creative Analysis: ---
[Critically assess the visual elements of the image. Is it high quality? Is it visually appealing and attention-grabbing? Does it clearly convey a message related to the product/offer? Identify potential visual hooks or distractions. Be critical and point out weaknesses.]

--- Ad Copy Analysis (Headline & Text): ---
[Critically analyze the written copy (headline and body text). Is it clear, concise, and compelling? Does it effectively communicate the value proposition? Is the call to action clear? Identify strengths and weaknesses in the messaging.]

--- Overall Ad Assessment: ---
[Assess how well the visual creative and written copy work together as a complete ad. Is the message consistent? Do they reinforce each other effectively? Is the overall ad likely to resonate with the target audience and drive action? Provide an overall critical evaluation.]

--- Opportunities for Improvement: ---
[Based on the analysis above, list specific, actionable opportunities to improve this ad. Focus on concrete suggestions for both the creative and the copy, and how they could be better integrated or optimized for performance.]
"""

# --- Configure Gemini ---
gemini_configured = False
if GEMINI_LIBS_AVAILABLE:
    # --- NOTE: Ensure your actual GEMINI_API_KEY is here, the one in the prompt was truncated ---
    if not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSyDAfqg0tqIkVAE_DV4vdOOJz_pXdnHso":
         st.error("❌ Gemini API Key is missing or looks incomplete. Please set a valid key.")
         GEMINI_LIBS_AVAILABLE = False # Prevent configuration attempt
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_configured = True
        except Exception as e: st.error(f"❌ Error configuring Gemini API: {e}"); GEMINI_LIBS_AVAILABLE = False

# --- Helper Function to Safely Extract Ad Details ---
def extract_ad_details(ad_item):
    if not isinstance(ad_item, dict):
        return {'headline': 'N/A', 'creative_text': 'N/A', 'image_url': None, 'cta_text': 'N/A', 'landing_page': '#'}
    headline, creative_text, image_url, cta_text, landing_page = 'N/A', 'N/A', None, 'N/A', '#'
    snapshot = ad_item.get('snapshot', {})
    if isinstance(snapshot, dict):
        cards = snapshot.get('cards', [])
        if isinstance(cards, list) and cards:
            first_card = cards[0] if isinstance(cards[0], dict) else {}
            headline = first_card.get('title', 'N/A')
            creative_text = first_card.get('body', 'N/A')
            cta_text = first_card.get('ctaText', 'N/A')
            landing_page = first_card.get('linkUrl', '#')
            image_url = first_card.get('videoPreviewImageUrl') or first_card.get('originalImageUrl') or first_card.get('resizedImageUrl')
    if headline == 'N/A' and 'title' in snapshot: headline = snapshot.get('title')
    if creative_text == 'N/A' and 'body' in snapshot: creative_text = snapshot.get('body')
    if cta_text == 'N/A' and 'ctaText' in snapshot: cta_text = snapshot.get('ctaText')
    if landing_page == '#' and 'linkUrl' in snapshot: landing_page = snapshot.get('linkUrl')
    if not image_url: image_url = snapshot.get('videoPreviewImageUrl') or snapshot.get('originalImageUrl') or snapshot.get('resizedImageUrl')
    if headline == 'N/A': headline = ad_item.get('title', 'N/A')
    if creative_text == 'N/A': creative_text = ad_item.get('body', ad_item.get('message', 'N/A'))
    if cta_text == 'N/A': cta_text = ad_item.get('ctaText', ad_item.get('callToActionText', 'N/A'))
    if landing_page == '#': landing_page = ad_item.get('linkUrl', '#')
    if not image_url: image_url = ad_item.get('videoPreviewImageUrl') or ad_item.get('originalImageUrl') or ad_item.get('resizedImageUrl')
    return {'headline': headline or 'N/A', 'creative_text': creative_text or 'N/A', 'image_url': image_url, 'cta_text': cta_text or 'N/A', 'landing_page': landing_page or '#'}

# --- Gemini Analysis Functions ---
def generate_analysis_and_summary(screenshots_dict, domain):
    if not gemini_configured or not GEMINI_LIBS_AVAILABLE: return "**Fallback Summary (Gemini Unavailable)**"
    if not PIL_AVAILABLE: return "**Fallback Summary (Pillow Unavailable)**"
    desktop_images, mobile_images = screenshots_dict.get("desktop", []), screenshots_dict.get("mobile", [])
    if not desktop_images and not mobile_images: return "No screenshots captured to analyze."
    try:
        model = genai.GenerativeModel(GEMINI_MULTIMODAL_MODEL_NAME)
        request_contents = [COMBINED_ANALYSIS_SUMMARY_PROMPT.format(domain=domain)]
        if desktop_images:
             request_contents.append("\n\n--- Desktop Screenshots ---")
             for i, img_bytes in enumerate(desktop_images):
                 try: Image.open(io.BytesIO(img_bytes)); img_part = glm.Part(inline_data=glm.Blob(mime_type="image/png", data=img_bytes)); request_contents.extend([f"Desktop Screenshot D{i}:", img_part])
                 except Exception as img_err: st.warning(f"Could not process desktop screenshot {i}: {img_err}")
        if mobile_images:
            request_contents.append("\n\n--- Mobile Screenshots ---")
            for i, img_bytes in enumerate(mobile_images):
                 try: Image.open(io.BytesIO(img_bytes)); img_part = glm.Part(inline_data=glm.Blob(mime_type="image/png", data=img_bytes)); request_contents.extend([f"Mobile Screenshot M{i}:", img_part])
                 except Exception as img_err: st.warning(f"Could not process mobile screenshot {i}: {img_err}")
        if len(request_contents) > 1:
            response = model.generate_content(request_contents, generation_config=genai.types.GenerationConfig(max_output_tokens=8192))
            if response.prompt_feedback.block_reason: return f"**Error: Analysis blocked by Gemini ({response.prompt_feedback.block_reason})**"
            if not response.parts: return "**Warning: Gemini returned no analysis text.**"
            return response.text
        else: return "Error: Could not process any valid images to send to Gemini."
    except Exception as e: st.error(f"❌ Gemini Multimodal API Error: {e}\n{traceback.format_exc()}"); return f"❌ Gemini Multimodal API Error: {e}"

def analyze_ad_image_with_gemini(image_url, ad_headline, ad_text):
    log_func = st.session_state.get("log_func", print); log_func(f"Attempting Gemini ad image analysis for {image_url[:60]}...")
    if not gemini_configured or not GEMINI_LIBS_AVAILABLE: return "**Fallback Analysis (Gemini Unavailable)**"
    if not PIL_AVAILABLE: return "**Fallback Analysis (Pillow Unavailable)**"
    if not image_url: return "No image URL provided for analysis."
    try:
        model = genai.GenerativeModel(GEMINI_MULTIMODAL_MODEL_NAME)
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}; response = requests.get(image_url, timeout=REQUESTS_TIMEOUT, headers=headers, stream=True); response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('image/'): return f"Error: URL did not return an image (Content-Type: {content_type})."
            img_bytes = response.content;
            if not img_bytes: return "Error: Fetched empty image content."
            try: Image.open(io.BytesIO(img_bytes))
            except Exception as img_v_err: return f"Error: Invalid or corrupted image data received ({img_v_err})."
            mime_type = content_type if content_type.startswith('image/') else "image/png"; img_part = glm.Part(inline_data=glm.Blob(mime_type=mime_type, data=img_bytes))
        except Exception as fetch_e: return f"Error fetching/processing image: {fetch_e}"
        prompt_text = AD_IMAGE_ANALYSIS_PROMPT + (f"\n\nAd Headline: {ad_headline}" if ad_headline and ad_headline != 'N/A' else "\n\nAd Headline: [Not Available]") + (f"\n\nAd Text: {ad_text}" if ad_text and ad_text != 'N/A' else "\n\nAd Text: [Not Available]")
        response = model.generate_content([prompt_text, img_part], generation_config=genai.types.GenerationConfig(max_output_tokens=4096))
        if response.prompt_feedback.block_reason: return f"**Error: Analysis blocked by Gemini ({response.prompt_feedback.block_reason})**"
        if not response.parts: return "**Warning: Gemini returned no analysis text for this ad.**"
        return response.text
    except Exception as e: log_func(f"❌ Gemini Ad Image Analysis Error ({image_url[:60]}): {e}\n{traceback.format_exc()}"); return f"❌ Gemini Ad Image Analysis Error: {e}"

def generate_overall_ad_quality_summary(ads_list):
    log_func = st.session_state.get("log_func", print); log_func("Attempting overall ad quality summary...")
    if not gemini_configured or not GEMINI_LIBS_AVAILABLE: return "**Fallback Summary (Gemini Unavailable)**"
    valid_critiques = [f"--- Ad {i+1} Critique ---\n{ad.get('gemini_ad_analysis_result', '')}\n\n" for i, ad in enumerate(ads_list) if ad.get('gemini_ad_analysis_result') and not ad.get('gemini_ad_analysis_result', '').startswith(("Error:", "**Fallback", "**Warning:", "No image URL"))]
    if not valid_critiques: return "No valid individual ad analyses found for overall summary."
    try:
        model = genai.GenerativeModel(GEMINI_MULTIMODAL_MODEL_NAME)
        prompt = f"Based *only* on the following individual ad critiques, synthesize a concise overall summary of ad quality. Focus on common strengths, weaknesses, themes in creative/copy, overall impression, and broad improvement areas. Provide a 1-2 paragraph summary.\n\n--- Individual Ad Critiques ---\n\n{''.join(valid_critiques)}"
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=2048))
        if response.prompt_feedback.block_reason: return f"**Error: Overall summary blocked by Gemini ({response.prompt_feedback.block_reason})**"
        if not response.parts: return "**Warning: Gemini returned no text for overall summary.**"
        return response.text
    except Exception as e: log_func(f"❌ Gemini Overall Summary Error: {e}\n{traceback.format_exc()}"); return f"❌ Gemini Overall Summary Error: {e}"

# --- Playwright Screenshot Functions ---
async def take_scrolling_screenshots_for_viewport(page, url, viewport_name):
    screenshots_bytes, log_func = [], st.session_state.get("log_func", print)
    try:
        # Changed 'networkidle' to 'load' for more reliability
        await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT, wait_until='load')
        await page.wait_for_timeout(3000) # Allow some time for JS to settle after load
        screenshots_bytes.append(await page.screenshot(full_page=False, type='png'))
        last_scroll_y, viewport_height = 0, page.viewport_size['height'] if page.viewport_size else 800
        for i in range(1, MAX_SCROLLS + 1):
            await page.evaluate(f'window.scrollBy(0, {int(viewport_height * 0.9)})'); await page.wait_for_timeout(SCROLL_DELAY_MS)
            current_scroll_y = await page.evaluate("window.scrollY")
            if current_scroll_y <= last_scroll_y: log_func(f"Scrolling stopped {viewport_name} scroll {i}."); break
            screenshots_bytes.append(await page.screenshot(full_page=False, type='png'))
            if await page.evaluate('() => window.scrollY + window.innerHeight >= document.body.scrollHeight - 10'): log_func(f"Reached bottom {viewport_name} scroll {i}."); break
            last_scroll_y = current_scroll_y
    except Exception as e: log_func(f"❌ Screenshot Error ({viewport_name}): {e}"); st.error(f"❌ Screenshot Error ({viewport_name}): {e}")
    return screenshots_bytes

async def capture_desktop_and_mobile_screenshots(url: str):
    if not PLAYWRIGHT_AVAILABLE: return {}
    results, browser, p, log_func = {"desktop": [], "mobile": []}, None, None, st.session_state.get("log_func", print)
    try:
        p = await async_playwright().start(); browser = await p.chromium.launch(headless=True)
        log_func("--- Desktop View (1280x800) ---")
        desktop_context = await browser.new_context(viewport={'width': 1280, 'height': 800}, user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        desktop_page = await desktop_context.new_page(); results["desktop"] = await take_scrolling_screenshots_for_viewport(desktop_page, url, "Desktop")
        await desktop_page.close(); await desktop_context.close()
        log_func("--- Mobile View (iPhone 13) ---")
        mobile_context = await browser.new_context(**p.devices['iPhone 13']); mobile_page = await mobile_context.new_page()
        results["mobile"] = await take_scrolling_screenshots_for_viewport(mobile_page, url, "Mobile")
        await mobile_page.close(); await mobile_context.close()
        return results
    except Exception as e: log_func(f"❌ Playwright main capture error: {e}\n{traceback.format_exc()}"); st.error(f"❌ Playwright main capture error: {e}"); return results
    finally:
        if browser: await browser.close()
        if p: await p.stop()

def run_playwright_sync(url: str):
    if not PLAYWRIGHT_AVAILABLE: return {}
    log_func = st.session_state.get("log_func", print) # Ensure log_func is available
    screenshots_dict = {}

    # Define the async part of the work
    async def main_operation():
        nonlocal screenshots_dict
        # This function `capture_desktop_and_mobile_screenshots` is async and needs to be awaited
        screenshots_dict = await capture_desktop_and_mobile_screenshots(url)

    try:
        # Attempt to get the current event loop.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e_get_loop:
            # This is the "There is no current event loop" scenario from get_event_loop().
            if "no current event loop" in str(e_get_loop).lower():
                log_func(f"No current event loop detected in thread '{threading.current_thread().name}' ('{e_get_loop}'). Using asyncio.run() to create and manage loop.")
                # asyncio.run() will create a new event loop, run the coroutine, and close it.
                asyncio.run(main_operation())
                return screenshots_dict
            else:
                # A different RuntimeError from get_event_loop(), re-raise to be caught by the outer generic handler.
                log_func(f"Unexpected RuntimeError from get_event_loop: {e_get_loop}")
                raise

        # If we got here, get_event_loop() succeeded. Now check if it's running.
        if loop.is_running():
            # If a loop is already running in this thread, we can't use loop.run_until_complete()
            # or asyncio.run() directly without conflict.
            # Running the async operation in a new thread is a common solution.
            log_func(f"Event loop is already running in thread '{threading.current_thread().name}'. Launching Playwright in a new thread with asyncio.run().")

            thread_result_holder = [None] # Using a list to hold the result from the thread

            def thread_target():
                try:
                    # In the new thread, asyncio.run() can create and manage its own event loop.
                    # Pass the original async function that does the capture.
                    thread_result_holder[0] = asyncio.run(capture_desktop_and_mobile_screenshots(url))
                except Exception as ex_thread:
                    log_func(f"Error in Playwright thread: {ex_thread}\n{traceback.format_exc()}")
                    thread_result_holder[0] = {} # Default to empty on error

            thread = threading.Thread(target=thread_target)
            thread.start()
            thread.join() # Wait for the thread to complete.
            return thread_result_holder[0]
        else:
            # Loop exists but is not running. We can use it.
            log_func(f"Event loop exists but is not running in thread '{threading.current_thread().name}'. Using run_until_complete.")
            loop.run_until_complete(main_operation())
            return screenshots_dict

    except Exception as e_general:
        # Catch-all for other errors, including re-raised ones, direct RuntimeErrors from asyncio.run() itself
        # (e.g. "cannot be called from a coroutine"), or any other unexpected exceptions.
        st.error(f"❌ Playwright sync wrapper error: {e_general}\n{traceback.format_exc()}")
        return {}

# --- Helper Functions ---
def get_domain_from_url(url):
    if not url: return None
    try:
        current_url = url if url.startswith(('http://', 'https://')) else 'https://' + url
        domain = urlparse(current_url).netloc.split(':')[0]
        return domain[4:] if domain.lower().startswith('www.') else domain
    except: return None

def get_page_title(url):
    log_func = st.session_state.get("log_func", print)
    if not url: return "N/A"
    try:
        current_url = url if url.startswith(('http://', 'https://')) else 'https://' + url
        headers = { # Added common headers
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        response = requests.get(current_url, timeout=REQUESTS_TIMEOUT, headers=headers, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Ignore comments when finding title
        for element in soup.find_all(string=lambda text: isinstance(text, Comment)):
            element.extract()
        title_tag = soup.find('title')
        return title_tag.string.strip() if title_tag and title_tag.string else "N/A (Title tag not found or empty)"
    except Exception as e: log_func(f"Error fetching page title for {url}: {e}"); return f"N/A (Error: {type(e).__name__})"

def get_cms_with_whatcms(url):
    if not url: return "URL not provided."
    if not WHATCMS_API_KEY: return "CMS Check Failed (Missing API Key)"
    current_url = url if url.startswith(('http://', 'https://')) else 'https://' + url
    params = {"key": WHATCMS_API_KEY, "url": current_url}
    log_func = st.session_state.get("log_func", print)
    log_func(f"Querying WhatCMS for: {current_url}")
    try:
        # --- MODIFIED SECTION ---
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01', # WhatCMS API might prefer JSON accept
            'Accept-Language': 'en-US,en;q=0.9'
        }
        response = requests.get(WHATCMS_API_URL, params=params, timeout=REQUESTS_TIMEOUT, headers=headers)
        # --- END MODIFIED SECTION ---
        log_func(f"WhatCMS response status: {response.status_code}")
        if response.status_code == 403:
             log_func(f"WhatCMS returned 403 Forbidden. Raw response: {response.text[:200]}")
             return "CMS Check Failed (Forbidden - API Key issue or WhatCMS block?)"
        response.raise_for_status()
        data = response.json()
        log_func(f"WhatCMS JSON response (first 200 chars): {str(data)[:200]}")

        result_info = data.get('result', {})
        if result_info.get('code') == 200:
            detected_cms_technologies = [] # List to store all detected CMS technologies

            if 'results' in data and isinstance(data['results'], list):
                for tech in data['results']:
                    categories = tech.get('categories', [])
                    # Check if "CMS" is in the categories for this technology
                    if isinstance(categories, (list, tuple)) and "CMS" in categories:
                        cms_name = tech.get('name', 'Unknown CMS')
                        cms_confidence = tech.get('confidence', 0)
                        detected_cms_technologies.append(f"{cms_name} (Confidence: {cms_confidence}%)")

            if detected_cms_technologies:
                return ", ".join(detected_cms_technologies) # Join all found CMS technologies
            else:
                return "No technologies categorized as CMS detected by WhatCMS."
        else:
            # Original error handling for API codes
            error_code = result_info.get('code', 'N/A'); error_msg = result_info.get('msg', 'Unknown error')
            if error_code == 120: return "CMS Check Failed (Rate Limit)"
            if error_code == 101: return f"CMS Check Failed (WhatCMS API Error: {error_msg} - URL may be inaccessible to WhatCMS)"
            return f"CMS Check Failed (API Code: {error_code} - {error_msg})"

    except requests.exceptions.Timeout: return "CMS Check Failed (Timeout connecting to WhatCMS)"
    except requests.exceptions.HTTPError as e: return f"CMS Check Failed (HTTP Error: {e.response.status_code} - {e.response.reason})"
    except requests.exceptions.RequestException as e: return f"CMS Check Failed (Request Error: {type(e).__name__})"
    except json.JSONDecodeError: return "CMS Check Failed (Invalid JSON response from WhatCMS)"
    except Exception as e: return f"CMS Check Failed (Unexpected Error: {type(e).__name__})"

def export_to_google_doc(summary_text, folder_id=None):
    if not GOOGLE_LIBS_AVAILABLE: return "❌ Error: Google API libraries not installed."
    if not summary_text or not isinstance(summary_text, str) or summary_text.strip() == "": return "❌ Error: No valid summary text for export."
    try:
        creds = service_account.Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=GOOGLE_ALL_SCOPES)
        drive_service = build('drive', 'v3', credentials=creds); docs_service = build('docs', 'v1', credentials=creds)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        url_for_title = st.session_state.get('pp_url') or st.session_state.get('hp_url') or 'Audit'
        domain_for_title = get_domain_from_url(url_for_title) or "Audit"
        document_title = f"Product Page UX Summary - {domain_for_title} - {timestamp}"
        doc = docs_service.documents().create(body={'title': document_title}).execute()
        document_id, doc_link = doc.get('documentId'), f"https://docs.google.com/document/d/{document_id}/edit"
        st.info(f"📄 Created Google Doc: {doc_link}")
        if folder_id:
            try:
                file_meta = drive_service.files().get(fileId=document_id, fields='parents').execute()
                drive_service.files().update(fileId=document_id, addParents=folder_id, removeParents=",".join(file_meta.get('parents',[])), fields='id,parents').execute()
                st.info(f"📂 Moved to Google Drive folder: {folder_id}")
            except Exception as move_e: st.warning(f"⚠️ Could not move doc: {move_e}")
        cleaned_summary = summary_text.replace("**", "").replace("*", "").replace("\n### ", "\n").replace("\n## ", "\n").replace("\n# ", "\n").replace("\n- ", "\n\t- ").replace("\n1. ", "\n\t1. ")
        docs_service.documents().batchUpdate(documentId=document_id, body={'requests': [{'insertText': {'location': {'index': 1}, 'text': cleaned_summary}}]}).execute()
        return f"✅ Successfully exported to Google Doc: [View Document]({doc_link})"
    except FileNotFoundError: return f"❌ Export Failed: Service account key file not found at '{GOOGLE_SERVICE_ACCOUNT_FILE}'."
    except Exception as e: st.error(f"⚙️ Google Doc export error: {e}\n{traceback.format_exc()}"); return f"❌ Export Failed: {type(e).__name__}"

def detect_facebook_page(url):
    if not url: return None
    log_func = st.session_state.get("log_func", print)
    current_url = url if url.startswith(('http://', 'https://')) else 'https://' + url
    try:
        # --- MODIFIED SECTION ---
        # Using a more comprehensive set of headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br', # requests handles standard encodings
            'DNT': '1', # Do Not Track
            'Upgrade-Insecure-Requests': '1',
            'Referer': urlparse(current_url).scheme + "://" + urlparse(current_url).netloc + "/" # Add a referer
        }
        # log_func(f"Attempting FB detection GET to {current_url} with headers: {json.dumps(headers, indent=2)}")
        response = requests.get(
            current_url,
            timeout=REQUESTS_TIMEOUT,
            headers=headers,
            allow_redirects=True # Allow redirects to follow to the final page
        )
        # log_func(f"FB detection response status for {current_url}: {response.status_code}")
        # if response.status_code != 200 :
            # log_func(f"FB detection response content for {current_url} (status {response.status_code}): {response.text[:500]}")
        # --- END MODIFIED SECTION ---
        response.raise_for_status()
        base_url = response.url # Use the final URL after redirects
        soup, potential_links = BeautifulSoup(response.text, 'html.parser'), []

        common_fb_patterns = [
            'facebook.com/',
            'fb.me/',
            'fb.com/'
        ]

        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')): continue
            href_lower = href.lower()
            if any(pattern in href_lower for pattern in common_fb_patterns):
                 if not any(ex in href_lower for ex in ['sharer', 'share.php', 'dialog', 'plugin', 'login', 'logout']):
                     try:
                         absolute_href = urljoin(base_url, href)
                         parsed_abs_url = urlparse(absolute_href)
                         if 'facebook.com' in parsed_abs_url.netloc.lower():
                             potential_links.append(absolute_href)
                     except ValueError:
                         continue

        for element in soup.find_all(attrs={"aria-label": re.compile(r"facebook", re.IGNORECASE)}):
            if element.name == 'a' and element.get('href'):
                 href = element['href']
                 if not href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                     if not any(ex in href.lower() for ex in ['sharer', 'share.php', 'dialog', 'plugin', 'login', 'logout']):
                         try:
                             potential_links.append(urljoin(base_url, href))
                         except ValueError:
                             continue
        if potential_links:
            unique_links = set()
            for plink in potential_links: # Changed variable name from 'link' to 'plink' to avoid conflict
                try:
                    parsed = urlparse(plink)
                    clean_query = '&'.join(p for p in parsed.query.split('&') if not p.startswith(('fbclid=', 'ref=', 'utm_')))
                    clean_url = parsed._replace(query=clean_query).geturl()
                    unique_links.add(clean_url)
                except: unique_links.add(plink)
            sorted_unique_links = sorted(list(unique_links), key=len)
            preferred_links = [
                ulink for ulink in sorted_unique_links # Changed variable name
                if not re.search(r'/(posts|videos|photos|events|notes)/', ulink.lower())
                   and '/permalink.php' not in ulink.lower()
                   and '/story.php' not in ulink.lower()
            ]
            if preferred_links: return preferred_links[0]
            elif sorted_unique_links: return sorted_unique_links[0]
        return None
    except requests.exceptions.HTTPError as e_http:
        log_func(f"FB detection HTTPError for {current_url}: Status {e_http.response.status_code} - {e_http.response.reason}. URL: {e_http.request.url}")
        return None
    except requests.exceptions.RequestException as e_req:
        log_func(f"FB detection RequestException for {current_url}: {e_req}")
        return None
    except Exception as e_generic:
        log_func(f"FB detection error for {current_url}: {e_generic}\n{traceback.format_exc()}")
        return None

def get_funnel_stage(cta_text):
    if not cta_text or not isinstance(cta_text, str): return 3
    cta = cta_text.lower()
    if any(t in cta for t in ["shop now", "buy now", "order now", "get offer", "get deal", "install now", "play game", "donate now"]): return 3
    if any(t in cta for t in ["sign up", "subscribe", "download", "get quote", "apply now", "contact us", "register"]): return 2
    if any(t in cta for t in ["learn more", "watch more", "see more", "visit website", "listen now", "see menu"]): return 1
    return 3

# --- Homepage Extraction Function ---
def extract_from_homepage(homepage_url):
    log_func = st.session_state.get("log_func", print)
    if not homepage_url:
        return {"hero_products": [], "site_categories": []}

    products = set()
    categories = set()
    common_exclusions = { # Expanded list
        'home', 'about', 'contact', 'us', 'blog', 'news', 'cart', 'checkout', 'my account', 'account', 'login', 'log in', 'register', 'sign up',
        'shop', 'products', 'product', 'categories', 'category', 'all', 'new', 'sale', 'deals', 'offers', 'clearance', 'featured', 'popular',
        'privacy policy', 'privacy', 'terms', 'conditions', 'shipping', 'delivery', 'returns', 'refunds',
        'faq', 'customer service', 'service', 'more', 'view', 'details', 'read', 'search', 'filter', 'sort', 'help', 'support',
        'information', 'company', 'services', 'resources', 'gallery', 'portfolio', 'testimonials', 'sitemap', 'careers', 'jobs',
        'track order', 'wishlist', 'compare', 'locations', 'store locator', 'gift card', 'payment', 'warranty', 'size guide', 'manuals'
    }
    try:
        current_url = homepage_url if homepage_url.startswith(('http://', 'https://')) else 'https://' + homepage_url
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
            }
        response = requests.get(current_url, timeout=REQUESTS_TIMEOUT, headers=headers, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, footer, header potentially
        for tag in soup(['script', 'style', 'footer', 'header', 'aside', 'form']):
            tag.decompose()

        # --- Product Extraction Heuristics ---
        product_candidates = []
        # 1. Elements with product-related classes/ids
        product_containers = soup.select('[class*="product"], [class*="item"], [class*="card"], [id*="product"]')
        for container in product_containers:
            # Look for headings or distinctive text within
            title_element = container.find(['h2','h3','h4','h5','p[class*="title"]', 'a[class*="name"]'])
            if title_element:
                text = title_element.get_text(strip=True)
                if text and 5 < len(text) < 100 and text.lower() not in common_exclusions:
                    # Check for price nearby as a strong indicator
                    if container.find(string=re.compile(r'[\$£€]\s?\d+')):
                        product_candidates.append(text)
            # Check image alt text within product containers
            img = container.find('img', alt=True)
            if img and img['alt'] and 5 < len(img['alt']) < 100 and img['alt'].lower() not in common_exclusions:
                 if container.find(string=re.compile(r'[\$£€]\s?\d+')):
                      product_candidates.append(img['alt'].strip())

        # 2. Image alt text if the image is linked (common pattern)
        for link_tag in soup.find_all('a', href=True): # Renamed variable to avoid conflict
            img = link_tag.find('img', alt=True)
            if img and img['alt']:
                alt_text = img['alt'].strip()
                if alt_text and 5 < len(alt_text) < 100 and alt_text.lower() not in common_exclusions:
                    # Check if sibling text contains price
                    if link_tag.find_next_sibling(string=re.compile(r'[\$£€]\s?\d+')) or \
                       link_tag.find_parent().find(string=re.compile(r'[\$£€]\s?\d+')):
                           product_candidates.append(alt_text)

        # Deduplicate and add to the final set
        for p in product_candidates: products.add(p)


        # --- Category Extraction Heuristics ---
        category_candidates = []
        # 1. Navigation menus
        for nav in soup.select('nav, ul[class*="menu"], ul[class*="nav"], div[class*="nav"], div[id*="nav"]'):
             parent_classes = (nav.find_parent().get('class') if nav.find_parent() else []) or []
             is_likely_main_nav = any(cls in parent_classes for cls in ['header', 'main-nav-wrapper']) or nav.get('id') == 'main-nav'

             for link_tag_nav in nav.find_all('a', href=True): # Renamed variable
                 text = link_tag_nav.get_text(strip=True)
                 href = link_tag_nav.get('href', '')
                 href_lower = href.lower()

                 if not text or not (3 < len(text) < 35) or text.lower() in common_exclusions or text.isdigit(): continue
                 if href.startswith(('#', 'javascript:', 'mailto:')) or href == '/' or urlparse(href).netloc: continue
                 if any(ex in href_lower for ex in ['/product/', '/products/', '.html', '.php', 'login', 'cart', 'account', 'blog', 'contact']): continue

                 if is_likely_main_nav or any(hint in href_lower for hint in ['/category/', '/categories/', '/collection/', '/collections/', '/shop/']):
                     if text not in products:
                         category_candidates.append(text)
        seen_cats = set()
        for cat in category_candidates:
             cat_cleaned = re.sub(r'(?i)^(shop|view)\s+', '', cat).strip()
             if cat_cleaned and cat_cleaned.lower() not in seen_cats:
                 categories.add(cat_cleaned)
                 seen_cats.add(cat_cleaned.lower())

        log_func(f"Homepage extraction: Found {len(products)} potential products, {len(categories)} potential categories.")
        return {
            "hero_products": sorted(list(products))[:10],
            "site_categories": sorted(list(categories))[:10]
        }
    except requests.exceptions.RequestException as e:
        log_func(f"Error fetching homepage {homepage_url}: {e}")
    except Exception as e:
        log_func(f"Error parsing homepage {homepage_url}: {e}\n{traceback.format_exc()}")
    return {"hero_products": [], "site_categories": []}


# --- Executive Summary Generation Functions ---
def construct_meta_ads_library_search_link(fb_page_url):
    if not fb_page_url: return None
    try:
        parsed_url = urlparse(fb_page_url)
        path_segments = [segment for segment in parsed_url.path.strip('/').split('/') if segment and segment.lower() != 'pages']
        search_term = ""
        if path_segments:
            potential_terms = [s for s in reversed(path_segments) if s.lower() not in ['home', 'about', 'photos', 'videos', 'posts']]
            if potential_terms: search_term = potential_terms[0]
            elif parsed_url.query:
                query_params = dict(qc.split("=") for qc in parsed_url.query.split("&") if "=" in qc)
                if 'id' in query_params: search_term = query_params['id']
        if not search_term and parsed_url.netloc == "www.facebook.com" and path_segments: search_term = path_segments[-1]
        if search_term:
            if '-' in search_term and search_term.split('-')[-1].isdigit(): search_term = search_term.rsplit('-',1)[0]
            return f"https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=ALL&q={quote_plus(search_term)}&search_type=page&media_type=all"
        return None
    except: return None

def parse_gemini_ux_summary_for_exec_summary(gemini_summary_text):
    log_func = st.session_state.get("log_func", print)
    gaps = {"Above-the-Fold Clarity": [], "Image & Visual Trust Cues": [], "CTA & Checkout Flow": [], "Mobile Optimisation": [], "Conversion Elements": []}
    if not gemini_summary_text or not isinstance(gemini_summary_text, str) or "--- Product Page Findings: ---" not in gemini_summary_text:
        for key in gaps: gaps[key] = ["N/A (Summary not parsed or findings missing)"]
        return gaps
    findings_section = gemini_summary_text.split("--- Product Page Findings: ---", 1)[-1].split("--- Key Recommendations Summary: ---", 1)[0]
    finding_pattern = re.compile(r"\*\*Issue/Opportunity:\*\*\s*(.*?)\s*(?:\*\*Impact:\*\*\s*(.*?)\s*)?\*\*Recommendation:\*\*\s*(.*?)\s*(?=(\*\*Issue/Opportunity:\*\*|\Z))", re.DOTALL | re.IGNORECASE)
    for match in finding_pattern.finditer(findings_section):
        issue_text, recommendation_text = (match.group(1) or "").strip(), (match.group(3) or "No specific recommendation.").strip()
        summary_rec = (recommendation_text.split('.')[0] + ".")[:120]
        issue_lower = issue_text.lower()
        if any(k in issue_lower for k in ["modal", "popup", "above the fold", "above-the-fold"]): gaps["Above-the-Fold Clarity"].append(summary_rec)
        if any(k in issue_lower for k in ["cta", "call to action", "checkout", "cart button", "sticky cart", "add to cart", "persistent cta"]): gaps["CTA & Checkout Flow"].append(summary_rec)
        if any(k in issue_lower for k in ["image", "visual", "zoom", "gallery"]): gaps["Image & Visual Trust Cues"].append(summary_rec)
        if any(k in issue_lower for k in ["mobile", "responsive"]) or ("navigation" in issue_lower and ("mobile" in issue_lower or "small screen" in issue_lower)):
            if not (("modal" in issue_lower) and summary_rec in gaps["Mobile Optimisation"]): gaps["Mobile Optimisation"].append(summary_rec)
        if any(k in issue_lower for k in ["footer", "trust signal", "delivery info", "review", "conversion element"]): gaps["Conversion Elements"].append(summary_rec)
    for key, value_list in gaps.items():
        unique_notes = list(dict.fromkeys(value_list)) if value_list else []
        gaps[key] = "<br>".join(f"- {note}" for note in unique_notes) if unique_notes else "No specific gaps identified from summary for this category."
    return gaps

def parse_meta_ads_data_for_exec_summary(ads_list, overall_ad_summary_text, product_url):
    log_func = st.session_state.get("log_func", print)
    gaps = {"Active Ads": "N/A", "Ad Duration": "N/A", "Ad Variations": "N/A", "Creative Quality": "N/A", "Copywriting": "N/A", "Consistency": "Review ad creatives/copy against website UX/tone.", "Destination Relevance": "N/A"}
    if not ads_list: return gaps
    num_ads, active_ads_count = len(ads_list), sum(1 for ad in ads_list if ad.get('isActive'))
    gaps["Active Ads"] = f"{active_ads_count} active of {num_ads} analyzed." if num_ads > 0 else "No ads data."

    durations = []
    for ad in ads_list:
        start_date_str = ad.get('startDateFormatted')
        if start_date_str:
            try:
                launch_date = datetime.datetime.strptime(start_date_str.split('T')[0], '%Y-%m-%d').date()
                days = (datetime.date.today() - launch_date).days
                if days >= 0:
                    durations.append(days)
            except ValueError:
                if log_func: log_func(f"Could not parse date for ad duration: {start_date_str}")
            except Exception as e:
                if log_func: log_func(f"Error calculating ad duration for {start_date_str}: {e}")

    if durations: gaps["Ad Duration"] = f"{min(durations)}-{max(durations)} days (avg: {sum(durations)//len(durations)})." if len(durations) > 1 else f"{durations[0]} days."
    else: gaps["Ad Duration"] = "Durations N/A."

    if num_ads > 1:
        h_set = set()
        b_set = set()
        i_set = set()
        for ad in ads_list:
            details = extract_ad_details(ad)
            headline = details.get('headline', 'N/A')
            if not isinstance(headline, str): headline = str(headline) if headline is not None else 'N/A'
            h_set.add(headline)
            creative_text = details.get('creative_text', 'N/A')
            if not isinstance(creative_text, str): creative_text = str(creative_text) if creative_text is not None else 'N/A'
            b_set.add(creative_text)
            image_url = details.get('image_url')
            if image_url:
                if not isinstance(image_url, str): image_url = str(image_url)
                i_set.add(image_url)
        gaps["Ad Variations"] = f"Variations observed ({len(h_set)}H, {len(b_set)}B, {len(i_set)}I)." if len(h_set)>1 or len(b_set)>1 or len(i_set)>1 else "Limited variations."
    elif num_ads == 1: gaps["Ad Variations"] = "Single ad; variations N/A."
    else: gaps["Ad Variations"] = "No ads for variation check."

    first_ad_analysis = ads_list[0].get('gemini_ad_analysis_result', '') if ads_list else ''
    if overall_ad_summary_text and not overall_ad_summary_text.startswith(("Error:", "**Fallback")):
        gaps["Creative Quality"], gaps["Copywriting"] = "Overall: " + overall_ad_summary_text[:150] + "...", "Overall: " + overall_ad_summary_text[:150] + "..."
    elif first_ad_analysis and not first_ad_analysis.startswith(("Error:", "**Fallback")):
        if "--- Ad Creative Analysis: ---" in first_ad_analysis: gaps["Creative Quality"] = first_ad_analysis.split("--- Ad Creative Analysis: ---",1)[1].split("---",1)[0].strip()[:150]+"..."
        if "--- Ad Copy Analysis (Headline & Text): ---" in first_ad_analysis: gaps["Copywriting"] = first_ad_analysis.split("--- Ad Copy Analysis (Headline & Text): ---",1)[1].split("---",1)[0].strip()[:150]+"..."

    relevant_lps, main_domain = 0, get_domain_from_url(product_url)
    if main_domain and num_ads > 0:
        for ad in ads_list:
            lp_details = extract_ad_details(ad)
            landing_page_url = lp_details.get('landing_page')
            if isinstance(landing_page_url, str) and get_domain_from_url(landing_page_url) == main_domain:
                 relevant_lps +=1
            elif landing_page_url is not None and not isinstance(landing_page_url, str):
                 log_func(f"Warning: Landing page URL for ad is not a string: {landing_page_url}")
        gaps["Destination Relevance"] = f"{relevant_lps}/{num_ads} to main domain. Check tracking."
    else: gaps["Destination Relevance"] = "LP relevance N/A."
    return gaps

def generate_executive_summary_markdown(
    prospect_name, website_url, report_date,
    hero_products_list, detected_site_categories_list,
    tech_stack_cms_result, usp, audience, competitors,
    facebook_page_url, product_page_url_audited, ux_audit_gaps,
    meta_ads_library_search_link, meta_ads_audit_gaps
):
    def val(data, default="N/A"): return data if data and str(data).strip() and str(data).lower() != 'n/a' else default
    def list_to_str(lst, max_items=7, empty_msg="Not identified"):
        if not lst: return empty_msg
        return ", ".join(lst[:max_items]) + ("..." if len(lst) > max_items else "")

    md = f"## {val(prospect_name, 'Prospect')} – Car Parts Audit\n"
    md += f"**Website:** {val(website_url)}\n"
    md += f"**Date:** {val(report_date)}\n\n"
    md += "### 1. Precision Audit – Prospect Overview\n"
    md += f"- **Hero Products (from Homepage/Title):** {list_to_str(hero_products_list, 7, 'Not prominently identified.')}\n"
    md += f"- **Detected Site Categories (from Homepage):** {list_to_str(detected_site_categories_list, 7, 'Not prominently identified.')}\n"
    md += f"- **Tech Stack:** {val(tech_stack_cms_result, 'CMS check not run or failed.')}\n"
    if facebook_page_url: md += f"- **Facebook Page:** [{facebook_page_url}]({facebook_page_url})\n"
    else: md += f"- **Facebook Page:** Not detected or not applicable\n"
    md += "- **Positioning Summary:** (Manual input recommended for accuracy)\n"
    md += f"  - **USP:** {val(usp, 'To be defined')}\n"
    md += f"  - **Audience:** {val(audience, 'To be defined')}\n"
    md += f"  - **Competitors:** {val(competitors, 'To be defined')}\n\n"
    md += "### 2. Conversion Engine – Product Page UX Audit\n"
    md += f"**Page Audited:** {val(product_page_url_audited)}\n\n"
    md += "| What to Check                 | Why it Matters                          | Gaps or Opportunities                                           |\n"
    md += "|-------------------------------|-----------------------------------------|-----------------------------------------------------------------|\n"
    md += f"| Above-the-Fold Clarity        | Key info must be visible instantly      | {val(ux_audit_gaps.get('Above-the-Fold Clarity'), 'No specific gaps noted.')} |\n"
    md += f"| Image & Visual Trust Cues     | Increases product confidence            | {val(ux_audit_gaps.get('Image & Visual Trust Cues'), 'No specific gaps noted.')} |\n"
    md += f"| CTA & Checkout Flow           | Frictionless = more conversions         | {val(ux_audit_gaps.get('CTA & Checkout Flow'), 'No specific gaps noted.')} |\n"
    md += f"| Mobile Optimisation           | Majority of traffic = mobile            | {val(ux_audit_gaps.get('Mobile Optimisation'), 'No specific gaps noted.')} |\n"
    md += f"| Conversion Elements           | Trust, delivery info, reviews           | {val(ux_audit_gaps.get('Conversion Elements'), 'No specific gaps noted.')} |\n\n"
    md += "### 3. Brand Power Play – Meta Ads Audit\n"
    if meta_ads_library_search_link: md += f"**Meta Ads Library (Search):** [{meta_ads_library_search_link}]({meta_ads_library_search_link})\n\n"
    else: md += "**Meta Ads Library (Search):** Facebook Page not detected or link could not be constructed.\n\n"
    md += "| What to Check                 | Why it Matters                          | Gaps or Opportunities                                                  |\n"
    md += "|-------------------------------|-----------------------------------------|------------------------------------------------------------------------|\n"
    md += f"| Active Ads                    | Shows awareness strategy                | {val(meta_ads_audit_gaps.get('Active Ads'), 'N/A')}                     |\n"
    md += f"| Ad Duration                   | Stale vs. tested creative               | {val(meta_ads_audit_gaps.get('Ad Duration'), 'N/A')}                  |\n"
    md += f"| Ad Variations                 | Funnel coverage                         | {val(meta_ads_audit_gaps.get('Ad Variations'), 'N/A')}                |\n"
    md += f"| Creative Quality              | Hooks, resolution, emotion              | {val(meta_ads_audit_gaps.get('Creative Quality'), 'N/A')}             |\n"
    md += f"| Copywriting                   | Tone, CTAs, clarity                     | {val(meta_ads_audit_gaps.get('Copywriting'), 'N/A')}                  |\n"
    md += f"| Consistency                   | With site & tone                        | {val(meta_ads_audit_gaps.get('Consistency'), 'Review recommended.')}   |\n"
    md += f"| Destination Relevance         | Tracking, post-click UX                 | {val(meta_ads_audit_gaps.get('Destination Relevance'), 'N/A')}        |\n\n"
    md += "*Inspiration: Snap Shades – UGC + model compatibility + emotional pain point ads*\n"
    return md

def extract_top_recommendations(gemini_page_summary, overall_ad_summary_text):
    log_func = st.session_state.get("log_func", print)
    recommendations = []
    if gemini_page_summary and isinstance(gemini_page_summary, str):
        match = re.search(r"--- Key Recommendations Summary: ---\s*\n(.*?)(?=\n\n---|\Z)", gemini_page_summary, re.DOTALL | re.IGNORECASE)
        if match:
            for line in match.group(1).strip().split('\n'):
                line_clean = re.sub(r"^\s*\d+\.\s*", "", line).strip()
                if line_clean and len(line_clean) > 10: recommendations.append(line_clean)
        else: log_func("Could not find 'Key Recommendations Summary' in page summary.")
    if overall_ad_summary_text and isinstance(overall_ad_summary_text, str):
        match = re.search(r"--- Opportunities for Improvement: ---\s*\n(.*?)(?=\n\n---|\Z)", overall_ad_summary_text, re.DOTALL | re.IGNORECASE)
        if match:
            for line in match.group(1).strip().split('\n'):
                line_clean = re.sub(r"^\s*[\*\-\–\d+\.]+\s*", "", line).strip()
                if line_clean and len(line_clean) > 10: recommendations.append(f"Ad Improvement: {line_clean}")
        else: log_func("Could not find 'Opportunities for Improvement' in ad summary.")
    unique_recommendations, seen = [], set()
    for rec in recommendations:
        normalized_rec = rec.lower().strip().replace("ad improvement: ", "")
        if normalized_rec not in seen: unique_recommendations.append(rec); seen.add(normalized_rec)
    return unique_recommendations[:5]

# --- Main App Layout Starts Here ---
st.title("🤖 Gemini Product Page & Ad Auditor")
st.markdown("Analyzes product page screenshots, homepage content, CMS, Facebook Ads, and generates comprehensive reports.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Analysis Inputs")
    product_page_url_input = st.text_input("Product Page URL (Required for UX Audit):", placeholder="https://example.com/product/xyz", key="pp_url")
    homepage_url_input = st.text_input("Homepage URL (Required for some info):", placeholder="https://example.com", key="hp_url")
    manual_fb_page_url = st.text_input("Facebook Page URL (Optional Manual Override):", placeholder="https://facebook.com/yourpage", key="fb_manual_url")

    st.header("Export Options")
    google_drive_folder_id = st.text_input("Google Drive Folder ID (Optional):", placeholder="Enter Folder ID", key="gdrive_id")

    st.header("Additional Checks")
    do_check_facebook = st.checkbox("Attempt to Auto-Detect Facebook Page Link?", value=True, key="fb_check", help="If unchecked or fails, Manual URL above will be used if provided.")
    apify_disabled = not (APIFY_CLIENT_AVAILABLE and META_ADS_AUDIT_AVAILABLE)
    do_check_meta_ads = st.checkbox("Scrape Meta Ads Library via Apify?", value=False, key="ads_check", disabled=apify_disabled)

# --- Main Area ---
st.subheader("⚙️ Analysis Configuration")
st.write(f"**Product Page URL for UX Audit:** {product_page_url_input or '_Not Provided_'}")
st.write(f"**Homepage URL for Info Extraction:** {homepage_url_input or '_Not Provided_'}")
st.write(f"**Manual Facebook URL Provided:** {'Yes' if manual_fb_page_url else 'No'}")
st.markdown("---")

# --- State Variables Initialization ---
if 'analysis_ran' not in st.session_state: st.session_state.analysis_ran = False
if 'master_summary_text' not in st.session_state: st.session_state.master_summary_text = ""
if 'last_taken_screenshots_dict' not in st.session_state: st.session_state.last_taken_screenshots_dict = {"desktop": [], "mobile": []}
if 'final_fb_url_used' not in st.session_state: st.session_state.final_fb_url_used = None
if 'apify_ads_result' not in st.session_state: st.session_state.apify_ads_result = ([], None)
if 'cms_result' not in st.session_state: st.session_state.cms_result = "Not checked"
if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'overall_ad_summary' not in st.session_state: st.session_state.overall_ad_summary = ""
if 'executive_summary_md' not in st.session_state: st.session_state.executive_summary_md = ""
if 'hero_products_info' not in st.session_state: st.session_state.hero_products_info = []
if 'site_categories_info' not in st.session_state: st.session_state.site_categories_info = []
if 'top_recommendations_list' not in st.session_state: st.session_state.top_recommendations_list = []

# --- Analysis Trigger ---
if st.button("🚀 Run Analysis & Checks", key="run_button"):
    st.session_state.analysis_ran = True; st.session_state.master_summary_text = ""
    st.session_state.last_taken_screenshots_dict = {"desktop": [], "mobile": []}; st.session_state.final_fb_url_used = None
    st.session_state.apify_ads_result = ([], None); st.session_state.cms_result = "Not checked"
    st.session_state.log_messages = []; st.session_state.overall_ad_summary = ""
    st.session_state.executive_summary_md = ""; st.session_state.hero_products_info = []
    st.session_state.site_categories_info = []; st.session_state.top_recommendations_list = []

    log_placeholder = st.empty()
    def log_update(message):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        st.session_state.log_messages.append(f"- {timestamp} | {message}")
        log_placeholder.markdown(f"##### 📊 Analysis Execution Log\n```\n" + "\n".join(st.session_state.log_messages) + "\n```")
    st.session_state.log_func = log_update
    log_update("Starting Analysis Process...")

    if not homepage_url_input and not product_page_url_input:
        st.warning("⚠️ Please provide at least a Homepage URL or a Product Page URL.")
        log_update("❌ ERROR: At least one URL (Homepage or Product Page) is required.")
        st.session_state.analysis_ran = False; log_placeholder.markdown(f"##### 📊 Analysis Execution Log\n```\n" + "\n".join(st.session_state.log_messages) + "\n```"); st.stop()

    # --- 0. Homepage Extraction ---
    if homepage_url_input:
        log_update(f"🏠 Extracting products/categories from homepage: {homepage_url_input}...")
        with st.spinner("Analyzing homepage content..."):
            homepage_data = extract_from_homepage(homepage_url_input)
            st.session_state.hero_products_info = homepage_data.get("hero_products", [])
            st.session_state.site_categories_info = homepage_data.get("site_categories", [])
        log_update(f"✅ Homepage: Found {len(st.session_state.hero_products_info)} hero products, {len(st.session_state.site_categories_info)} site categories.")
    else:
        log_update("⚪ Homepage URL not provided, skipping direct homepage extraction.")

    if not st.session_state.hero_products_info and product_page_url_input:
        title_hero = get_page_title(product_page_url_input)
        if title_hero and "N/A" not in title_hero and len(title_hero) > 3:
            st.session_state.hero_products_info = [title_hero.strip()]
            log_update(f"ℹ️ Using product page title for Hero Product: {title_hero.strip()}")
        else:
            parsed_pp_url = urlparse(product_page_url_input)
            path_segment = parsed_pp_url.path.strip('/').split('/')[-1]
            if path_segment:
                 st.session_state.hero_products_info = [path_segment.replace('-', ' ').replace('_', ' ').title()]
                 log_update(f"ℹ️ Using product page URL segment for Hero Product: {st.session_state.hero_products_info[0]}")

    # --- 1. Run CMS Check ---
    url_for_cms = homepage_url_input if homepage_url_input else product_page_url_input
    if url_for_cms:
        log_update(f"🔍 Checking CMS via WhatCMS for {url_for_cms}...")
        with st.spinner("Detecting CMS..."): st.session_state.cms_result = get_cms_with_whatcms(url_for_cms)
        log_update(f"✅ CMS Check Result: {st.session_state.cms_result}")
    else:
        st.session_state.cms_result = "CMS Check Skipped (No URL for CMS check)"
        log_update(f"⚪ {st.session_state.cms_result}")

    # --- 2. Run Playwright Screenshotting ---
    if product_page_url_input:
        log_update(f"📸 Capturing Desktop & Mobile screenshots for Product Page: {product_page_url_input}...")
        with st.spinner("Running Playwright (Browser Automation)... This may take ~1-2 minutes."):
            screenshot_dict = run_playwright_sync(product_page_url_input)
        st.session_state.last_taken_screenshots_dict = screenshot_dict
        num_desktop = len(screenshot_dict.get('desktop',[])); num_mobile = len(screenshot_dict.get('mobile',[]))
        if not num_desktop and not num_mobile: log_update("❌ Failed to capture any screenshots for Product Page.")
        else: log_update(f"✅ Captured {num_desktop} desktop & {num_mobile} mobile screenshots for Product Page.")
    else:
        log_update("⚪ Product Page URL not provided, skipping screenshots and UX analysis.")
        st.session_state.master_summary_text = "Product Page UX Analysis skipped (No Product Page URL provided)."

    # --- 3. Generate Master Summary using Gemini ---
    screenshots_available = bool(st.session_state.last_taken_screenshots_dict.get('desktop') or st.session_state.last_taken_screenshots_dict.get('mobile'))
    if product_page_url_input and screenshots_available and gemini_configured and GEMINI_LIBS_AVAILABLE:
        log_update("📝 Generating Product Page UX Analysis using Gemini Vision...")
        try:
            domain_for_summary = get_domain_from_url(product_page_url_input) or "Analyzed Product Page"
            with st.spinner(f"Calling Gemini ({GEMINI_MULTIMODAL_MODEL_NAME})... Analyzing page screenshots."):
                 st.session_state.master_summary_text = generate_analysis_and_summary(st.session_state.last_taken_screenshots_dict, domain_for_summary)
            if st.session_state.master_summary_text.startswith(("Error:", "**Fallback", "**Warning:")): log_update(f"⚠️ Gemini Page Analysis: {st.session_state.master_summary_text[:100]}...")
            else: log_update("✅ Gemini Page Analysis generated.")
        except Exception as e:
            log_update(f"❌ Error generating Gemini page analysis: {e}"); st.error(f"Failed Gemini page analysis: {e}\n{traceback.format_exc()}")
            st.session_state.master_summary_text = f"Error generating page summary: {e}"
    elif product_page_url_input and not screenshots_available:
        log_update("⚠️ No screenshots available for Gemini page summary (Product Page).")
        st.session_state.master_summary_text = "Summary not generated: No screenshots captured for Product Page."
    elif product_page_url_input :
        log_update("⚠️ Gemini not configured/available, skipping page summary (Product Page).")
        st.session_state.master_summary_text = "Summary not generated: Gemini unavailable or not configured."

    # --- 4. Determine Facebook URL & Run Ad Checks ---
    st.session_state.final_fb_url_used = None
    url_for_social_checks = homepage_url_input if homepage_url_input else product_page_url_input
    if url_for_social_checks and do_check_facebook:
        log_update(f"🔗 Attempting to Auto-Detect Facebook Page link on {url_for_social_checks}...")
        with st.spinner("Scanning for Facebook link..."):
            detected_fb = detect_facebook_page(url_for_social_checks)
        if detected_fb:
            st.session_state.final_fb_url_used = detected_fb
            log_update(f"✅ FB Page Auto-Detected: {detected_fb}")
        else:
            log_update("⚠️ FB Page Auto-Detection failed.")
            if manual_fb_page_url:
                st.session_state.final_fb_url_used = manual_fb_page_url
                log_update(f"ℹ️ Using Manual Facebook URL: {manual_fb_page_url}")
            else:
                 log_update("⚪ No Manual Facebook URL provided.")
    elif manual_fb_page_url:
         st.session_state.final_fb_url_used = manual_fb_page_url
         log_update(f"ℹ️ Using Manual Facebook URL (Auto-Detect disabled): {manual_fb_page_url}")
    else:
         log_update("⚪ Skipping Facebook detection (disabled and no manual URL).")

    if do_check_meta_ads and META_ADS_AUDIT_AVAILABLE and APIFY_CLIENT_AVAILABLE:
        if st.session_state.final_fb_url_used:
            log_update("💰 Scraping Meta Ads Library via Apify...")
            fb_url_to_scrape = st.session_state.final_fb_url_used
            log_update(f"▶️ Starting Apify actor for URL: {fb_url_to_scrape}")
            with st.spinner(f"Running Apify actor for Facebook Ads Library..."):
                try:
                    items, error_msg = meta_ads_audit(fb_url_to_scrape)
                    items = items if isinstance(items, list) else []
                    st.session_state.apify_ads_result = (items, error_msg)
                except NameError: log_update(f"❌ Apify function 'meta_ads_audit' not found."); items, error_msg = [], "Internal app error: Apify func not found."
                except Exception as apify_e: log_update(f"❌ Exception calling meta_ads_audit: {apify_e}\n{traceback.format_exc()}"); items, error_msg = [], f"Internal app error: {apify_e}"
            items_res, error_msg_res = st.session_state.apify_ads_result
            if error_msg_res: log_update(f"❌ Apify scrape error: {error_msg_res}")
            elif items_res: log_update(f"✅ Apify: Found {len(items_res)} ad(s).")
            else: log_update(f"✅ Apify: No ad items found.")
        else:
             log_update("⚠️ Skipping Apify Meta Ads scrape (No Facebook Page URL available).")
             st.session_state.apify_ads_result = ([], "Skipped: No FB URL available.")
    elif do_check_meta_ads:
        log_update("⚪ Skipping Apify Meta Ads scrape (Dependencies missing).")
        st.session_state.apify_ads_result = ([], "Skipped: Dependencies missing.")
    else:
        log_update("⚪ Skipping Apify Meta Ads scrape (disabled by user).")
        st.session_state.apify_ads_result = ([], "Skipped: Disabled by user.")


    # --- 5. Perform Gemini Ad Image Analysis & Overall Ad Summary ---
    analyzed_ads_list = []
    ads_to_analyze, apify_error = st.session_state.apify_ads_result
    if ads_to_analyze and gemini_configured and GEMINI_LIBS_AVAILABLE and PIL_AVAILABLE:
         log_update(f"🧠 Analyzing {len(ads_to_analyze)} ad image(s) with Gemini Vision...")
         analysis_progress = st.progress(0); num_ads = len(ads_to_analyze)
         with st.spinner(f"Analyzing ad images with Gemini (0/{num_ads})..."):
              for i, ad_item in enumerate(ads_to_analyze):
                   st.spinner(f"Analyzing ad images ({i+1}/{num_ads})...")
                   ad_item_copy = ad_item.copy()
                   try:
                       extracted_details = extract_ad_details(ad_item_copy)
                       image_url, headline, creative_text = extracted_details['image_url'], extracted_details['headline'], extracted_details['creative_text']
                       if image_url:
                            analysis_result = analyze_ad_image_with_gemini(image_url, headline, creative_text)
                            ad_item_copy['gemini_ad_analysis_result'] = analysis_result
                       else: ad_item_copy['gemini_ad_analysis_result'] = "Ad image URL not available."
                   except Exception as ad_analysis_err:
                       ad_item_copy['gemini_ad_analysis_result'] = f"Error during prep: {ad_analysis_err}"
                   finally: analyzed_ads_list.append(ad_item_copy); analysis_progress.progress((i + 1) / num_ads)
         st.session_state.apify_ads_result = (analyzed_ads_list, apify_error)
         log_update("✅ Gemini Ad Image Analysis loop finished."); analysis_progress.empty()
    elif ads_to_analyze:
         for ad_item in ads_to_analyze: ad_item['gemini_ad_analysis_result'] = "Analysis skipped (Gemini/Pillow unavailable)."
         st.session_state.apify_ads_result = (ads_to_analyze, apify_error)

    ads_for_summary, _ = st.session_state.apify_ads_result
    if ads_for_summary and gemini_configured and GEMINI_LIBS_AVAILABLE:
         with st.spinner("Generating overall ad summary..."): st.session_state.overall_ad_summary = generate_overall_ad_quality_summary(ads_for_summary)
         log_update("✅ Overall Ad Quality Summary generated." if not st.session_state.overall_ad_summary.startswith(("Error:", "**Fallback")) else f"⚠️ Overall Ad Summary: {st.session_state.overall_ad_summary[:100]}...")
    elif ads_for_summary: st.session_state.overall_ad_summary = "Overall Summary not generated: Gemini unavailable."
    else: st.session_state.overall_ad_summary = "Overall Summary not generated: No ads."

    # --- 6. Generate Executive Summary ---
    log_update("📝 Generating Executive Summary...")
    try:
        prospect_name_val = "Prospect"
        main_url_for_name = homepage_url_input if homepage_url_input else product_page_url_input
        if main_url_for_name:
            domain_name = get_domain_from_url(main_url_for_name)
            if domain_name: prospect_name_val = domain_name.split('.')[0].replace('-', ' ').title()

        report_date_val = datetime.date.today().strftime('%B %d, %Y')
        parsed_ux_gaps = parse_gemini_ux_summary_for_exec_summary(st.session_state.master_summary_text)
        ads_for_exec_summary, _ = st.session_state.apify_ads_result
        url_for_ad_context = product_page_url_input if product_page_url_input else main_url_for_name
        parsed_meta_ads_gaps = parse_meta_ads_data_for_exec_summary(
            ads_for_exec_summary, st.session_state.overall_ad_summary, url_for_ad_context
        )
        ads_library_search_link = construct_meta_ads_library_search_link(st.session_state.final_fb_url_used)

        st.session_state.executive_summary_md = generate_executive_summary_markdown(
            prospect_name=prospect_name_val, website_url=main_url_for_name or "N/A",
            report_date=report_date_val, hero_products_list=st.session_state.hero_products_info,
            detected_site_categories_list=st.session_state.site_categories_info,
            tech_stack_cms_result=st.session_state.cms_result,
            usp="To be defined", audience="To be defined", competitors="To be defined",
            facebook_page_url=st.session_state.final_fb_url_used,
            product_page_url_audited=product_page_url_input or "N/A (Not provided for UX Audit)",
            ux_audit_gaps=parsed_ux_gaps,
            meta_ads_library_search_link=ads_library_search_link,
            meta_ads_audit_gaps=parsed_meta_ads_gaps
        )
        log_update("✅ Executive Summary generated.")
    except Exception as e:
        log_update(f"❌ Error generating Executive Summary: {e}\n{traceback.format_exc()}")
        st.session_state.executive_summary_md = f"### Executive Summary Error\nCould not generate: {e}"

    # --- 7. Extract Top Recommendations ---
    log_update("📝 Extracting Top Recommendations...")
    try:
        st.session_state.top_recommendations_list = extract_top_recommendations(
            st.session_state.master_summary_text, st.session_state.overall_ad_summary
        )
        log_update(f"✅ Extracted {len(st.session_state.top_recommendations_list)} top recommendations.")
    except Exception as e:
        log_update(f"❌ Error extracting top recommendations: {e}\n{traceback.format_exc()}")
        st.session_state.top_recommendations_list = ["Error extracting recommendations."]

    log_update("✅ Analysis & Checks Completed!")
    log_placeholder.markdown(f"##### 📊 Analysis Execution Log (Completed)\n```\n" + "\n".join(st.session_state.log_messages) + "\n```")

# --- Display Results Area ---
if st.session_state.analysis_ran:
    if st.session_state.executive_summary_md:
        st.markdown("---"); st.markdown(st.session_state.executive_summary_md, unsafe_allow_html=True)

    if st.session_state.top_recommendations_list:
        st.markdown("---"); st.subheader("⭐ Top 5 Key Recommendations")
        if st.session_state.top_recommendations_list == ["Error extracting recommendations."]: st.warning(st.session_state.top_recommendations_list[0])
        elif not st.session_state.top_recommendations_list: st.info("No specific recommendations were extracted.")
        else:
            for i, rec in enumerate(st.session_state.top_recommendations_list): st.markdown(f"{i+1}. {rec}")

    st.markdown("---"); st.header("📈 Analysis Results Dashboard")
    col_website_info, col_fb_detection, col_ads_scrape = st.columns(3)
    with col_website_info:
        st.subheader("🌐 Website Info")
        url_disp = homepage_url_input if homepage_url_input else product_page_url_input
        st.write(f"**Primary URL Analyzed:** {url_disp or '_Not Provided_'}")
        if url_disp:
            domain = get_domain_from_url(url_disp)
            st.write(f"**Domain:** {domain or '_Could not parse_'}")

        st.markdown("**Hero Product(s) (Detected):**")
        if st.session_state.hero_products_info:
            md_list = "\n".join([f"- {p}" for p in st.session_state.hero_products_info])
            st.markdown(md_list)
        else:
            st.write("_N/A or not detected_")
        st.caption("_Note: Product/category detection from homepage is heuristic._")


        st.markdown("**Site Categories (Detected):**")
        if st.session_state.site_categories_info:
            md_list_cats = "\n".join([f"- {c}" for c in st.session_state.site_categories_info])
            st.markdown(md_list_cats)
        else:
            st.write("_N/A or not detected_")

        cms_res_display = st.session_state.get('cms_result', '_Not checked_')
        st.write(f"**Detected CMS/Tech Stack:** {cms_res_display}")

    with col_fb_detection:
        st.subheader("Facebook Page Detection")
        fb_url_final = st.session_state.get('final_fb_url_used')
        manual_url_provided = bool(st.session_state.get('fb_manual_url'))
        auto_detect_tried = st.session_state.get('fb_check', True)

        if fb_url_final:
            source = "(Manually Provided)" if fb_url_final == st.session_state.get('fb_manual_url') else "(Auto-Detected)"
            st.success(f"✅ Link Used: [{fb_url_final}]({fb_url_final}) {source}")
        elif auto_detect_tried and manual_url_provided:
             st.warning("⚠️ Auto-detection failed, and manual URL was not valid or used.")
        elif auto_detect_tried:
             st.warning("⚠️ Auto-detection failed, no manual URL provided.")
        elif manual_url_provided:
             st.info(f"ℹ️ Using Manual URL: [{st.session_state.get('fb_manual_url')}]({st.session_state.get('fb_manual_url')})")
        else:
             st.info("ℹ️ FB Page detection disabled or no URL provided.")


    with col_ads_scrape:
        st.subheader("Meta Ads Library Scrape (Apify)")
        ads_items, ads_error = st.session_state.get('apify_ads_result', ([], "Check not run"))
        ads_check_enabled = st.session_state.get('ads_check', False); apify_deps_ok = APIFY_CLIENT_AVAILABLE and META_ADS_AUDIT_AVAILABLE

        if ads_check_enabled and apify_deps_ok:
             if ads_error == "Skipped: No FB URL available.":
                 st.warning(f"⚠️ {ads_error}")
             elif ads_error and not ads_items:
                  if "Skipped:" in ads_error: st.info(f"ℹ️ {ads_error}")
                  else: st.error(f"❌ Scrape Failed/Warning: {ads_error}")
             elif ads_items:
                  ads_to_display = ads_items[:5]; st.success(f"✅ Scrape successful: Found {len(ads_items)} ad(s).")
                  if ads_to_display:
                       st.markdown("**Found Ads (Links):**"); links_found = 0
                       for i, ad_item_disp in enumerate(ads_to_display):
                            ad_archive_id = ad_item_disp.get('adArchiveID')
                            if ad_archive_id: st.markdown(f"- Ad {i+1}: [View in Ad Library](https://www.facebook.com/ads/library/?id={ad_archive_id})"); links_found += 1
                            else: st.markdown(f"- Ad {i+1}: `ID not available`")
                       if links_found == 0: st.write("_No Ad Library links in results._")
                  if ads_error and "Skipped:" not in ads_error: st.warning(f"⚠️ Issues during scrape: {ads_error}")
             elif not ads_error and not ads_items: st.success("✅ Scrape successful: No ads found.")
        elif ads_check_enabled and not apify_deps_ok: st.error("❌ Apify check skipped: missing dependencies.")
        else: st.info("ℹ️ Apify scrape was disabled.")

    st.markdown("---"); st.subheader("Overall Meta Ad Quality Summary")
    overall_summary_text = st.session_state.get('overall_ad_summary', '')
    if not overall_summary_text or overall_summary_text.startswith(("Error:", "**Fallback", "**Warning:", "No valid", "Overall Summary not generated:")):
        st.warning(f"Overall Ad Quality Summary: {overall_summary_text or 'Not Available'}")
    else: st.markdown(overall_summary_text)

    st.markdown("---")
    ads_items_for_display, _ = st.session_state.get('apify_ads_result', ([], None))
    st.header(f"Ads Audit Details (Displaying Max {min(len(ads_items_for_display), 5)} Ads)")
    if ads_items_for_display:
        for i, ad_item_detail in enumerate(ads_items_for_display[:5]):
            st.markdown(f"--- \n ### Ad {i+1}")
            col_ad_creative, col_ad_analysis = st.columns([1, 2])
            with col_ad_creative:
                st.subheader("Ad Display")
                ad_details = extract_ad_details(ad_item_detail)
                image_url, headline, creative_text, cta_text, landing_page = ad_details['image_url'], ad_details['headline'], ad_details['creative_text'], ad_details['cta_text'], ad_details['landing_page']
                if image_url:
                    try: st.image(image_url, caption="Ad Creative Preview", use_column_width=True)
                    except Exception as img_e: st.warning(f"Could not load image: {img_e}")
                else: st.write("_No image/video preview_")
                st.markdown(f"**Headline:** {headline}"); st.markdown(f"**Body Text:**")
                st.text_area(f"BodyText_{i}", creative_text, height=100, disabled=True, key=f"body_text_disp_{i}")
                st.markdown(f"**CTA:** `{cta_text}`")
                is_active = ad_item_detail.get('isActive'); status = "Active" if is_active else "Inactive" if is_active is not None else "N/A"
                platforms = ", ".join(ad_item_detail.get('publisherPlatform', [])) or "N/A"; start_date_str_detail = ad_item_detail.get('startDateFormatted')
                launch_date_display, days_running_display = "N/A", "N/A"; ad_archive_id_detail = ad_item_detail.get('adArchiveID')
                if start_date_str_detail:
                    try:
                        date_part = start_date_str_detail.split('T')[0]; launch_date = datetime.datetime.strptime(date_part, '%Y-%m-%d').date()
                        launch_date_display = launch_date.strftime('%Y-%m-%d'); delta = datetime.date.today() - launch_date
                        days_running_display = f"{delta.days} days" if delta.days >= 0 else "Not Started"
                    except: launch_date_display = f"Invalid ({start_date_str_detail})"; days_running_display = "Cannot Calc."
                st.markdown(f"**Status:** `{status}` | **Platforms:** {platforms}")
                st.markdown(f"**Launched:** {launch_date_display} | **Running:** {days_running_display}")
                if ad_archive_id_detail: st.markdown(f"**Ad Library:** [View Link](https://www.facebook.com/ads/library/?id={ad_archive_id_detail})")
                else: st.markdown("**Ad Library:** `Link unavailable`")
                st.markdown(f"**Landing Page:** [{landing_page}]({landing_page})")
            with col_ad_analysis:
                st.subheader("Gemini Analysis")
                ad_analysis_result = ad_item_detail.get('gemini_ad_analysis_result', 'Analysis not generated.')
                if ad_analysis_result.startswith(("Error:", "**Fallback", "**Warning:", "No image URL", "Analysis skipped", "Analysis not generated")): st.warning(f"Analysis Note: {ad_analysis_result}")
                else: st.markdown(ad_analysis_result)
    else:
        st.info("No ads found or scraped to display details for.")

    if product_page_url_input:
        st.markdown("---"); st.header("Product Page UX Analysis (Gemini Summary)")
        page_summary_text = st.session_state.get('master_summary_text', '')
        if page_summary_text and not page_summary_text.startswith(("Error:", "**Fallback", "**Warning:", "Summary not generated:", "Product Page UX Analysis skipped")):
            st.markdown(page_summary_text)
        elif page_summary_text: st.warning(f"Product Page Analysis Note: {page_summary_text}")
        else: st.info("Product Page analysis was not generated.")

    st.markdown("---"); st.subheader("📄 Export Product Page Summary to Google Doc")
    can_export_gdoc = product_page_url_input and GOOGLE_LIBS_AVAILABLE and st.session_state.master_summary_text and not st.session_state.master_summary_text.startswith(("Error:", "**Fallback", "**Warning:", "Summary not generated:", "Product Page UX Analysis skipped"))
    if can_export_gdoc:
         if st.button("Export Page Summary to Google Doc", key="export_gdoc"):
              with st.spinner("Exporting to Google Docs..."):
                   export_status = export_to_google_doc(st.session_state.master_summary_text, google_drive_folder_id or None)
                   if export_status.startswith("✅"): st.success(export_status)
                   else: st.error(export_status)
    elif not product_page_url_input: st.info("Google Doc export requires a Product Page URL and its UX analysis.")
    elif not GOOGLE_LIBS_AVAILABLE: st.info("Google Doc export disabled (libraries not found).")
    else: st.info("Cannot export: Product page summary not successfully generated or is unavailable.")

    st.markdown("---"); st.subheader("📊 Analysis Execution Log (Recap)")
    if st.session_state.log_messages: st.markdown(f"```\n" + "\n".join(st.session_state.log_messages) + "\n```")
    else: st.info("Log not available.")

# --- Footer ---
st.markdown("---")
st.caption("Gemini Product Page UX Auditor | v1.5.2") # Updated version number