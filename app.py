# --- Core Imports ---
import streamlit as st
import json
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import datetime # Make sure datetime is imported
import traceback
import io
import asyncio
import webbrowser # Import the webbrowser module
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
    # Install Playwright browsers and dependencies if Playwright is available
    import os
    st.info("Attempting to install Playwright browsers and dependencies...")
    os.system('playwright install')
    os.system('playwright install-deps')
    st.info("Playwright installation commands executed.")


# --- Image Handling ---
try:
    from PIL import Image
    PIL_AVAILABLE = True
    # Also import google.ai.generativelanguage Parts for direct use
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
    st.warning("⚠️ Google API libraries not found (`pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib`). Google Doc export disabled.")
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


# --- Configuration (Hardcoded - As Requested by User) ---
# WARNING: Storing secrets directly in code is a security risk.
# Consider using Streamlit Secrets or environment variables.
GOOGLE_SERVICE_ACCOUNT_FILE = "service_account_key.json"
WHATCMS_API_KEY = "w3xz6q7bamb7zixn1skvj2ei8wkz2xafrrjszv5fkk8yscm4019cim6wtgxuk13y20u2wu" # Replace if needed
GEMINI_API_KEY = "AIzaSyDAfqg0tqIkVAE_DV4vd6OOjJz_pXdnHso" # Replace with your actual key

# --- API URLs & Settings ---
WHATCMS_API_URL = "https://whatcms.org/API/Tech"
GOOGLE_DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.file']
GOOGLE_DOCS_SCOPES = ['https://www.googleapis.com/auth/documents']
GOOGLE_ALL_SCOPES = GOOGLE_DRIVE_SCOPES + GOOGLE_DOCS_SCOPES
REQUESTS_TIMEOUT = 15
PLAYWRIGHT_TIMEOUT = 45000  # 45 seconds
SCROLL_DELAY_MS = 2000      # 2 seconds between scrolls
MAX_SCROLLS = 5             # Max scrolls after initial view

# --- Gemini Model Configuration ---
GEMINI_MULTIMODAL_MODEL_NAME = "gemini-1.5-flash-latest" # Or another suitable multimodal model

# --- Single Call Analysis Prompt ---
COMBINED_ANALYSIS_SUMMARY_PROMPT = """
Analyze the following sequence of screenshots for a product page (domain: '{domain}'). Screenshots are provided for both Desktop and Mobile viewports, captured at different scroll positions (labeled D0, D1... M0, M1...).

Based *only* on these images, perform a visual UX critique and generate a structured report following the EXACT format below. Synthesize findings across all images for each point.

**Output Format (Use Markdown):**

Overall Summary:
[Provide a concise 1-2 paragraph summary. Include overall impression (e.g., clean, professional), key strengths (e.g., imagery, clear info), the most critical weakness observed (e.g., sticky CTA issue), and the primary opportunity for improvement based on the visual evidence.]

--- Product Page Findings: ---
[List 5-7 key findings synthesized from analyzing *all* screenshots (desktop and mobile). For each finding, use the precise sub-headings and formatting shown below. Focus on actionable UX/UI issues or strengths related to conversion, usability, layout, and trust, considering the product page context and differences between desktop/mobile.]

**Issue/Opportunity:**
[Describe the specific UX issue or strength observed across the screenshots. Be precise. e.g., "Primary 'Add to Cart' button is not sticky on desktop or mobile."]
**Impact:**
[Explain the likely positive or negative effect on the user experience or conversion rate. e.g., "Users reviewing details must scroll back up, adding friction..."]
**Recommendation:**
[Suggest a specific, actionable improvement or way to maintain the strength. e.g., "Implement a sticky 'Add to Cart' button fixed at the bottom..."]

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
"""

# --- Ad Image Analysis Prompt ---
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
    if not GEMINI_API_KEY: st.error("❌ Gemini API Key missing.")
    else:
        try: genai.configure(api_key=GEMINI_API_KEY); gemini_configured = True
        except Exception as e: st.error(f"❌ Error configuring Gemini API: {e}"); GEMINI_LIBS_AVAILABLE = False
else: pass

# --- Gemini Analysis Function ---
def generate_analysis_and_summary(screenshots_dict, domain):
    """Generates analysis and summary from images in a single API call."""
    if not gemini_configured or not GEMINI_LIBS_AVAILABLE:
        st.warning("⚠️ Gemini analysis/summary skipped (Gemini unavailable)."); return "**Fallback Summary (Gemini Unavailable)**"
    if not PIL_AVAILABLE:
        st.warning("⚠️ Pillow library not installed. Screenshot analysis disabled."); return "**Fallback Summary (Pillow Unavailable)**"

    desktop_images = screenshots_dict.get("desktop", [])
    mobile_images = screenshots_dict.get("mobile", [])

    if not desktop_images and not mobile_images:
        return "No screenshots captured to analyze."

    try:
        model = genai.GenerativeModel(GEMINI_MULTIMODAL_MODEL_NAME)
        request_contents = [COMBINED_ANALYSIS_SUMMARY_PROMPT.format(domain=domain)]

        # Add desktop images
        if desktop_images:
             request_contents.append("\n\n--- Desktop Screenshots ---")
             for i, img_bytes in enumerate(desktop_images):
                 try:
                     Image.open(io.BytesIO(img_bytes)) # Verify image data
                     img_part = glm.Part(inline_data=glm.Blob(mime_type="image/png", data=img_bytes))
                     request_contents.append(f"Desktop Screenshot D{i}:")
                     request_contents.append(img_part)
                 except Exception as img_err:
                     st.warning(f"Could not process desktop screenshot {i}: {img_err}")

        # Add mobile images
        if mobile_images:
            request_contents.append("\n\n--- Mobile Screenshots ---")
            for i, img_bytes in enumerate(mobile_images):
                 try:
                     Image.open(io.BytesIO(img_bytes)) # Verify image data
                     img_part = glm.Part(inline_data=glm.Blob(mime_type="image/png", data=img_bytes))
                     request_contents.append(f"Mobile Screenshot M{i}:")
                     request_contents.append(img_part)
                 except Exception as img_err:
                      st.warning(f"Could not process mobile screenshot {i}: {img_err}")

        # Make the single API call only if we successfully added some images
        if len(request_contents) > 1: # More than just the initial prompt text
            response = model.generate_content(request_contents)
            # Check for safety ratings or blocks if necessary (can add later if issues arise)
            # if response.prompt_feedback.block_reason: st.error(f"Gemini content blocked: {response.prompt_feedback.block_reason}")
            return response.text
        else:
            return "Error: Could not process any valid images to send to Gemini."

    except Exception as e:
        st.error(f"❌ Error calling Gemini Multimodal API ({GEMINI_MULTIMODAL_MODEL_NAME}): {e}")
        st.error(traceback.format_exc())
        return f"❌ Gemini Multimodal API Error: {e}"

# --- Gemini Ad Image Analysis Function ---
def analyze_ad_image_with_gemini(image_url, ad_headline, ad_text):
    """Analyzes a single ad image URL using Gemini Vision."""
    if not gemini_configured or not GEMINI_LIBS_AVAILABLE:
        st.warning("⚠️ Gemini ad image analysis skipped (Gemini unavailable)."); return "**Fallback Analysis (Gemini Unavailable)**"
    if not PIL_AVAILABLE:
        st.warning("⚠️ Pillow library not installed. Ad image analysis disabled."); return "**Fallback Analysis (Pillow Unavailable)**"
    if not image_url:
        return "No image URL provided for analysis."

    log_func = st.session_state.get("log_func", print)
    log_func(f"Analyzing ad image: {image_url}")

    try:
        model = genai.GenerativeModel(GEMINI_MULTIMODAL_MODEL_NAME)

        # Fetch the image from the URL
        try:
            response = requests.get(image_url, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status() # Raise HTTPError for bad responses
            img_bytes = response.content
            Image.open(io.BytesIO(img_bytes)) # Verify image data
            img_part = glm.Part(inline_data=glm.Blob(mime_type=response.headers['Content-Type'], data=img_bytes)) # Use actual mime type
        except requests.exceptions.RequestException as e:
            log_func(f"❌ Error fetching ad image {image_url}: {e}")
            return f"Error fetching image: {e}"
        except Exception as e:
            log_func(f"❌ Error processing fetched ad image {image_url}: {e}")
            return f"Error processing image: {e}"

        # Prepare the prompt with ad context
        prompt_text = AD_IMAGE_ANALYSIS_PROMPT
        if ad_headline and ad_headline != 'N/A':
            prompt_text += f"\n\nAd Headline: {ad_headline}"
        if ad_text and ad_text != 'N/A':
            prompt_text += f"\n\nAd Text: {ad_text}"

        request_contents = [prompt_text, img_part]

        # Make the API call
        response = model.generate_content(request_contents)
        # Check for safety ratings or blocks if necessary
        # if response.prompt_feedback.block_reason: log_func(f"Gemini content blocked for image {image_url}: {response.prompt_feedback.block_reason}")

        return response.text

    except Exception as e:
        log_func(f"❌ Error calling Gemini for ad image analysis ({image_url}): {e}")
        log_func(traceback.format_exc())
        return f"❌ Gemini Ad Image Analysis Error: {e}"

# --- Playwright Screenshot Functions ---
async def take_scrolling_screenshots_for_viewport(page, url, viewport_name):
    """Takes screenshots at different scroll positions for a given viewport."""
    screenshots_bytes = []
    log_func = st.session_state.get("log_func", print) # Use logger from state
    try:
        log_func(f"Navigating to {url} for {viewport_name} view...")
        await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT, wait_until='load') # wait_until='load' is often sufficient
        await page.wait_for_timeout(3000) # Allow some time for dynamic content load (user confirmed OK)

        log_func(f"Taking initial screenshot ({viewport_name} - Scroll 0)...")
        screenshot = await page.screenshot(full_page=False) # Capture viewport only
        screenshots_bytes.append(screenshot)

        last_scroll_y = 0
        current_scroll_y = await page.evaluate("window.scrollY")
        viewport_height = page.viewport_size['height'] if page.viewport_size else 800 # Default height

        for i in range(1, MAX_SCROLLS + 1):
            # Scroll down by approx 90% of the viewport height
            await page.evaluate(f'window.scrollBy(0, {viewport_height * 0.9})')
            await page.wait_for_timeout(SCROLL_DELAY_MS) # Wait for scroll and potential lazy loading

            new_scroll_y = await page.evaluate("window.scrollY")
            scroll_height = await page.evaluate("document.body.scrollHeight")

            # Check if scrolling stopped (or very little change) or if we are near the bottom
            # Add buffer to bottom check
            if new_scroll_y <= last_scroll_y + (viewport_height * 0.1) or (new_scroll_y + viewport_height >= scroll_height - 20) : # Increased buffer slightly
                 log_func(f"Reached bottom/stopped scrolling on {viewport_name} view after scroll {i}.")
                 # Take final screenshot only if scroll position actually changed
                 if new_scroll_y > last_scroll_y:
                      log_func(f"Taking final screenshot ({viewport_name} - Scroll {i})...")
                      screenshot = await page.screenshot(full_page=False)
                      screenshots_bytes.append(screenshot)
                 break # Exit loop

            log_func(f"Taking screenshot ({viewport_name} - Scroll {i})...")
            screenshot = await page.screenshot(full_page=False)
            screenshots_bytes.append(screenshot)
            last_scroll_y = new_scroll_y

        # Log if max scrolls reached without hitting the bottom condition
        if i == MAX_SCROLLS and not (new_scroll_y <= last_scroll_y + (viewport_height * 0.1) or (new_scroll_y + viewport_height >= scroll_height - 20)):
             log_func(f"Reached max scrolls ({MAX_SCROLLS}) for {viewport_name} before reaching bottom.")

    except PlaywrightError as e:
        st.error(f"❌ Playwright Error during screenshot ({viewport_name}): {e}")
        log_func(f"❌ Playwright Error during screenshot ({viewport_name}): {e}")
        raise # Re-raise to be caught by the main capture function
    except Exception as e:
        st.error(f"❌ Unexpected Error during screenshot ({viewport_name}): {e}")
        log_func(f"❌ Unexpected Error during screenshot ({viewport_name}): {e}")
        raise # Re-raise
    return screenshots_bytes

async def capture_desktop_and_mobile_screenshots(url: str):
    """Captures scrolling screenshots for both desktop and mobile viewports."""
    if not PLAYWRIGHT_AVAILABLE:
        st.error("Playwright not available. Cannot capture screenshots.")
        return {}

    results = {"desktop": [], "mobile": []}
    browser = None
    p = None
    log_func = st.session_state.get("log_func", print)

    try:
        log_func("Initializing Playwright...")
        p = await async_playwright().start()
        log_func("Launching browser (Chromium)...")
        # Consider adding headless=True for deployments or non-debugging runs
        # browser = await p.chromium.launch(headless=True)
        browser = await p.chromium.launch()

        log_func("--- Processing Desktop View (1280x800) ---")
        desktop_context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        desktop_page = await desktop_context.new_page()
        results["desktop"] = await take_scrolling_screenshots_for_viewport(desktop_page, url, "Desktop")
        await desktop_context.close()
        log_func("--- Desktop View Processing Complete ---")

        log_func("--- Processing Mobile View (iPhone 13) ---")
        # Use a common mobile device emulation
        mobile_context = await browser.new_context(**p.devices['iPhone 13'])
        mobile_page = await mobile_context.new_page()
        results["mobile"] = await take_scrolling_screenshots_for_viewport(mobile_page, url, "Mobile")
        await mobile_context.close()
        log_func("--- Mobile View Processing Complete ---")

        return results
    except Exception as e:
        st.error(f"❌ Playwright Error in main capture function: {e}")
        st.error(traceback.format_exc())
        log_func(f"❌ Playwright Error in main capture function: {e}")
        return results # Return potentially partial results
    finally:
        # --- CORRECTED BLOCK (Ensure this part matches exactly) ---
        if browser:
            try:
                await browser.close()
                log_func("Browser closed.")
            except Exception as close_err:
                log_func(f"Warning: Error closing browser: {close_err}") # Indented correctly under except
        if p:
            try:
                await p.stop()
                log_func("Playwright instance stopped.")
            except Exception as stop_err:
                log_func(f"Warning: Error stopping Playwright: {stop_err}") # Indented correctly under except
        # --- END CORRECTED BLOCK ---

def run_playwright_sync(url: str):
    """Synchronous wrapper to run the async Playwright capture function."""
    if not PLAYWRIGHT_AVAILABLE: return {}

    screenshots_dict = {}
    # Define the async main function locally to capture the result
    async def main():
        nonlocal screenshots_dict
        screenshots_dict = await capture_desktop_and_mobile_screenshots(url)

    try:
        # Get or create an event loop for the current thread
        # This handles environments like Streamlit that might manage the loop differently
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Loop already running")
        except RuntimeError:
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)

        loop.run_until_complete(main())
        return screenshots_dict

    except Exception as e:
        st.error(f"❌ Error running Playwright sync wrapper: {e}")
        st.error(traceback.format_exc())
        return {}
    # Removed unnecessary finally block for loop closing, as run_until_complete handles it


# --- Helper Functions ---
def get_domain_from_url(url):
    """Extracts the domain name (without www.) from a URL."""
    if not url: return None
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.split(':')[0] # Remove port if present
    except Exception as e:
        st.error(f"⚠️ Error parsing URL '{url}': {e}")
        return None
    # Remove 'www.' if it exists
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

def get_cms_with_whatcms(url):
    """Detects CMS using the WhatCMS API."""
    if not url: return "URL not provided."
    if not WHATCMS_API_KEY:
        st.error("❌ WhatCMS API Key is missing.")
        return "CMS Check Failed (Missing Key)"

    params = {"key": WHATCMS_API_KEY, "url": url}
    try:
        response = requests.get(WHATCMS_API_URL, params=params, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        result_info = data.get('result', {})

        # Check API result code
        if result_info.get('code') == 200: # Success code from WhatCMS
            cms_found = None
            if 'results' in data:
                # Iterate through detected technologies
                for tech in data['results']:
                    categories = tech.get('categories', [])
                    # Check if 'CMS' is in the categories list/tuple
                    if isinstance(categories, (list, tuple)) and "CMS" in categories:
                        cms_found = tech.get('name', 'Unknown CMS')
                        break # Found the first CMS, stop looking
            return cms_found if cms_found else "CMS not detected by WhatCMS."
        else:
            # Handle WhatCMS API specific errors/warnings
            error_code = result_info.get('code', 'N/A')
            error_msg = result_info.get('msg', 'Unknown error')
            if error_code == 120: # Rate limit code
                st.warning(f"⚠️ WhatCMS Rate Limited for {url}. Please wait before trying again.")
                return "CMS Check Failed (Rate Limit)"
            else:
                st.warning(f"⚠️ WhatCMS API Warning for {url}: Code {error_code} - {error_msg}")
                return f"CMS Check Failed (API Code: {error_code})"

    # Handle network/request errors
    except requests.exceptions.Timeout:
        st.warning(f"⏳ Timeout connecting to WhatCMS API for {url}.")
        return "CMS Check Failed (Timeout)"
    except requests.exceptions.RequestException as e:
        st.warning(f"🌐 Error fetching CMS data from WhatCMS API for {url}: {e}")
        return "CMS Check Failed (Request Error)"
    # Handle JSON parsing errors
    except json.JSONDecodeError:
        st.warning(f"📄 Invalid JSON response from WhatCMS API for {url}.")
        return "CMS Check Failed (Invalid Response)"
    # Handle other unexpected errors
    except Exception as e:
        st.warning(f"⚙️ Unexpected error processing WhatCMS response for {url}: {e}")
        return f"CMS Check Failed (Error: {type(e).__name__})"

def export_to_google_doc(summary_text, folder_id=None):
    """Exports the summary text to a new Google Doc."""
    if not GOOGLE_LIBS_AVAILABLE:
        return "❌ Error: Google API libraries not installed."
    if not summary_text:
        return "❌ Error: No summary text provided."

    try:
        # Authenticate using service account
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_SERVICE_ACCOUNT_FILE, scopes=GOOGLE_ALL_SCOPES)
        drive_service = build('drive', 'v3', credentials=creds)
        docs_service = build('docs', 'v1', credentials=creds)

        # Create the document
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        document_title = f"Product Page UX Summary - {timestamp}"
        body = {'title': document_title}
        doc = docs_service.documents().create(body=body).execute()
        document_id = doc.get('documentId')
        doc_link = f"https://docs.google.com/document/d/{document_id}/edit"
        st.info(f"📄 Created Google Doc: {doc_link}")

        # Move to folder if ID provided
        if folder_id:
            try:
                # Get the file's current parents to remove them
                file_metadata = drive_service.files().get(fileId=document_id, fields='parents').execute()
                previous_parents = ",".join(file_metadata.get('parents', []))
                # Move the file to the new folder
                drive_service.files().update(
                    fileId=document_id,
                    addParents=folder_id,
                    removeParents=previous_parents,
                    fields='id, parents' # Specify fields to avoid unnecessary data transfer
                ).execute()
                st.info(f"📂 Moved document to Google Drive folder ID: {folder_id}")
            except HttpError as error:
                # Handle specific Google API errors for moving
                status_code = getattr(error.resp, 'status', None)
                if status_code == 404:
                    st.warning(f"⚠️ Could not move document: Folder ID '{folder_id}' not found or permission denied.")
                elif status_code == 403:
                     st.warning(f"⚠️ Could not move document: Permission denied for Folder ID '{folder_id}'. Ensure service account has access.")
                else:
                    st.warning(f"⚠️ Could not move document to folder '{folder_id}'. Google API Error: {error}")
            except Exception as e:
                st.warning(f"⚙️ Unexpected error moving document: {e}.")

        # Insert the summary text into the document
        # Ensure summary_text is a string before cleaning
        if not isinstance(summary_text, str):
            st.warning("Summary text was not a string, attempting conversion for export.")
            summary_text = str(summary_text)

        # Basic cleaning for Google Docs insertion (remove markdown emphasis)
        cleaned_summary = summary_text.replace("**", "").replace("*", "").replace("### ", "").replace("## ", "").replace("# ", "")
        requests_body = [
            {
                'insertText': {
                    'location': {'index': 1}, # Insert at the beginning of the doc body
                    'text': cleaned_summary
                }
            }
        ]
        docs_service.documents().batchUpdate(
            documentId=document_id, body={'requests': requests_body}
        ).execute()

        return f"✅ Successfully exported summary to Google Doc: [View Document]({doc_link})"

    except FileNotFoundError:
        st.error(f"❌ Error: Google service account key file not found at '{GOOGLE_SERVICE_ACCOUNT_FILE}'.")
        return f"❌ Export Failed: File not found."
    except HttpError as error:
        st.error(f"❌ Google API Error during export: {error}")
        try:
            # Try to parse more detailed error message
            error_content = json.loads(error.content)
            error_details_msg = error_content.get('error', {}).get('message', str(error.content))
        except:
            error_details_msg = str(error.content) # Fallback to raw content
        st.error(f"Details: {error_details_msg}")
        st.error("Check service account permissions/quota and ensure Docs/Drive APIs are enabled.")
        return "❌ Export Failed: Google API error."
    except Exception as e:
        st.error(f"⚙️ Unexpected error during Google Doc export: {e}")
        st.error(traceback.format_exc())
        return f"❌ Export Failed: Unexpected error ({type(e).__name__})"

def detect_facebook_page(url):
    """Attempts to find a Facebook Page link on the given URL."""
    if not url: return None
    try:
        # Use a common user agent
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=REQUESTS_TIMEOUT, headers=headers, allow_redirects=True)
        response.raise_for_status()

        # Use the final URL after redirects as the base for relative links
        base_url = response.url
        soup = BeautifulSoup(response.text, 'html.parser')
        potential_links = []

        for link in soup.find_all('a', href=True):
            href = link['href']
            # Skip empty, anchor, mailto, or tel links
            if not href or href.startswith(('#', 'mailto:', 'tel:')):
                continue

            try:
                # Create absolute URL
                absolute_href = urljoin(base_url, href)
                parsed_abs_url = urlparse(absolute_href)
            except ValueError:
                continue # Skip invalid URLs

            # Check if it's a Facebook domain
            if 'facebook.com' in parsed_abs_url.netloc.lower():
                fb_host = parsed_abs_url.netloc.lower()
                # Allow standard Facebook domains
                if fb_host in ['www.facebook.com', 'facebook.com', 'm.facebook.com', 'web.facebook.com']:
                    path = parsed_abs_url.path.lower().strip('/')

                    # --- Heuristics to filter out non-page links ---
                    # Common paths/prefixes to exclude
                    exclude_paths_start = ('sharer', 'dialog', 'login', 'logout', 'plugins', 'watch', 'video', 'photo', 'story', 'events', 'notes', 'groups', 'marketplace', 'gaming', 'pages/create', 'pages/launchpoint', 'pages/browser', 'help', 'legal', 'policies', 'privacy', 'settings', 'developers', 'apps', 'badges', 'bookmarks', 'business_help', 'campaign', 'careers', 'contact_importer', 'directory', 'find-friends', 'fundraisers', 'games', 'groups_discover', 'imbox', 'instant_games', 'jobs', 'latest', 'livemap', 'lookaside.fbsbx.com', 'maps', 'media', 'memories', 'messages', 'mobile', 'movies', 'notifications', 'offers', 'page_insights', 'pages_manager', 'payments', 'people', 'permalink', 'photos', 'places', 'reactions', 'saved', 'search', 'security', 'share', 'stories', 'support', 'terms', 'weather', 'whitehat', 'profile.php')
                    # Common path segments to exclude (might indicate posts, specific sections)
                    exclude_paths_contain = ('/posts/', '/videos/', '/photos/', '/reviews/', '/about/', '/community/')
                    # Paths unlikely to be the main page identifier
                    exclude_paths_exact = ('home.php', '')

                    path_segments = [seg for seg in path.split('/') if seg] # Get non-empty path parts

                    is_excluded_start = path.startswith(exclude_paths_start)
                    is_excluded_contain = any(ex_path in f"/{path}/" for ex_path in exclude_paths_contain)
                    is_excluded_exact = path in exclude_paths_exact

                    if not is_excluded_start and not is_excluded_contain and not is_excluded_exact:
                         # Potential page link patterns:
                         # 1. facebook.com/PageName (single segment, not all digits)
                         # 2. facebook.com/pages/PageName/PageID (starts with 'pages', has ID)
                         if len(path_segments) == 1 and not path_segments[0].isdigit():
                              potential_links.append(absolute_href)
                         elif len(path_segments) > 1 and path_segments[0] == 'pages':
                              potential_links.append(absolute_href)
                         # Can add more heuristics here if needed

        if potential_links:
            # Prefer shorter, cleaner URLs if multiple valid ones are found
            potential_links.sort(key=len)
            return potential_links[0]
        else:
            return None # No suitable link found

    except requests.exceptions.Timeout:
        st.warning(f"⏳ Timeout detecting FB page for {url}.")
        return None
    except requests.exceptions.RequestException as e:
        st.warning(f"🌐 Error fetching {url} for FB detection: {e}")
        return None
    except Exception as e:
        st.error(f"⚙️ Error parsing {url} for FB detection: {e}")
        st.error(traceback.format_exc())
        return None

# ---> HELPER FUNCTION for Funnel Stage <---
def get_funnel_stage(cta_text):
    """Assigns a rough funnel stage based on CTA text."""
    if not cta_text or not isinstance(cta_text, str):
        return 3 # Default to bottom/unknown if no CTA

    cta_lower = cta_text.lower()
    # Bottom Funnel (Action/Conversion)
    if any(term in cta_lower for term in ["shop", "buy", "order", "book", "get offer", "get deal", "install", "play game"]):
        return 3
    # Middle Funnel (Consideration/Lead Gen)
    elif any(term in cta_lower for term in ["sign up", "subscribe", "download", "get quote", "apply", "contact", "register"]):
        return 2
    # Top Funnel (Awareness/Interest)
    elif any(term in cta_lower for term in ["learn more", "watch more", "see more", "visit", "listen"]):
        return 1
    # Default / Unknown
    else:
        return 3 # Default to bottom/unknown

# ---> HELPER FUNCTION for HTML Export <---
def generate_full_audit_html(ads_list, gemini_summary, product_url, cms):
    """Generates a complete HTML report including page info, summary, and a sorted ad table."""

    # --- Basic Markdown to HTML conversion ---
    def basic_md_to_html(md_text):
        if not isinstance(md_text, str):
             return "<p>Summary not available.</p>"
        # Replace Markdown headings, bold, italics, lists, line breaks
        # Note: This is a very basic conversion and might not handle all markdown perfectly
        html = md_text.replace("**", "</strong>") # Close bold first
        html = html.replace("__", "</strong>")
        html = html.replace("<strong>", "<strong>") # Re-open bold

        html = html.replace("*", "</em>") # Close italic first
        html = html.replace("_", "</em>")
        html = html.replace("<em>", "<em>") # Re-open italic

        # Ensure correct nesting for headings
        html = html.replace("### ", "<h3>").replace("</h3>", "</h3>")
        html = html.replace("## ", "<h2>").replace("</h2>", "</h2>")
        html = html.replace("# ", "<h1>").replace("</h1>", "</h1>")

        # Handle simple lists (assumes '- ' start)
        html_lines = []
        in_list = False
        for line in html.split('\n'):
            stripped_line = line.strip()
            if stripped_line.startswith("- "):
                item_text = stripped_line[2:]
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{item_text}</li>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                # Handle paragraphs - wrap non-list, non-heading lines
                if line.strip() and not line.startswith("<h") and not line.startswith("<ul") and not line.startswith("<li"):
                     html_lines.append(f"<p>{line}</p>")
                elif line.strip(): # Keep existing html tags or lines with content
                     html_lines.append(line)

        if in_list: # Close list if file ends with one
            html_lines.append("</ul>")

        html = "\n".join(html_lines)
        # Handle potential empty paragraphs from double line breaks
        html = html.replace("<p></p>", "")
        return html

    # --- Process and Sort Ads ---
    processed_ads = []
    if ads_list:
        for ad_item in ads_list[:5]: # Process max 5 ads
            # Extract details (same logic as before)
            is_active = ad_item.get('isActive')
            status = "Active" if is_active else "Inactive" if is_active is not None else "N/A"
            creative_text = 'N/A'
            headline = 'N/A'
            image_url = None
            cta_text = 'N/A'
            landing_page = '#'
            ad_archive_id = ad_item.get('adArchiveID')
            platforms = ", ".join(ad_item.get('publisherPlatform', [])) or "N/A"
            start_date_str = ad_item.get('startDateFormatted')
            launch_date_display = "N/A"
            days_running_display = "N/A"

            snapshot = ad_item.get('snapshot')
            if isinstance(snapshot, dict):
                cards = snapshot.get('cards')
                if isinstance(cards, list) and cards:
                    first_card = cards[0]
                    if isinstance(first_card, dict):
                        creative_text = first_card.get('body', 'N/A')
                        headline = first_card.get('title', 'N/A')
                        cta_text = first_card.get('ctaText', 'N/A')
                        landing_page = first_card.get('linkUrl', '#')
                        image_url = first_card.get('videoPreviewImageUrl') or \
                                    first_card.get('originalImageUrl') or \
                                    first_card.get('resizedImageUrl')

            if start_date_str:
                try:
                    date_part = start_date_str.split('T')[0]
                    launch_date = datetime.datetime.strptime(date_part, '%Y-%m-%d').date()
                    launch_date_display = launch_date.strftime('%Y-%m-%d')
                    today = datetime.date.today()
                    delta = today - launch_date
                    days_running_display = f"{delta.days} days" if delta.days >= 0 else "Not Started Yet"
                except Exception:
                    launch_date_display = f"Invalid ({start_date_str})"
                    days_running_display = "Cannot Calc."

            funnel_stage_score = get_funnel_stage(cta_text)

            processed_ads.append({
                "stage_score": funnel_stage_score, # Lower score = Top funnel
                "status": status,
                "platforms": platforms,
                "launch_date": launch_date_display,
                "days_running": days_running_display,
                "ad_archive_id": ad_archive_id,
                "image_url": image_url,
                "headline": headline,
                "creative_text": creative_text,
                "cta_text": cta_text,
                "landing_page": landing_page
            })

        # Sort ads: Top funnel (1) to Bottom funnel (3)
        processed_ads.sort(key=lambda ad: ad['stage_score'])

    # --- Build HTML ---
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Product Page & Ad Audit Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; color: #333; }
        .container { max-width: 1000px; margin: auto; background-color: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; } /* Increased max-width */
        h1, h2, h3 { color: #2c3e50; }
        h1 { text-align: center; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-top: 30px; }
        p { margin: 10px 0; }
        strong { color: #16a085; }
        code { background-color: #ecf0f1; padding: 3px 6px; border-radius: 4px; font-family: Consolas, monospace; color: #34495e; }
        blockquote { border-left: 4px solid #bdc3c7; padding-left: 15px; margin-left: 0; color: #555; font-style: italic; background-color: #f9f9f9; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; table-layout: fixed; }
        th, td { border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; word-wrap: break-word; } /* Added word-wrap */
        th { background-color: #f2f2f2; color: #333; font-weight: bold; }

        /* --- Adjusted Column Widths --- */
        th.preview-col { width: 130px; }
        th.text-col { width: 23%; }
        th.cta-col { width: 8%; }
        th.link-col { width: 7%; }
        th.status-col { width: 7%; }
        th.date-col { width: 10%; }
        th.days-col { width: 9%; }

        td img { max-width: 120px; height: auto; display: block; border: 1px solid #eee; margin: 5px auto; } /* Adjusted max-width */
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .missing { color: #95a5a6; font-style: italic; }
        .summary-section { background-color: #fdfefe; border: 1px solid #eaefda; padding: 15px; border-radius: 5px; margin-top: 20px;}
        .summary-section h2, .summary-section h3 { border: none; margin-top: 0;}
        .page-info { margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px dashed #ccc; }
        .footer { text-align: center; margin-top: 30px; font-size: 0.9em; color: #7f8c8d; }
        ul { padding-left: 20px; margin-top: 0; } /* Adjusted list padding/margin */
        li { margin-bottom: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Product Page & Ad Audit Report</h1>
"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_content += f"<p class='footer'>Report generated on: {timestamp}</p>"

    # --- Product Page Info ---
    html_content += '<div class="page-info">\n<h2>Product Page Information</h2>\n'
    html_content += f"<p><strong>URL:</strong> <a href='{product_url}' target='_blank'>{product_url}</a></p>\n"
    html_content += f"<p><strong>Detected CMS:</strong> {cms or 'Not Checked / Failed'}</p>\n"
    html_content += '</div>\n'

    # --- Gemini Summary ---
    html_content += '<div class="summary-section">\n<h2>Gemini Vision UX Analysis Summary</h2>\n'
    html_content += basic_md_to_html(gemini_summary) # Add converted summary
    html_content += '</div>\n'

    # --- Ad Audit Table ---
    html_content += '<h2>Meta Ad Library Audit (Max 5 Ads, Sorted by Funnel Stage)</h2>\n'
    if not processed_ads:
        html_content += "<p>No ads found or scrape was not run.</p>"
    else:
        html_content += """
        <table>
            <thead>
                <tr>
                    <th class="preview-col">Preview</th>
                    <th class="text-col">Headline</th>
                    <th class="text-col">Body Text</th>
                    <th class="cta-col">CTA</th>
                    <th class="link-col">LP Link</th>
                    <th class="status-col">Status</th>
                    <th class="date-col">Launched</th>
                    <th class="days-col">Running</th>
                    <th class="link-col">Ad Link</th>
                </tr>
            </thead>
            <tbody>
"""
        # Loop through SORTED ads
        for ad in processed_ads:
            html_content += '<tr>\n'

            # Image Preview
            img_tag = f'<img src="{ad["image_url"]}" alt="Ad Preview">' if ad["image_url"] else '<span class="missing">No Preview</span>'
            html_content += f'<td>{img_tag}</td>\n'

            # Headline
            headline_display = ad["headline"] if ad["headline"] and ad["headline"] != 'N/A' else '<span class="missing">N/A</span>'
            html_content += f'<td>{headline_display}</td>\n'

            # Body Text
            text_display = ad["creative_text"] if ad["creative_text"] and ad["creative_text"] != 'N/A' else '<span class="missing">N/A</span>'
            html_content += f'<td>{text_display}</td>\n'

            # CTA
            html_content += f'<td><code>{ad["cta_text"]}</code></td>\n'

            # Landing Page
            lp_display = f'<a href="{ad["landing_page"]}" target="_blank" title="{ad["landing_page"]}">Link</a>' if ad["landing_page"] != '#' else '<span class="missing">N/A</span>'
            html_content += f'<td>{lp_display}</td>\n'

            # Status
            html_content += f'<td><code>{ad["status"]}</code></td>\n'

            # Launched
            html_content += f'<td>{ad["launch_date"]}</td>\n'

            # Days Running
            html_content += f'<td>{ad["days_running"]}</td>\n'

            # Ad Library Link
            if ad["ad_archive_id"]:
                 ad_lib_link = f'https://www.facebook.com/ads/library/?id={ad["ad_archive_id"]}'
                 link_tag = f'<a href="{ad_lib_link}" target="_blank">View</a>'
            else:
                 link_tag = '<span class="missing">N/A</span>'
            html_content += f'<td>{link_tag}</td>\n'

            html_content += '</tr>\n' # End table row

        html_content += """
            </tbody>
        </table>
"""
    html_content += '<p class="footer">End of Report</p>'
    html_content += """
    </div> <!-- Close container -->
</body>
</html>
"""
    return html_content

# --- Main App Layout Starts Here ---
st.title("🤖 Gemini Product Page UX Auditor")
st.markdown("Analyzes product page screenshots (taken automatically via Playwright) with Gemini Vision, detects CMS, optionally scrapes Facebook Ads Library via Apify using the detected page link, and exports summaries.")

# --- Dependency Checks ---
if not GEMINI_LIBS_AVAILABLE: st.error("INSTALLATION REQUIRED: Run `pip install google-generativeai` in your terminal.")
if not PLAYWRIGHT_AVAILABLE: st.error("INSTALLATION REQUIRED: Run `pip install playwright && playwright install` in your terminal.")
if not APIFY_CLIENT_AVAILABLE: st.error("INSTALLATION REQUIRED: Run `pip install apify-client` in your terminal.")
if not META_ADS_AUDIT_AVAILABLE: st.error("MISSING FILE: `meta_ads_audit.py` not found. Apify scraping disabled.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Analysis Inputs")
    product_page_url = st.text_input("Product Page URL (Required):", placeholder="https://example.com/product/xyz", key="pp_url")

    st.header("Export Options")
    google_drive_folder_id = st.text_input("Google Drive Folder ID (Optional):", placeholder="Enter Folder ID", help="If provided, exported Doc is moved here.", key="gdrive_id")

    st.header("Additional Checks")
    do_check_facebook = st.checkbox("Detect Facebook Page Link?", value=True, key="fb_check", help="Attempts to find a Facebook Page link on the Product Page URL.")
    # Disable Apify checkbox if library or file is missing
    apify_disabled = not (APIFY_CLIENT_AVAILABLE and META_ADS_AUDIT_AVAILABLE)
    do_check_meta_ads = st.checkbox(
        "Scrape Meta Ads Library via Apify?",
        value=False,
        key="ads_check",
        help="Uses the detected Facebook Page URL to scrape up to 5 ads via Apify (requires Apify token in meta_ads_audit.py)." if not apify_disabled else "Disabled: Requires 'apify-client' library and 'meta_ads_audit.py' file.",
        disabled=apify_disabled
    )


# --- Main Area ---
# Display Config and Info first
col1, col2 = st.columns(2)
with col1:
    st.subheader("⚙️ Analysis Configuration")
    st.write(f"**Product Page URL:** {product_page_url or '_Not Provided_'}")
    st.write(f"**Screenshots:** Automated via Playwright ({MAX_SCROLLS+1} max views each for Desktop & Mobile)")
    st.write(f"**Check Facebook Page:** {'Yes' if do_check_facebook else 'No'}")
    st.write(f"**Scrape Meta Ads Library (Apify):** {'Yes (Max 5 Ads)' if do_check_meta_ads and not apify_disabled else 'No'}") # Updated label
with col2:
    st.subheader("🌐 Website Info (from Product URL)")
    url_for_info = product_page_url
    if url_for_info:
        domain = get_domain_from_url(url_for_info)
        st.write(f"**Domain:** {domain or '_Could not parse_'}")
        # Display CMS result if already computed, otherwise show placeholder
        cms_display = st.session_state.get('cms_result', '_Check will run during analysis_')
        st.write(f"**Detected CMS:** {cms_display}")
    else:
        st.write("_Enter Product Page URL to get website info._")
        # Clear previous CMS result if URL is removed
        if 'cms_result' in st.session_state: del st.session_state.cms_result

st.markdown("---") # Divider before button

# --- State Variables Initialization ---
# Use st.session_state consistently
if 'analysis_ran' not in st.session_state: st.session_state.analysis_ran = False
if 'master_summary_text' not in st.session_state: st.session_state.master_summary_text = ""
if 'last_taken_screenshots_dict' not in st.session_state: st.session_state.last_taken_screenshots_dict = {"desktop": [], "mobile": []}
if 'detected_fb_url' not in st.session_state: st.session_state.detected_fb_url = None
if 'apify_ads_result' not in st.session_state: st.session_state.apify_ads_result = ([], None) # (list_of_items, error_message)
if 'cms_result' not in st.session_state: st.session_state.cms_result = None
if 'log_messages' not in st.session_state: st.session_state.log_messages = []

# --- Analysis Trigger ---
if st.button("🚀 Run Analysis & Checks", key="run_button"):
    # --- Reset State Variables for New Run ---
    st.session_state.analysis_ran = True
    st.session_state.master_summary_text = ""
    st.session_state.last_taken_screenshots_dict = {"desktop": [], "mobile": []}
    st.session_state.detected_fb_url = None
    st.session_state.apify_ads_result = ([], None) # Reset Apify results
    st.session_state.cms_result = None
    st.session_state.log_messages = [] # Clear previous logs

    # --- Input Validation ---
    if not product_page_url:
        st.warning("⚠️ Please provide the Product Page URL.")
        st.session_state.analysis_ran = False # Prevent proceeding
        st.stop() # Halt execution for this run
    if not PLAYWRIGHT_AVAILABLE:
        st.error("❌ Playwright not installed/available. Cannot run analysis.")
        st.session_state.analysis_ran = False
        st.stop()
    # Check Apify dependencies again before running if the checkbox is ticked
    if do_check_meta_ads and apify_disabled:
         st.error("❌ Cannot run Apify check: Library or `meta_ads_audit.py` missing.")
         # Don't proceed with the Apify part of the check
         do_check_meta_ads = False


    # --- EXECUTION LOG SETUP ---
    log_placeholder = st.empty()
    def log_update(message):
        """Appends a message to the execution log in session state and updates the placeholder."""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        st.session_state.log_messages.append(f"- {timestamp} | {message}")
        # Display log using markdown code block for better readability
        log_placeholder.markdown(f"##### 📊 Analysis Execution Log\n```\n" + "\n".join(st.session_state.log_messages) + "\n```")
    st.session_state.log_func = log_update # Make logger available globally via state

    log_update("Starting Analysis Process...")

    # --- 1. Run CMS Check ---
    if product_page_url:
        log_update("🔍 Checking CMS via WhatCMS...")
        with st.spinner("Detecting CMS..."):
            st.session_state.cms_result = get_cms_with_whatcms(product_page_url)
        log_update(f"✅ CMS Check Result: {st.session_state.cms_result}")

    # --- 2. Run Playwright Screenshotting ---
    log_update(f"📸 Capturing Desktop & Mobile screenshots for {product_page_url}...")
    screenshot_dict = {}
    with st.spinner("Running Playwright (Browser Automation)... This may take a minute or two."):
        screenshot_dict = run_playwright_sync(product_page_url)
    st.session_state.last_taken_screenshots_dict = screenshot_dict
    num_desktop = len(screenshot_dict.get('desktop',[]))
    num_mobile = len(screenshot_dict.get('mobile',[]))
    if not num_desktop and not num_mobile:
        log_update("❌ Failed to capture any screenshots.")
        st.error("Failed to capture screenshots. Cannot proceed with Gemini analysis.")
        # Don't stop execution, maybe other checks can run, but summary will be skipped.
    else:
        log_update(f"✅ Captured {num_desktop} desktop & {num_mobile} mobile screenshots.")

    # --- 3. Generate Master Summary using Gemini ---
    screenshots_available = bool(num_desktop or num_mobile)
    if screenshots_available and gemini_configured and GEMINI_LIBS_AVAILABLE:
        log_update("📝 Generating Analysis & Summary using Gemini Vision...")
        try:
            domain_for_summary = get_domain_from_url(product_page_url) or "Analyzed Product Page"
            with st.spinner(f"Calling Gemini ({GEMINI_MULTIMODAL_MODEL_NAME})... Analyzing images."):
                 st.session_state.master_summary_text = generate_analysis_and_summary(
                     st.session_state.last_taken_screenshots_dict, domain_for_summary
                 )
            log_update("✅ Gemini Analysis & Summary generated.")
        except Exception as e:
            log_update(f"❌ Error generating Gemini analysis/summary: {e}")
            st.error(f"Failed to generate Gemini analysis/summary: {e}\n{traceback.format_exc()}")
            st.session_state.master_summary_text = f"Error generating summary: {e}"
    elif not screenshots_available:
        log_update("⚠️ No screenshots available to generate Gemini summary.")
        st.session_state.master_summary_text = "Summary not generated: No screenshots were captured."
    else: # Gemini not available/configured
        log_update("⚠️ Gemini not configured/available, skipping summary generation.")
        st.session_state.master_summary_text = "Summary not generated: Gemini unavailable or not configured."

    # --- 4. Perform Additional Checks ---
    log_update("⚙️ Performing Additional Checks...")
    url_for_checks = product_page_url
    if url_for_checks:
        # --- Facebook Page Detection ---
        if do_check_facebook:
            log_update("🔗 Detecting Facebook Page link...")
            with st.spinner("Scanning Product Page HTML for Facebook link..."):
                detected_fb = detect_facebook_page(url_for_checks)
            st.session_state.detected_fb_url = detected_fb # Store result in state
            status_msg = f"✅ Facebook Page link detection complete. Result: {detected_fb}" if detected_fb else "⚠️ Facebook Page link detection complete. No valid link found."
            log_update(status_msg)
        else:
            log_update("⚪ Skipping Facebook Page detection (disabled by user).")
            st.session_state.detected_fb_url = None # Ensure it's None if skipped

        # --- Apify Meta Ads Library Scraping ---
        # Only proceed if the check is enabled AND dependencies are met
        if do_check_meta_ads and not apify_disabled: # Re-check apify_disabled here
            log_update("💰 Scraping Meta Ads Library via Apify (Max 5 Ads)...")
            if not st.session_state.detected_fb_url and do_check_facebook:
                # FB check was enabled but failed
                log_update("⚠️ Skipping Apify scrape (Required Facebook Page URL was not detected).")
                st.session_state.apify_ads_result = ([], "Skipped: No Facebook Page URL detected.")
            elif not do_check_facebook:
                 # FB check was disabled entirely
                 log_update("⚠️ Skipping Apify scrape (Facebook Page detection was disabled).")
                 st.session_state.apify_ads_result = ([], "Skipped: Facebook detection disabled.")
            elif st.session_state.detected_fb_url:
                 # Proceed with Apify call
                 fb_url_to_scrape = st.session_state.detected_fb_url
                 log_update(f"▶️ Starting Apify actor for URL: {fb_url_to_scrape}")
                 with st.spinner(f"Running Apify actor for Facebook Ads Library... (this can take some time)"):
                     try:
                         # Call the Apify function from meta_ads_audit.py
                         items, error_msg = meta_ads_audit(fb_url_to_scrape)
                         st.session_state.apify_ads_result = (items, error_msg) # Store results
                     except NameError: # meta_ads_audit function itself wasn't imported
                          log_update(f"❌ Apify function 'meta_ads_audit' not found. Check import.")
                          st.error("Error: The 'meta_ads_audit' function is not available. Make sure 'meta_ads_audit.py' exists and is imported correctly.")
                          st.session_state.apify_ads_result = ([], "Internal app error: Apify function not found.")
                          items, error_msg = [], "Internal app error: Apify function not found." # Define for logging
                     except Exception as apify_e: # Catch other errors during the call
                         log_update(f"❌ Exception calling meta_ads_audit: {apify_e}")
                         st.error(f"An unexpected error occurred during the Apify call: {apify_e}")
                         st.session_state.apify_ads_result = ([], f"Internal app error calling Apify: {apify_e}")
                         items, error_msg = [], f"Internal app error calling Apify: {apify_e}" # Define for logging

                 # Log Apify results after spinner finishes
                 # Use the results stored in state for consistency
                 items_res, error_msg_res = st.session_state.apify_ads_result
                 if error_msg_res:
                     log_update(f"❌ Apify scrape finished with error: {error_msg_res}")
                 elif items_res:
                     log_update(f"✅ Apify scrape successful: Found {len(items_res)} ad items.")
                 else: # No error, but no items
                     log_update(f"✅ Apify scrape successful: No ad items found in the library.")
            else: # Should not be reached if logic is correct
                log_update("⚪ Skipping Apify Meta Ads Library scrape (condition not met).")
                st.session_state.apify_ads_result = ([], "Skipped: Condition not met.")

        elif do_check_meta_ads and apify_disabled:
             log_update("⚪ Skipping Apify Meta Ads Library scrape (dependencies missing).")
             st.session_state.apify_ads_result = ([], "Skipped: Dependencies missing.")

        else: # Apify check disabled by user
            log_update("⚪ Skipping Apify Meta Ads Library scrape (disabled by user).")
            st.session_state.apify_ads_result = ([], "Skipped: Apify check disabled.")

    log_update("✅ Analysis & Checks Completed!")
    # Keep log visible after run completes
    log_placeholder.markdown(f"##### 📊 Analysis Execution Log (Completed)\n```\n" + "\n".join(st.session_state.log_messages) + "\n```")


# --- Display Results Area ---
# This section only shows if the 'Run Analysis' button was clicked and analysis_ran is True
if st.session_state.analysis_ran:
    st.markdown("---") # Divider before results dashboard
    st.header("📈 Analysis Results Dashboard")

    # --- Column 1: Website Info ---
    col_info_res, col_checks_res = st.columns([1, 2]) # Give more space to checks/ads

    with col_info_res:
        st.subheader("🌐 Website Info")
        url_for_info_res = product_page_url # Use the URL from the input
        if url_for_info_res:
             domain = get_domain_from_url(url_for_info_res)
             st.write(f"**Product Page URL:**")
             st.markdown(f"[{url_for_info_res}]({url_for_info_res})") # Make URL clickable
             st.write(f"**Domain:** {domain or '_Could not parse_'}")
             cms_res_display = st.session_state.get('cms_result', '_Not checked_')
             st.write(f"**Detected CMS:** {cms_res_display}")
        else:
             st.write("_No URL provided during analysis run._")

    # --- Column 2: Additional Check Results (FB Link, Apify Ads) ---
    with col_checks_res:
        st.subheader("🔗 Additional Check Results")
        if url_for_info_res: # Only show if a URL was processed
            # Facebook Page Detection Result
            st.markdown("**Facebook Page Detection:**")
            # Check if the user enabled this check during the run by checking sidebar state
            if st.session_state.get('fb_check', True):
                fb_link = st.session_state.get('detected_fb_url')
                if fb_link is not None: # Check if detection ran (even if it found nothing)
                    if fb_link:
                        st.success(f"✅ Link Found: [{fb_link}]({fb_link})")
                    else:
                        st.warning("⚠️ No valid Facebook Page link detected on the page.")
                # else: This state shouldn't be reached if analysis_ran is True
            else:
                st.info("ℹ️ Detection was disabled by user.")
            st.markdown("---") # Divider

            # Apify Meta Ads Library Scrape Result
            st.markdown("**Meta Ads Library Scrape (Apify - Max 5 Ads):**")
            # Retrieve results from state
            ads_items, ads_error = st.session_state.get('apify_ads_result', ([], "Check not run"))
            # Check conditions for displaying results
            if st.session_state.get('ads_check', False) and not apify_disabled:
                 # Case 1: Error occurred
                 if ads_error and not ads_items:
                      if "Skipped:" in ads_error: st.info(f"ℹ️ {ads_error}")
                      elif "Dependencies missing" in ads_error: st.error(f"❌ {ads_error}")
                      else: st.error(f"❌ Scrape Failed/Warning: {ads_error}")

                 # Case 2: Items found
                 elif ads_items:
                      # Limit display to the items actually returned (max 5 due to Apify input)
                      ads_to_display = ads_items[:5] # Ensure we don't try to display more than 5
                      st.success(f"✅ Scrape successful: Found {len(ads_items)} ad(s). Displaying details:")

                      # --- ADD DOWNLOAD BUTTON FOR HTML EXPORT ---
                      if ads_to_display: # Only show button if there are ads
                            # Call the NEW function with ALL data needed for the report
                            html_report = generate_full_audit_html(
                                ads_list=ads_to_display, # Use the list of ads found
                                gemini_summary=st.session_state.master_summary_text,
                                product_url=product_page_url,
                                cms=st.session_state.cms_result
                            )
                            report_domain = get_domain_from_url(product_page_url) or "audit"
                            report_filename = f"{report_domain}_ad_audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"

                            # --- Download Button for HTML Export ---
                            st.download_button(
                                label="⬇️ Download Full Audit Report (HTML)",
                                data=html_report,
                                file_name=report_filename,
                                mime="text/html"
                            )
                            # --- End Download Button ---


                      # **** Ad display loop using VERTICAL layout within st.container ****
                      for i, ad_item in enumerate(ads_to_display):
                          # --- Use st.container to group each ad ---
                          with st.container():
                              st.markdown(f"--- \n **Ad {i+1} / {len(ads_to_display)}**")

                              # --- Extract Ad Details ---
                              # Using keys identified from the provided DEBUG output
                              is_active = ad_item.get('isActive') # Get the boolean status
                              status = "Active" if is_active else "Inactive" if is_active is not None else "N/A" # Convert to text

                              # Initialize variables for nested data
                              creative_text = 'N/A'
                              headline = 'N/A'
                              image_url = None
                              cta_text = 'N/A'
                              landing_page = '#' # Default link

                              # Safely access nested 'snapshot' and 'cards'
                              snapshot = ad_item.get('snapshot')
                              if isinstance(snapshot, dict):
                                  cards = snapshot.get('cards')
                                  if isinstance(cards, list) and cards: # Check if cards list exists and is not empty
                                      first_card = cards[0]
                                      if isinstance(first_card, dict): # Check if first card is a dictionary
                                          # Extract pertinent info from the first card
                                          creative_text = first_card.get('body', 'N/A')
                                          headline = first_card.get('title', 'N/A')
                                          cta_text = first_card.get('ctaText', 'N/A') # Or first_card.get('ctaType')
                                          landing_page = first_card.get('linkUrl', '#')
                                          # Image URL fallback logic
                                          image_url = first_card.get('videoPreviewImageUrl') or \
                                                      first_card.get('originalImageUrl') or \
                                                      first_card.get('resizedImageUrl')

                              # Extract Publisher Platforms
                              platforms = ad_item.get('publisherPlatform', [])
                              platforms_str = ", ".join(platforms) if platforms else "N/A"

                              # ---> Extract Ad Archive ID <---
                              ad_archive_id = ad_item.get('adArchiveID') # Or 'adArchiveId' if that was the key
                              # --------------------------------------

                              # ---> Extract and Calculate Launch Date / Days Running <---
                              launch_date_display = "N/A"
                              days_running_display = "N/A"
                              start_date_str = ad_item.get('startDateFormatted')

                              if start_date_str:
                                  try:
                                      # Parse the date part (ignore time and timezone for simplicity)
                                      date_part = start_date_str.split('T')[0]
                                      launch_date = datetime.datetime.strptime(date_part, '%Y-%m-%d').date()
                                      launch_date_display = launch_date.strftime('%Y-%m-%d') # Format for display

                                      today = datetime.date.today()
                                      delta = today - launch_date
                                      if delta.days < 0:
                                            days_running_display = "Not Started Yet"
                                      else:
                                            days_running_display = f"{delta.days} days"
                                  except (ValueError, TypeError, IndexError):
                                      launch_date_display = f"Invalid Format ({start_date_str})" # Show original if invalid
                                      days_running_display = "Cannot Calculate"
                                  except Exception as date_e: # Catch unexpected date errors
                                      launch_date_display = f"Error ({type(date_e).__name__})"
                                      days_running_display = "Error"
                              # -------------------------------------------------------------


                              # --- Display Vertically (No Inner Columns) ---
                              st.write(f"**Status:** `{status}`")
                              st.write(f"**Platforms:** {platforms_str}")
                              st.write(f"**Launch Date:** {launch_date_display}")
                              st.write(f"**Days Running:** {days_running_display}")

                              # Display Ad Library Link
                              if ad_archive_id:
                                  ad_library_url = f"https://www.facebook.com/ads/library/?id={ad_archive_id}"
                                  st.markdown(f"**View in Ad Library:** [{ad_library_url}]({ad_library_url})")
                              else:
                                  st.write("**View in Ad Library:** `Link unavailable (ID not found)`")

                              if image_url:
                                  try:
                                      st.image(image_url, caption="Ad Creative Preview", width=350) # Updated caption
                                      # --- Call Gemini for Ad Image Analysis ---
                                      ad_image_analysis_result = analyze_ad_image_with_gemini(image_url, headline, creative_text)
                                      st.markdown("---") # Small separator
                                      st.markdown("**Gemini Ad Image Analysis:**")
                                      st.markdown(ad_image_analysis_result)
                                      st.markdown("---") # Small separator
                                      # --- End Gemini Call ---
                                  except Exception as img_e:
                                      st.warning(f"Could not load image preview: {img_e}")
                              else:
                                  st.write("_No image/video preview found_")

                              st.write(f"**Headline:**")
                              st.markdown(f"##### {headline if headline and headline != 'N/A' else '_Not available_'}") # Display headline more prominently

                              st.write("**Ad Text:**")
                              text_display = creative_text if creative_text and creative_text != 'N/A' else '_Not available_'
                              st.markdown(f"> {text_display}") # Use blockquote for text

                              st.write(f"**Call To Action:** `{cta_text}`")
                              st.write(f"**Landing Page:** [{landing_page}]({landing_page})")


                              # --- End Vertical Display ---
                          # --- End Ad Display Container ---
                      # --- End Loop for Ads ---

                      # No longer needed as we limit results in Apify call
                      # if len(ads_items) > 5: # This condition is unlikely if resultsLimit=5 worked
                      #     st.info(f"ℹ️ Note: Only the first {len(ads_to_display)} ads are displayed.")
                      if ads_error:
                           st.warning(f"⚠️ Issues reported during scrape (but ads were found): {ads_error}")

                 # Case 3: No items, no error
                 elif not ads_error and not ads_items:
                      st.success("✅ Scrape successful: No ads found in the library for this page.")

                 # Case 4: Check not run yet
                 elif ads_error == "Check not run":
                       st.info("ℹ️ Apify check did not run.")

            elif st.session_state.get('ads_check', False) and apify_disabled:
                 st.error("❌ Apify check skipped due to missing dependencies.")

            else: # Check was disabled by user
                st.info("ℹ️ Apify scrape was disabled by user.")
        else:
            st.info("Additional checks require a Product Page URL to be analyzed first.")

    # --- Display Master Summary & Export Button (Outside main columns) ---
    st.markdown("---") # Divider
    st.subheader("✨ Gemini Vision Analysis Summary ✨")
    if not st.session_state.master_summary_text or "Error generating summary:" in st.session_state.master_summary_text or "Fallback Summary" in st.session_state.master_summary_text:
        st.warning(f"Summary could not be generated or is incomplete. Status: {st.session_state.master_summary_text or 'Not Available'}")
    else:
        st.markdown(st.session_state.master_summary_text) # Display the generated summary/report
        # Add export button only if summary exists and Google libs are available
        if GOOGLE_LIBS_AVAILABLE:
            if st.button("📄 Export Gemini Summary to Google Doc", key="export_gs_button"): # Changed key slightly
                with st.spinner("📤 Exporting to Google Docs..."):
                    export_result = export_to_google_doc(st.session_state.master_summary_text, google_drive_folder_id)
                # Display result message based on success/failure prefix
                if export_result.startswith("✅"):
                    st.success(export_result)
                else:
                    st.error(export_result)
        else:
            st.warning("⚠️ Google Doc export unavailable (libraries missing).")

    # --- Display Execution Log Recap ---
    st.markdown("---")
    st.subheader("📊 Analysis Execution Log (Recap)")
    if st.session_state.log_messages:
        # Use markdown code block for log display
        log_text = "\n".join(st.session_state.log_messages)
        st.markdown(f"```\n{log_text}\n```")
    else:
        st.info("Log not available (analysis might not have completed or was cleared).")


# --- Footer ---
st.markdown("---")
st.caption("Gemini Product Page UX Auditor | Requires: streamlit, requests, beautifulsoup4, google-api-python-client, google-auth-httplib2, google-auth-oauthlib, google-generativeai, Pillow, playwright, apify-client. Ensure `service_account_key.json` and `meta_ads_audit.py` exist and API keys are set. Run `playwright install` and `pip install -r requirements.txt` (if applicable).")
