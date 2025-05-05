# --- Core Imports ---
import streamlit as st
import json
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import datetime
import traceback
import io
import asyncio

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
    st.warning("⚠️ Google API libraries not found (`pip install ...`). Google Doc export disabled.")
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

# --- Configuration (Hardcoded - As Requested) ---
GOOGLE_SERVICE_ACCOUNT_FILE = "service_account_key.json"
WHATCMS_API_KEY = "w3xz6q7bamb7zixn1skvj2ei8wkz2xafrrjszv5fkk8yscm4019cim6wtgxuk13y20u2wu"
META_ACCESS_TOKEN_HARDCODED = "EAAOtZCivQIXkBO4U27ViuGlGfO1ZA1s3ZAwozC0RzOK4gpFdcnnlV49Iwt0DAq1ZBN3OOV54J6vjZBXG9deAxg6PjZCIZB6kK4wLZBmyjx7pgsJaJqMo0evnKsW4VTVUX7j9oODeAn2M7qDDJowGgUXMZCG1EHEc4uExTeakcCQLl0zZAh0TViiypJ"
GEMINI_API_KEY = "AIzaSyDAfqg0tqIkVAE_DV4vd6OOjJz_pXdnHso"

# --- API URLs & Settings ---
WHATCMS_API_URL = "https://whatcms.org/API/Tech"; META_GRAPH_API_VERSION = 'v19.0'; META_API_URL = f'https://graph.facebook.com/{META_GRAPH_API_VERSION}'; GOOGLE_DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.file']; GOOGLE_DOCS_SCOPES = ['https://www.googleapis.com/auth/documents']; GOOGLE_ALL_SCOPES = GOOGLE_DRIVE_SCOPES + GOOGLE_DOCS_SCOPES; REQUESTS_TIMEOUT = 15; PLAYWRIGHT_TIMEOUT = 45000; SCROLL_DELAY_MS = 2000; MAX_SCROLLS = 5

# --- Gemini Model Configuration ---
GEMINI_MULTIMODAL_MODEL_NAME = "gemini-1.5-flash-latest" # Use a model supporting multimodal input

# --- Single Call Analysis Prompt --- ## <<<< MODIFIED PROMPT HERE >>>> ##
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
        st.warning("⚠️ Gemini analysis/summary skipped."); return "**Fallback Summary (Gemini Unavailable)**"
    if not PIL_AVAILABLE:
        st.warning("⚠️ Pillow library not installed. Analysis skipped."); return "**Fallback Summary (Pillow Unavailable)**"

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
                 # Ensure PIL can read the bytes before creating the part
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
            return response.text
        else:
            return "Error: Could not process any valid images to send to Gemini."

    except Exception as e:
        st.error(f"❌ Error calling Gemini Multimodal API ({GEMINI_MULTIMODAL_MODEL_NAME}): {e}")
        st.error(traceback.format_exc())
        return f"❌ Gemini Multimodal API Error: {e}"

# --- Playwright Screenshot Functions ---
async def take_scrolling_screenshots_for_viewport(page, url, viewport_name):
    screenshots_bytes = []
    log_func = st.session_state.get("log_func", print)
    try:
        log_func(f"Navigating to {url} for {viewport_name} view...")
        await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT, wait_until='load')
        await page.wait_for_timeout(3000)
        log_func(f"Taking initial screenshot ({viewport_name} - Scroll 0)..."); screenshot = await page.screenshot(full_page=False); screenshots_bytes.append(screenshot)
        last_scroll_y = 0; current_scroll_y = await page.evaluate("window.scrollY"); viewport_height = page.viewport_size['height'] if page.viewport_size else 800
        for i in range(1, MAX_SCROLLS + 1):
            await page.evaluate(f'window.scrollBy(0, {viewport_height * 0.9})'); await page.wait_for_timeout(SCROLL_DELAY_MS)
            new_scroll_y = await page.evaluate("window.scrollY"); scroll_height = await page.evaluate("document.body.scrollHeight")
            # Add buffer to bottom check
            if new_scroll_y <= last_scroll_y + (viewport_height * 0.1) or (new_scroll_y + viewport_height >= scroll_height - 20) : # Increased buffer slightly
                 log_func(f"Reached bottom/stopped scrolling on {viewport_name} view after scroll {i}.")
                 # Take final screenshot only if scroll position actually changed
                 if new_scroll_y > last_scroll_y:
                      log_func(f"Taking final screenshot ({viewport_name} - Scroll {i})..."); screenshot = await page.screenshot(full_page=False); screenshots_bytes.append(screenshot)
                 break
            log_func(f"Taking screenshot ({viewport_name} - Scroll {i})..."); screenshot = await page.screenshot(full_page=False); screenshots_bytes.append(screenshot); last_scroll_y = new_scroll_y
        if i == MAX_SCROLLS and not (new_scroll_y <= last_scroll_y + (viewport_height * 0.1) or (new_scroll_y + viewport_height >= scroll_height - 20)):
             log_func(f"Reached max scrolls ({MAX_SCROLLS}) for {viewport_name} before reaching bottom.")
    except PlaywrightError as e: st.error(f"❌ Playwright Error ({viewport_name}): {e}"); raise
    except Exception as e: st.error(f"❌ Unexpected Error ({viewport_name}): {e}"); raise
    return screenshots_bytes
async def capture_desktop_and_mobile_screenshots(url: str):
    if not PLAYWRIGHT_AVAILABLE: st.error("Playwright not available."); return {}
    results = {"desktop": [], "mobile": []}; browser = None; p = None; log_func = st.session_state.get("log_func", print)
    try:
        p = await async_playwright().start(); log_func("Launching browser..."); browser = await p.chromium.launch(); log_func("--- Processing Desktop View ---")
        desktop_context = await browser.new_context(viewport={'width': 1280, 'height': 800}); desktop_page = await desktop_context.new_page()
        results["desktop"] = await take_scrolling_screenshots_for_viewport(desktop_page, url, "Desktop"); await desktop_context.close(); log_func("--- Desktop View Processing Complete ---")
        log_func("--- Processing Mobile View ---"); mobile_context = await browser.new_context(**p.devices['iPhone 13']); mobile_page = await mobile_context.new_page()
        results["mobile"] = await take_scrolling_screenshots_for_viewport(mobile_page, url, "Mobile"); await mobile_context.close(); log_func("--- Mobile View Processing Complete ---")
        return results
    except Exception as e: st.error(f"❌ Playwright Error in main capture: {e}"); st.error(traceback.format_exc()); return results
    finally:
        if browser: try: await browser.close(); log_func("Browser closed.") except Exception as close_err: log_func(f"Warning: Error closing browser: {close_err}")
        if p: try: await p.stop(); log_func("Playwright instance stopped.") except Exception as stop_err: log_func(f"Warning: Error stopping Playwright: {stop_err}")
def run_playwright_sync(url: str):
    if not PLAYWRIGHT_AVAILABLE: return {}
    screenshots_dict = {}
    async def main(): nonlocal screenshots_dict; screenshots_dict = await capture_desktop_and_mobile_screenshots(url)
    try: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); loop.run_until_complete(main()); return screenshots_dict
    except Exception as e: st.error(f"❌ Error running Playwright sync wrapper: {e}"); st.error(traceback.format_exc()); return {}

# --- Helper Functions ---
# (get_domain_from_url, get_cms_with_whatcms, export_to_google_doc, detect_facebook_page, check_active_ads omitted for brevity)
def get_domain_from_url(url):
    if not url: return None
    try: parsed_url = urlparse(url); domain = parsed_url.netloc.split(':')[0];
    except Exception as e: st.error(f"⚠️ Error parsing URL '{url}': {e}"); return None
    if domain.startswith('www.'): domain = domain[4:]
    return domain
def get_cms_with_whatcms(url):
    if not url: return "URL not provided."
    if not WHATCMS_API_KEY: st.error("❌ WhatCMS API Key is missing."); return "CMS Check Failed (Missing Key)"
    params = {"key": WHATCMS_API_KEY, "url": url}
    try:
        response = requests.get(WHATCMS_API_URL, params=params, timeout=REQUESTS_TIMEOUT); response.raise_for_status(); data = response.json(); result_info = data.get('result', {})
        if result_info.get('code') == 200:
            cms_found = None
            if 'results' in data:
                for tech in data['results']:
                    categories = tech.get('categories', []);
                    if isinstance(categories, (list, tuple)) and "CMS" in categories: cms_found = tech.get('name', 'Unknown CMS'); break
            return cms_found if cms_found else "CMS not detected by WhatCMS."
        else:
            error_code = result_info.get('code', 'N/A'); error_msg = result_info.get('msg', 'Unknown error')
            if error_code == 120: st.warning(f"⚠️ WhatCMS Rate Limited for {url}. Please wait before trying again."); return "CMS Check Failed (Rate Limit)"
            else: st.warning(f"⚠️ WhatCMS API Warning for {url}: Code {error_code} - {error_msg}"); return f"CMS Check Failed (API Code: {error_code})"
    except requests.exceptions.Timeout: st.warning(f"⏳ Timeout connecting to WhatCMS API for {url}."); return "CMS Check Failed (Timeout)"
    except requests.exceptions.RequestException as e: st.warning(f"🌐 Error fetching CMS data from WhatCMS API for {url}: {e}"); return "CMS Check Failed (Request Error)"
    except json.JSONDecodeError: st.warning(f"📄 Invalid JSON response from WhatCMS API for {url}."); return "CMS Check Failed (Invalid Response)"
    except Exception as e: st.warning(f"⚙️ Unexpected error processing WhatCMS response for {url}: {e}"); return f"CMS Check Failed (Error: {type(e).__name__})"
def export_to_google_doc(summary_text, folder_id=None):
    if not GOOGLE_LIBS_AVAILABLE: return "❌ Error: Google API libraries not installed."
    if not summary_text: return "❌ Error: No summary text provided."
    try:
        creds = service_account.Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=GOOGLE_ALL_SCOPES)
        drive_service = build('drive', 'v3', credentials=creds); docs_service = build('docs', 'v1', credentials=creds)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'); document_title = f"Product Page UX Summary - {timestamp}"; body = {'title': document_title}
        doc = docs_service.documents().create(body=body).execute(); document_id = doc.get('documentId'); doc_link = f"https://docs.google.com/document/d/{document_id}/edit"
        st.info(f"📄 Created Google Doc: {doc_link}")
        if folder_id:
            try:
                file_metadata = drive_service.files().get(fileId=document_id, fields='parents').execute(); previous_parents = ",".join(file_metadata.get('parents'))
                drive_service.files().update(fileId=document_id, addParents=folder_id, removeParents=previous_parents, fields='id, parents').execute(); st.info(f"📂 Moved document to Google Drive folder ID: {folder_id}")
            except HttpError as error:
                status_code = getattr(error.resp, 'status', None)
                if status_code == 404: st.warning(f"⚠️ Could not move document: Folder ID '{folder_id}' not found or permission denied.")
                elif status_code == 403: st.warning(f"⚠️ Could not move document: Permission denied for Folder ID '{folder_id}'.")
                else: st.warning(f"⚠️ Could not move document to folder '{folder_id}'. Google API Error: {error}")
            except Exception as e: st.warning(f"⚙️ Unexpected error moving document: {e}.")
        # Ensure summary_text is a string before cleaning
        if not isinstance(summary_text, str):
            st.warning("Summary text was not a string, attempting conversion for export.")
            summary_text = str(summary_text)
        cleaned_summary = summary_text.replace("**", "").replace("*", "").replace("### ", "").replace("## ", "").replace("# ", "")
        requests_body = [{'insertText': {'location': {'index': 1}, 'text': cleaned_summary}}]; docs_service.documents().batchUpdate(documentId=document_id, body={'requests': requests_body}).execute()
        return f"✅ Successfully exported summary to Google Doc: [View Document]({doc_link})"
    except FileNotFoundError: st.error(f"❌ Error: Google service account key file not found at '{GOOGLE_SERVICE_ACCOUNT_FILE}'."); return f"❌ Export Failed: File not found."
    except HttpError as error:
        st.error(f"❌ Google API Error during export: {error}")
        try: error_content = json.loads(error.content); error_details_msg = error_content.get('error', {}).get('message', str(error.content))
        except: error_details_msg = str(error.content)
        st.error(f"Details: {error_details_msg}"); st.error("Check service account permissions/quota."); return "❌ Export Failed: Google API error."
    except Exception as e: st.error(f"⚙️ Unexpected error during Google Doc export: {e}"); st.error(traceback.format_exc()); return f"❌ Export Failed: Unexpected error ({type(e).__name__})"
def detect_facebook_page(url):
    if not url: return None
    try:
        headers={'User-Agent':'Mozilla/5.0...'}; response = requests.get(url, timeout=REQUESTS_TIMEOUT, headers=headers, allow_redirects=True); response.raise_for_status()
        base_url = response.url; soup = BeautifulSoup(response.text, 'html.parser'); potential_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href or href.startswith(('#', 'mailto:', 'tel:')): continue
            try: absolute_href = urljoin(base_url, href); parsed_abs_url = urlparse(absolute_href)
            except ValueError: continue
            if 'facebook.com' in parsed_abs_url.netloc.lower():
                fb_host = parsed_abs_url.netloc.lower()
                if fb_host in ['www.facebook.com', 'facebook.com', 'm.facebook.com', 'web.facebook.com']:
                    path = parsed_abs_url.path.lower().strip('/')
                    exclude_paths_start = ('sharer', 'dialog', 'login', 'logout', 'plugins', 'watch', 'video', 'photo', 'story', 'events', 'notes', 'groups', 'marketplace', 'gaming', 'pages/create', 'pages/launchpoint', 'pages/browser', 'help', 'legal', 'policies', 'privacy', 'settings', 'developers', 'apps', 'badges', 'bookmarks', 'business_help', 'campaign', 'careers', 'contact_importer', 'directory', 'find-friends', 'fundraisers', 'games', 'groups_discover', 'imbox', 'instant_games', 'jobs', 'latest', 'livemap', 'lookaside.fbsbx.com', 'maps', 'media', 'memories', 'messages', 'mobile', 'movies', 'notifications', 'offers', 'page_insights', 'pages_manager', 'payments', 'people', 'permalink', 'photos', 'places', 'reactions', 'saved', 'search', 'security', 'share', 'stories', 'support', 'terms', 'weather', 'whitehat')
                    exclude_paths_contain = ('/posts/', '/videos/', '/photos/', '/reviews/', '/about/', '/community/')
                    path_segments = [seg for seg in path.split('/') if seg]
                    if path and path != 'home.php' and not path.startswith(exclude_paths_start) and not any(ex_path in f"/{path}/" for ex_path in exclude_paths_contain) and 'profile.php' not in path:
                        if len(path_segments) == 1 and not path_segments[0].isdigit(): potential_links.append(absolute_href)
                        elif len(path_segments) > 1 and path_segments[0] == 'pages': potential_links.append(absolute_href)
        if potential_links: potential_links.sort(key=len); return potential_links[0]
        else: return None
    except requests.exceptions.Timeout: st.warning(f"⏳ Timeout detecting FB page for {url}."); return None
    except requests.exceptions.RequestException as e: st.warning(f"🌐 Error fetching {url} for FB detection: {e}"); return None
    except Exception as e: st.error(f"⚙️ Error parsing {url} for FB detection: {e}"); st.error(traceback.format_exc()); return None
def check_active_ads(page_url_or_id, meta_token):
    if not page_url_or_id: return (None, "Facebook Page URL or ID not provided.")
    if not meta_token: return (None, "Meta Access Token not provided (required for check).")
    page_id = None; search_term = None
    if "facebook.com" in str(page_url_or_id):
        try: parsed_url = urlparse(page_url_or_id); path_segments = [seg for seg in parsed_url.path.split('/') if seg];
        except Exception as e: return (None, f"Error parsing Facebook URL: {e}")
        if not path_segments: return (None, "Could not extract identifier from Facebook URL path.")
        if path_segments[0].lower() == 'pages' and len(path_segments) > 1 and path_segments[-1].isdigit(): page_id = path_segments[-1]
        else: search_term = path_segments[-1]
    elif str(page_url_or_id).isdigit(): page_id = str(page_url_or_id)
    else: search_term = str(page_url_or_id)
    try:
        log_func = st.session_state.get("log_func", print)
        if not page_id and search_term:
            log_func(f"🔍 Searching for Page ID using term: '{search_term}'...")
            search_url = f"{META_API_URL}/pages/search"; search_params = {'q': search_term, 'fields': 'id,name', 'access_token': meta_token}
            search_resp = requests.get(search_url, params=search_params, timeout=REQUESTS_TIMEOUT)
            if search_resp.status_code == 400:
                try: err_data=search_resp.json().get('error',{}); err_msg=err_data.get('message','Bad Request'); err_code=err_data.get('code');
                except json.JSONDecodeError: return (None, f"Meta API Search Error (400): Bad Request - {search_resp.text}")
                if err_code == 190: err_msg += " (Likely invalid/expired token)"
                elif err_code == 10: err_msg += " (Permission denied)"
                return (None, f"Meta API Search Error (400): {err_msg}")
            search_resp.raise_for_status(); search_data = search_resp.json()
            if not search_data.get('data'): return (None, f"Could not find FB Page via search: '{search_term}'.")
            page_id = search_data['data'][0].get('id'); page_name = search_data['data'][0].get('name', 'Unknown'); log_func(f"✅ Found Page ID via search: {page_id} ('{page_name}')")
            if len(search_data['data']) > 1: log_func(f"ℹ️ Multiple pages found; using the first result.")
        if not page_id: return (None, "Could not determine Page ID.")
        log_func(f"🔗 Getting ad accounts linked to Page ID: {page_id}...")
        adaccounts_url = f"{META_API_URL}/{page_id}/adaccounts"; adaccounts_params = {'access_token': meta_token, 'fields': 'id,name'}
        adaccounts_response = requests.get(adaccounts_url, params=adaccounts_params, timeout=REQUESTS_TIMEOUT)
        if adaccounts_response.status_code != 200:
            error_msg = f"Getting ad accounts (Status {adaccounts_response.status_code})"
            try: error_data=adaccounts_response.json().get('error',{}); msg=error_data.get('message', adaccounts_response.text); code=error_data.get('code'); subcode=error_data.get('error_subcode'); error_msg += f": {msg} (Code: {code}, Subcode: {subcode})"
            except json.JSONDecodeError: error_msg += f": {adaccounts_response.text}"
            if code in [10, 200, 803, 190] or subcode in [1341011, 1341006] or 'permission' in msg.lower() or 'requires' in msg.lower() or 'cannot access' in msg.lower(): error_msg += " - Check token validity/permissions ('ads_read') and page access."
            return (None, error_msg)
        adaccounts_data = adaccounts_response.json()
        if 'error' in adaccounts_data: return (None, f"API Error getting ad accounts: {adaccounts_data['error'].get('message', 'Unknown')}")
        ad_accounts = adaccounts_data.get('data', [])
        if not ad_accounts: return ([], "No ad accounts linked or accessible.")
        log_func(f"✅ Found {len(ad_accounts)} accessible ad account(s). Checking...")
        active_campaign_details = []; account_errors = []
        for account in ad_accounts:
            account_id = account.get('id'); account_name = account.get('name', account_id)
            campaigns_url = f"{META_API_URL}/{account_id}/campaigns"; campaigns_params = {'access_token': meta_token, 'filtering': json.dumps([{'field': 'effective_status', 'operator': 'IN', 'value': ['ACTIVE']}]), 'fields': 'id,name,status,effective_status'}
            try:
                campaigns_response = requests.get(campaigns_url, params=campaigns_params, timeout=REQUESTS_TIMEOUT + 5); campaigns_data = campaigns_response.json()
                if 'error' in campaigns_data: err_msg = campaigns_data['error'].get('message', 'Unknown campaign fetch error'); account_errors.append(f"Acc {account_name}: {err_msg}"); continue
                campaigns_found = campaigns_data.get('data', [])
                if campaigns_found:
                    for campaign in campaigns_found:
                        campaign_id = campaign.get('id'); campaign_name = campaign.get('name', 'Unnamed'); act_numeric_id = account_id.replace('act_', '')
                        ads_manager_link = f"https://adsmanager.facebook.com/adsmanager/manage/campaigns?act={act_numeric_id}&selected_campaign_ids={campaign_id}"
                        active_campaign_details.append(f"'{campaign_name}' (ID: {campaign_id}) in Acc {act_numeric_id} - [View]({ads_manager_link})")
            except requests.exceptions.Timeout: account_errors.append(f"Acc {account_name}: Timeout")
            except requests.exceptions.RequestException as e: account_errors.append(f"Acc {account_name}: Request Error - {e}")
            except Exception as e_inner: account_errors.append(f"Acc {account_name}: Unexpected Error - {e_inner}")
        final_error_message = "; ".join(account_errors) if account_errors else None
        return active_campaign_details, final_error_message
    except requests.exceptions.Timeout: return (None, "Timeout connecting to Meta API.")
    except requests.exceptions.RequestException as e: return (None, f"Meta API Request Error: {e}")
    except Exception as e: st.error(f"⚙️ Unexpected error during Meta Ads check: {e}"); st.error(traceback.format_exc()); return (None, f"Unexpected error: {type(e).__name__}")


# --- Main App Layout Starts Here ---
st.title("🤖 Gemini Product Page UX Auditor")
st.markdown("Analyzes product page screenshots (taken automatically via Playwright) with Gemini Vision, detects CMS, optionally checks Facebook Pages/Meta Ads, and exports summaries.")
if not GEMINI_LIBS_AVAILABLE: st.error("INSTALLATION REQUIRED: Run `pip install google-generativeai` in your terminal.")
if not PLAYWRIGHT_AVAILABLE: st.error("INSTALLATION REQUIRED: Run `pip install playwright && playwright install` in your terminal.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Analysis Inputs")
    product_page_url = st.text_input("Product Page URL (Required):", placeholder="https://example.com/product/xyz", key="pp_url")
    st.header("Export Options")
    google_drive_folder_id = st.text_input("Google Drive Folder ID (Optional):", placeholder="Enter Folder ID", help="If provided, exported Doc is moved here.", key="gdrive_id")
    st.header("Additional Checks")
    do_check_facebook = st.checkbox("Detect Facebook Page Link?", value=True, key="fb_check", help="Checks the Product Page URL")
    do_check_meta_ads = st.checkbox("Check for Active Meta Ads?", value=False, help="Requires a valid Meta Access Token.", key="ads_check", )
    meta_token_for_check = META_ACCESS_TOKEN_HARDCODED if do_check_meta_ads else None

# --- Main Area ---
# Display Config and Info first
col1, col2 = st.columns(2)
with col1:
    st.subheader("⚙️ Analysis Configuration")
    st.write(f"**Product Page URL:** {product_page_url or '_Not Provided_'}")
    st.write(f"**Screenshots:** Automated via Playwright ({MAX_SCROLLS+1} max views each for Desktop & Mobile)")
    st.write(f"**Check Facebook Page:** {'Yes' if do_check_facebook else 'No'}")
    st.write(f"**Check Meta Ads:** {'Yes' if do_check_meta_ads else 'No'}")
    if do_check_meta_ads: st.caption("⚠️ Using hardcoded Meta Access Token.")
with col2:
    st.subheader("🌐 Website Info (from Product URL)")
    url_for_info = product_page_url
    if url_for_info:
        domain = get_domain_from_url(url_for_info)
        st.write(f"**Domain:** {domain or '_Could not parse_'}")
        cms_display = st.session_state.get('cms_result', '_Check will run during analysis_')
        st.write(f"**Detected CMS:** {cms_display}")
    else:
        st.write("_Enter Product Page URL to get website info._")
        if 'cms_result' in st.session_state: del st.session_state.cms_result

st.markdown("---") # Divider before button

# --- State Variables ---
if 'analysis_ran' not in st.session_state: st.session_state.analysis_ran = False
if 'master_summary_text' not in st.session_state: st.session_state.master_summary_text = ""
# Removed: 'vision_analysis_results_dict'
if 'last_taken_screenshots_dict' not in st.session_state: st.session_state.last_taken_screenshots_dict = {"desktop": [], "mobile": []}
if 'detected_fb_url' not in st.session_state: st.session_state.detected_fb_url = None
if 'active_ads_result' not in st.session_state: st.session_state.active_ads_result = (None, None)
if 'cms_result' not in st.session_state: st.session_state.cms_result = None
if 'log_messages' not in st.session_state: st.session_state.log_messages = []

# --- Analysis Trigger ---
if st.button("🚀 Run Analysis & Checks", key="run_button"):
    st.session_state.analysis_ran = True
    st.session_state.master_summary_text = ""
    # Removed: st.session_state.vision_analysis_results_dict = {"desktop": [], "mobile": []}
    st.session_state.last_taken_screenshots_dict = {"desktop": [], "mobile": []}
    st.session_state.detected_fb_url = None
    st.session_state.active_ads_result = (None, None)
    st.session_state.cms_result = None
    st.session_state.log_messages = [] # Clear previous logs

    if not product_page_url: st.warning("⚠️ Please provide the Product Page URL."); st.session_state.analysis_ran = False; st.stop()
    if not PLAYWRIGHT_AVAILABLE: st.error("❌ Playwright not installed/available."); st.session_state.analysis_ran = False; st.stop()

    # --- EXECUTION LOG SETUP ---
    log_placeholder = st.empty()
    def log_update(message):
        st.session_state.log_messages.append(f"- {datetime.datetime.now().strftime('%H:%M:%S')} | {message}")
        log_placeholder.markdown("##### 📊 Analysis Execution Log\n" + "\n".join(st.session_state.log_messages))
    st.session_state.log_func = log_update

    log_update("Starting Analysis Process...")

    # Run CMS Check first
    if product_page_url:
        log_update("🔍 Checking CMS...")
        with st.spinner("Detecting CMS..."): st.session_state.cms_result = get_cms_with_whatcms(product_page_url)
        log_update(f"✅ CMS Check Result: {st.session_state.cms_result}")

    # Run Playwright Screenshotting
    log_update(f"📸 Capturing Desktop & Mobile screenshots for {product_page_url}...")
    screenshot_dict = {}
    with st.spinner("Running Playwright... (this may take a minute or two)"): screenshot_dict = run_playwright_sync(product_page_url)
    st.session_state.last_taken_screenshots_dict = screenshot_dict
    if not screenshot_dict.get("desktop") and not screenshot_dict.get("mobile"): log_update("❌ Failed to capture any screenshots.")
    else: log_update(f"✅ Captured {len(screenshot_dict.get('desktop',[]))} desktop & {len(screenshot_dict.get('mobile',[]))} mobile screenshots.")

    # --- REMOVED: Individual Screenshot Analysis Loop ---

    # Generate Master Summary using Single Multimodal Call
    screenshots_available = bool(st.session_state.last_taken_screenshots_dict.get("desktop")) or bool(st.session_state.last_taken_screenshots_dict.get("mobile"))
    if screenshots_available and gemini_configured and GEMINI_LIBS_AVAILABLE:
        log_update("📝 Generating Analysis & Summary using Gemini Vision...")
        try:
            domain_for_summary = get_domain_from_url(product_page_url) or "Analyzed Product Page"
            with st.spinner(f"Generating analysis & summary with Gemini ({GEMINI_MULTIMODAL_MODEL_NAME})..."):
                 st.session_state.master_summary_text = generate_analysis_and_summary(st.session_state.last_taken_screenshots_dict, domain_for_summary)
            log_update("✅ Analysis & Summary generated.")
        except Exception as e: log_update(f"❌ Error generating analysis/summary: {e}"); st.error(f"Failed to generate analysis/summary: {e}\n{traceback.format_exc()}"); st.session_state.master_summary_text = f"Error generating summary: {e}"
    elif not screenshots_available: log_update("⚠️ No screenshots available to generate summary."); st.session_state.master_summary_text = "Summary not generated: No screenshots were captured."
    else: log_update("⚠️ Gemini not configured/available, skipping summary generation."); st.session_state.master_summary_text = "Summary not generated: Gemini unavailable."

    # Perform Additional Checks
    log_update("⚙️ Performing Additional Checks...")
    url_for_checks = product_page_url
    if url_for_checks:
        if do_check_facebook:
            log_update("🔗 Detecting Facebook Page link...")
            # Corrected Indentation for spinner:
            with st.spinner("Scanning for Facebook link..."):
                detected_fb = detect_facebook_page(url_for_checks)
            st.session_state.detected_fb_url = detected_fb
            status_msg = f"✅ Facebook Page link detected: {detected_fb}" if detected_fb else "⚠️ No valid Facebook Page link found."
            log_update(status_msg) # Log result after spinner finishes
        else:
            log_update("⚪ Skipping Facebook Page detection.")

        if do_check_meta_ads:
            log_update("💰 Checking for active Meta Ads...")
            if not meta_token_for_check:
                log_update("⚠️ Skipping Meta Ads check (Token missing).")
                st.session_state.active_ads_result = (None, "Token missing.")
            elif do_check_facebook and st.session_state.detected_fb_url:
                 # Corrected Indentation for spinner:
                 with st.spinner("Checking Meta Ads status..."):
                     ads_details, ads_error = check_active_ads(st.session_state.detected_fb_url, meta_token_for_check)
                 st.session_state.active_ads_result = (ads_details, ads_error)
                 # Log result after spinner finishes
                 if ads_error and not ads_details: log_update(f"❌ Meta Ads check failed/Warning: {ads_error}")
                 elif ads_details: log_update(f"✅ Meta Ads check complete: Found {len(ads_details)} active campaign(s).")
                 elif ads_error and ads_details: log_update(f"✅ Meta Ads check partially complete: Found {len(ads_details)} active campaign(s), but errors: {ads_error}")
                 else: log_update(f"✅ Meta Ads check complete: {ads_error or 'No active campaigns found.'}")
            elif do_check_facebook:
                log_update("⚠️ Skipping Meta Ads check (No Facebook Page URL detected).")
                st.session_state.active_ads_result = (None, "No Facebook Page URL detected.")
            else:
                log_update("⚠️ Skipping Meta Ads check (Facebook Page detection was disabled).")
                st.session_state.active_ads_result = (None, "Facebook detection disabled.")
        else:
            log_update("⚪ Skipping Meta Ads check.")
            st.session_state.active_ads_result = (None, "Ads check disabled.") # Ensure state is set when skipped

    log_update("✅ Analysis & Checks Completed!")
    # Keep log visible
    # log_placeholder.empty()
    # --- EXECUTION LOG GENERATION ENDS HERE ---


# --- Display Results Area ---
if st.session_state.analysis_ran:
    st.markdown("---") # Divider before results dashboard
    st.header("📈 Analysis Results Dashboard")

    # Display Top Info: Website Info & Additional Checks
    col_info_res, col_checks_res = st.columns(2)
    with col_info_res:
        st.subheader("🌐 Website Info")
        url_for_info_res = product_page_url
        if url_for_info_res:
             domain = get_domain_from_url(url_for_info_res); st.write(f"**Product Page URL:** {url_for_info_res}"); st.write(f"**Domain:** {domain or '_Could not parse_'}")
             cms_res_display = st.session_state.get('cms_result', '_Not checked_'); st.write(f"**Detected CMS:** {cms_res_display}")
        else: st.write("_No URL provided during analysis run._")
    with col_checks_res:
        st.subheader("🔗 Additional Check Results")
        if url_for_info_res:
            st.markdown("**Facebook Page Detection:**")
            if do_check_facebook:
                fb_link = st.session_state.get('detected_fb_url')
                if fb_link is not None:
                    if fb_link: st.success(f"✅ Link Found: [{fb_link}]({fb_link})")
                    else: st.warning("⚠️ No valid Facebook Page link detected.")
                else: st.info("ℹ️ Detection skipped or failed.")
            else: st.info("ℹ️ Detection was disabled by user.")
            st.markdown("**Meta Ads Check:**")
            ads_details, ads_error = st.session_state.active_ads_result
            if do_check_meta_ads:
                if ads_details is not None or ads_error is not None:
                    if ads_error and not ads_details: st.error(f"❌ Check Failed/Warning: {ads_error}")
                    elif ads_details:
                         st.success(f"✅ Active Campaigns Found ({len(ads_details)}):"); [st.markdown(f"- {ad_info}") for ad_info in ads_details]
                         if ads_error: st.warning(f"⚠️ Issues during check: {ads_error}")
                    elif ads_error: st.error(f"❌ Check Failed/Warning: {ads_error}")
                    else: st.success("✅ No active ad campaigns found or accessible.")
                else: st.info("ℹ️ Check skipped or failed.")
            else: st.info("ℹ️ Check was disabled by user.")
        else: st.info("Additional checks require a Product Page URL.")

    st.markdown("---")

    # Display Master Summary & Export Button NEXT
    st.subheader("✨ Master Summary (Gemini Generated from Vision Analysis) ✨")
    if not st.session_state.master_summary_text: st.info("Analysis run, but summary could not be generated (check logs).")
    else:
        st.markdown(st.session_state.master_summary_text) # Display the generated summary/report
        if GOOGLE_LIBS_AVAILABLE:
            if st.button("📄 Export Summary to Google Doc", key="export_button"):
                with st.spinner("📤 Exporting to Google Docs..."): export_result = export_to_google_doc(st.session_state.master_summary_text, google_drive_folder_id)
                if "✅" in export_result: st.success(export_result)
                else: st.error(export_result)
        else: st.warning("⚠️ Google Doc export unavailable.")

    # --- REMOVED Detailed Screenshot Display Section ---

    # Display Execution Log LAST
    st.markdown("---")
    st.subheader("📊 Analysis Execution Log (Recap)")
    if st.session_state.log_messages:
        log_text = "\n".join(st.session_state.log_messages)
        st.markdown(f"```\n{log_text}\n```") # Use markdown code block
    else: st.info("Log not available (analysis might not have completed).")


# --- Footer ---
st.markdown("---")
st.caption("Gemini Product Page UX Auditor | Requires: streamlit, requests, beautifulsoup4, google-api-python-client, google-auth-httplib2, google-auth-oauthlib, google-generativeai, Pillow, playwright. Ensure `service_account_key.json` exists and API keys are set. Run `playwright install`.")