import google.generativeai as genai
from playwright.sync_api import sync_playwright
from PIL import Image
import io
import os
import json
import time
from typing import Optional
import re # Import the regex module

# --- Configuration ---
# WARNING: Hardcoding API keys is insecure. Use environment variables or secrets management in production.
GEMINI_API_KEY = "AIzaSyDAfqg0tqIkVAE_DV4vd6OOjJz_pXdnHso" # Replace with your actual key if needed

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Initialize the Gemini models
    # Using gemini-2.5-flash-preview-04-17 as requested
    vision_model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    text_model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # Using the same model for text
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # Handle the error appropriately, maybe raise it or exit
    vision_model = None
    text_model = None

# --- Browser Interaction (Playwright) ---

def capture_scrolling_screenshots(url: str, viewport_size: dict = None) -> list[bytes]:
    """Navigates to a URL, scrolls down, and takes screenshots at intervals."""
    screenshots = []
    try:
        print(f"[{url}] Starting Playwright browser launch for scrolling screenshots...")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            print(f"[{url}] Browser launched. Creating new page...")
            context_options = {}
            if viewport_size:
                 context_options['viewport'] = viewport_size

            page = browser.new_page(**context_options)
            print(f"[{url}] Page created. Navigating to URL with wait_until='commit' and timeout=120000ms...")
            page.goto(url, wait_until="commit", timeout=120000)
            print(f"[{url}] Navigation committed. Waiting for 3 seconds for dynamic content...")
            time.sleep(3) # Wait a bit for dynamic content

            # Get the total scrollable height of the page
            total_height = page.evaluate("document.body.scrollHeight")
            viewport_height = page.viewport_size['height']
            scroll_position = 0

            print(f"[{url}] Total height: {total_height}, Viewport height: {viewport_height}")

            while scroll_position <= total_height:
                print(f"[{url}] Scrolling to position: {scroll_position}")
                page.evaluate(f"window.scrollTo(0, {scroll_position})")
                time.sleep(1) # Give a moment for content to settle after scroll
                screenshot_bytes = page.screenshot() # Take screenshot of the current viewport
                screenshots.append(screenshot_bytes)
                print(f"[{url}] Screenshot captured at scroll position {scroll_position}.")

                scroll_position += viewport_height # Scroll down by one viewport height

            browser.close()
            print(f"[{url}] Browser closed. Captured {len(screenshots)} screenshots.")
            return screenshots
    except Exception as e:
        print(f"[{url}] Error during scrolling screenshot process: {e}")
        return []

# --- Gemini API Interaction ---

def analyze_image_with_gemini(image_bytes: bytes, prompt: str) -> Optional[str]:
    """Sends an image and prompt to Gemini Vision model."""
    if not vision_model:
        print("Gemini Vision model not initialized.")
        return None
    try:
        image_parts = [
            {"mime_type": "image/png", "data": image_bytes}
        ]
        response = vision_model.generate_content([prompt, image_parts[0]], stream=False)
        response.resolve() # Ensure response is fully generated
        return response.text
    except Exception as e:
        print(f"Error analyzing image with Gemini: {e}")
        # Check for specific API errors if needed (e.g., content filtering)
        # if hasattr(e, 'message') and 'block_reason' in e.message:
        #     print(f"Content blocked: {e.message}")
        return f"Error during Gemini Vision analysis: {e}"

def analyze_images_with_gemini(image_bytes_list: list[bytes], prompt: str) -> Optional[str]:
    """Sends a list of images and prompt to Gemini Vision model."""
    if not vision_model:
        print("Gemini Vision model not initialized.")
        return None
    try:
        image_parts = [{"mime_type": "image/png", "data": img_bytes} for img_bytes in image_bytes_list]
        content_parts = [prompt] + image_parts
        response = vision_model.generate_content(content_parts, stream=False)
        response.resolve()
        return response.text
    except Exception as e:
        print(f"Error analyzing images with Gemini: {e}")
        return f"Error during Gemini Vision analysis: {e}"


def analyze_text_with_gemini(text_input: str, prompt: str) -> Optional[str]:
    """Sends text input and prompt to Gemini text model."""
    if not text_model:
        print("Gemini Text model not initialized.")
        return None
    try:
        full_prompt = f"{prompt}\n\n{text_input}"
        response = text_model.generate_content(full_prompt, stream=False)
        response.resolve() # Ensure response is fully generated
        return response.text
    except Exception as e:
        print(f"Error analyzing text with Gemini: {e}")
        return f"Error during Gemini Text analysis: {e}"

# --- Summary Generation ---
def generate_summary(report_text: str) -> Optional[str]:
    """Generates a concise summary of the audit report using Gemini text model."""
    if not report_text:
        return "No report text to summarize."
    prompt = "Provide a concise summary of the following website audit report, focusing on key findings, gaps, and opportunities for improvement. Format the summary as bullet points."
    return analyze_text_with_gemini(report_text, prompt)


# --- Report Generation & Orchestration ---

# Placeholder for the main audit function
def run_audit(url: str, page_type: str, update_status=None) -> dict:
    """Orchestrates the website audit process."""
    print(f"Running audit for: {url}, Page Type: {page_type}")
    results = {"desktop": None, "mobile": None}

    # Define viewports
    desktop_viewport = {"width": 1920, "height": 1080}
    mobile_viewport = {"width": 390, "height": 844, "is_mobile": True} # iPhone 13 Pro dimensions

    # --- Desktop Audit ---
    print("Starting Desktop Audit...")
    # Use the new function to capture scrolling screenshots
    desktop_screenshots = capture_scrolling_screenshots(url, viewport_size=desktop_viewport)
    if desktop_screenshots:
        print(f"Captured {len(desktop_screenshots)} desktop screenshots.")
        # Define prompts based on page_type - modify to analyze multiple screenshots
        desktop_vision_prompt = f"Analyze this sequence of DESKTOP screenshots of a '{page_type}' page ({url}) captured while scrolling down. Focus on overall layout, clarity, visual appeal, calls to action, and how elements behave (e.g., sticky headers/footers) as the user scrolls."
        if page_type == 'Product Page':
             desktop_vision_prompt += """
Specifically check for:
- Above-the-Fold Clarity: Is key product info (name, price, main image) instantly visible in the first screenshot?
- Image & Visual Trust Cues: Are product images clear across the screenshots? Are there multiple angles or lifestyle/contextual images visible as you scroll? Are trust badges visible?
- CTA & Checkout Flow: Is the primary Call to Action (e.g., 'Add to Cart') clear and prominent? Does it remain visible (sticky) as you scroll down the page?
- Conversion Elements: Are elements like delivery info, reviews, or trust badges easily visible as you scroll?
- Sticky Elements: Are there any sticky headers, footers, or CTAs that remain in place as the user scrolls?
Provide a concise summary for each point in the following format:
- What to Check: [Name of the check, e.g., Above-the-Fold Clarity]
- Why it Matters: [Brief explanation, e.g., Key info visible instantly]
- Gaps or Opportunities: [Your analysis/findings for this specific check based on the screenshots]

Repeat this structure for each of the checks listed above for the DESKTOP view, considering the scrolling behavior shown in the sequence of screenshots.
"""
        else: # Homepage or other
             desktop_vision_prompt += """
Provide a UX/SEO audit of this sequence of DESKTOP screenshots of a 'Homepage' page ({url}) captured while scrolling down. Focus on overall impression, clarity of purpose, visual hierarchy, calls to action, navigation, and how elements behave (e.g., sticky headers/footers) as the user scrolls.

**Overall Impression & Clarity of Purpose:**
*   **Findings:** What is your initial impression from the first screenshot? Is the website's purpose and value proposition immediately clear? Does this change as you scroll?
*   **Recommendations/Opportunities:** Suggest ways to improve clarity or first impression.

**Visual Hierarchy & Appeal:**
*   **Findings:** Is the layout visually appealing across the screenshots? Is there a clear visual hierarchy guiding the user's eye to important elements as you scroll?
*   **Recommendations/Opportunities:** Suggest improvements to visual design or hierarchy.

**Primary Calls to Action:**
*   **Findings:** Are the main calls to action (CTAs) prominent and clear in the screenshots? Do they remain visible (sticky) as you scroll down the page?
*   **Recommendations/Opportunities:** Suggest ways to improve CTA visibility, clarity, placement, or stickiness.

**Navigation:**
*   **Findings:** Is the main navigation easy to find and use in the screenshots? Does it remain visible (sticky) as you scroll? Is it clear where different links lead?
*   **Recommendations/Opportunities:** Suggest improvements to navigation structure, design, or stickiness.

**Trust Signals / Social Proof:**
*   **Findings:** Are there any visible trust signals (e.g., security badges, testimonials, review counts) in the screenshots? Are they consistently visible as you scroll?
*   **Recommendations/Opportunities:** Suggest adding or improving trust signals.

**Sticky Elements:**
*   **Findings:** Are there any sticky headers, footers, or other elements that remain in place as the user scrolls through the page?
*   **Recommendations/Opportunities:** Comment on the effectiveness of sticky elements or suggest adding them.
"""

        # Analyze the list of screenshots
        desktop_analysis = analyze_images_with_gemini(desktop_screenshots, desktop_vision_prompt)
        if desktop_analysis:
            print("Desktop analysis received from Gemini.")
            # Structure the report - need to adapt parsing for the new prompt structure
            report_text = f"Desktop Audit for {url} ({page_type}):\n{desktop_analysis}"
            structured_audit = [] # Need to implement parsing for the new structure

            # Attempt to parse structured data based on the new prompt structure
            try:
                sections = report_text.split('**')
                for i in range(1, len(sections), 2):
                    section_header = sections[i].strip().replace(':', '')
                    section_content = sections[i+1].strip()

                    findings_match = re.search(r'\*\*Findings:\*\*(.*?)(?=\*\*Recommendations/Opportunities:\*\*|$)', section_content, re.DOTALL)
                    recommendations_match = re.search(r'\*\*Recommendations/Opportunities:\*\*(.*)', section_content, re.DOTALL)

                    findings = findings_match.group(1).strip() if findings_match else "N/A"
                    recommendations = recommendations_match.group(1).strip() if recommendations_match else "N/A"

                    structured_audit.append({
                         "area": section_header,
                         "findings": findings,
                         "recommendations": recommendations
                     })
            except Exception as parse_e:
                print(f"Could not parse structured desktop report: {parse_e}")
                structured_audit = [{"area": "Overall Analysis", "findings": desktop_analysis, "recommendations": "N/A"}] # Fallback


            # Generate summary for desktop report
            desktop_summary = generate_summary(report_text)
            if desktop_summary:
                print("Desktop summary generated.")
            else:
                print("Desktop summary generation failed.")

            results["desktop"] = {
                "screenshots": desktop_screenshots, # Include list of screenshot bytes
                "report_text": report_text,
                "report_json": {"url": url, "page_type": page_type, "view": "desktop", "audit": structured_audit},
                "summary": desktop_summary # Include summary
            }
        else:
            print("Desktop analysis failed.")
            results["desktop"] = {"screenshots": desktop_screenshots, "report_text": "Desktop analysis failed.", "report_json": {}, "summary": "Desktop analysis failed."} # Include screenshots even on analysis failure
    else:
        if update_status:
            update_status("Failed to capture desktop scrolling screenshots.")
        else:
            print("Failed to capture desktop scrolling screenshots.")
        results["desktop"] = {"screenshots": [], "report_text": "Failed to capture desktop scrolling screenshots.", "report_json": {}, "summary": "Failed to capture desktop scrolling screenshots."}


    # --- Mobile Audit ---
    print("\nStarting Mobile Audit...")
    # Use the new function to capture scrolling screenshots
    mobile_screenshots = capture_scrolling_screenshots(url, viewport_size=mobile_viewport)
    if mobile_screenshots:
        print(f"Captured {len(mobile_screenshots)} mobile screenshots.")
        # Define prompts based on page_type - modify to analyze multiple screenshots
        mobile_vision_prompt = f"Analyze this sequence of MOBILE screenshots of a '{page_type}' page ({url}) captured while scrolling down. Focus on mobile optimization, readability, touch target size, navigation, and how elements behave (e.g., sticky headers/footers) as the user scrolls."
        if page_type == 'Product Page':
             mobile_vision_prompt += """
Analyze this sequence of MOBILE screenshots of a 'Product Page' page ({url}) captured while scrolling down. Provide a UX/SEO audit focusing on the following areas. For each area, provide your Findings based on the screenshot analysis and Recommendations/Opportunities for improvement.

**Sticky Elements:**
*   **Findings:** **Examine the sequence of screenshots carefully.** Does the primary Call to Action (e.g., 'Add to Cart') remain visible and in a fixed position (sticky) as you scroll down the page? Are there any other sticky headers, footers, or elements? Describe the behavior of sticky elements. **It is critical to accurately identify if the 'Add to Cart' button is sticky.**
*   **Recommendations/Opportunities:** Comment on the effectiveness of sticky elements or suggest adding/improving them, particularly for the 'Add to Cart' button.

**Above-the-Fold Clarity:**
*   **Findings:** Is key product info (name, price, main image) instantly visible in the first screenshot on mobile?
*   **Recommendations/Opportunities:** Suggest ways to improve clarity above the fold on mobile.

**Image & Visual Trust Cues:**
*   **Findings:** Are images optimized for mobile across the screenshots? Are trust cues visible without excessive scrolling?
*   **Recommendations/Opportunities:** Suggest improvements to product images or trust cues.

**CTA & Checkout Flow:**
*   **Findings:** Is the primary Call to Action (e.g., 'Add to Cart') button large enough and easy to tap on mobile? Is the checkout process streamlined for mobile? (Note: Sticky CTA behavior is covered in the 'Sticky Elements' section).
*   **Recommendations/Opportunities:** Suggest ways to improve CTA tap-ability or the checkout process.

**Conversion Elements:**
*   **Findings:** Are elements like delivery info, reviews, or trust badges easily visible on mobile as you scroll?
*   **Recommendations/Opportunities:** Suggest adding or improving conversion elements.

**Mobile Optimisation:**
*   **Findings:** Is the layout responsive and easy to use on a small screen across the screenshots? Are touch targets appropriately sized?
*   **Recommendations/Opportunities:** Suggest improvements for overall mobile usability.
"""
        else: # Homepage or other
             mobile_vision_prompt += """
Provide a UX/SEO audit of this sequence of MOBILE screenshots of a 'Homepage' page ({url}) captured while scrolling down. Focus on mobile optimization, readability, touch target size, navigation, and how elements behave (e.g., sticky headers/footers) as the user scrolls.

**Overall Impression & Clarity of Purpose:**
*   **Findings:** What is your initial impression from the first screenshot on mobile? Is the website's purpose and value proposition immediately clear? Does this change as you scroll?
*   **Recommendations/Opportunities:** Suggest ways to improve clarity or first impression on mobile.

**Mobile Optimisation & Visual Appeal:**
*   **Findings:** Is the layout responsive and easy to use on a small screen across the screenshots? Are images and other visual elements optimized for mobile?
*   **Recommendations/Opportunities:** Suggest improvements for mobile usability and visual appeal.

**Navigation:**
*   **Findings:** Is the mobile navigation (e.g., hamburger menu) clear and easy to find and use in the screenshots? Are touch targets appropriately sized? Does the navigation remain visible (sticky) as you scroll?
*   **Recommendations/Opportunities:** Suggest improvements to mobile navigation or touch target sizes.

**Primary Calls to Action:**
*   **Findings:** Are the main calls to action (CTAs) prominent and easy to tap on mobile in the screenshots? Is the primary CTA sticky when scrolling?
*   **Recommendations/Opportunities:** Suggest ways to improve mobile CTA visibility, tap-ability, or stickiness.

**Trust Signals / Social Proof:**
*   **Findings:** Are there any visible trust signals (e.g., security badges, testimonials, review counts) on the mobile screen in the screenshots? Are they consistently visible as you scroll?
*   **Recommendations/Opportunities:** Suggest adding or improving trust signals for mobile users.

**Sticky Elements:**
*   **Findings:** Are there any sticky headers, footers, or other elements that remain in place as the user scrolls through the page on mobile?
*   **Recommendations/Opportunities:** Comment on the effectiveness of sticky elements or suggest adding them.
"""
        mobile_analysis = analyze_images_with_gemini(mobile_screenshots, mobile_vision_prompt)
        if mobile_analysis:
            print("Mobile analysis received from Gemini.")
            # Structure the report - need to adapt parsing for the new prompt structure
            report_text = f"Mobile Audit for {url} ({page_type}):\n{mobile_analysis}"
            structured_audit = [] # Need to implement parsing for the new structure

            # Attempt to parse structured data based on the new prompt structure
            try:
                sections = report_text.split('**')
                for i in range(1, len(sections), 2):
                     section_header = sections[i].strip().replace(':', '')
                     section_content = sections[i+1].strip()

                     findings_match = re.search(r'\*\*Findings:\*\*(.*?)(?=\*\*Recommendations/Opportunities:\*\*|$)', section_content, re.DOTALL)
                     recommendations_match = re.search(r'\*\*Recommendations/Opportunities:\*\*(.*)', section_content, re.DOTALL)

                     findings = findings_match.group(1).strip() if findings_match else "N/A"
                     recommendations = recommendations_match.group(1).strip() if recommendations_match else "N/A"

                     structured_audit.append({
                         "area": section_header,
                         "findings": findings,
                         "recommendations": recommendations
                     })
            except Exception as parse_e:
                print(f"Could not parse structured mobile report: {parse_e}")
                structured_audit = [{"area": "Overall Analysis", "findings": mobile_analysis, "recommendations": "N/A"}] # Fallback


            # Generate summary for mobile report
            mobile_summary = generate_summary(report_text)
            if mobile_summary:
                print("Mobile summary generated.")
            else:
                print("Mobile summary generation failed.")

            results["mobile"] = {
                "screenshots": mobile_screenshots, # Include list of screenshot bytes
                "report_text": report_text,
                "report_json": {"url": url, "page_type": page_type, "view": "mobile", "audit": structured_audit},
                "summary": mobile_summary # Include summary
            }
        else:
            print("Mobile analysis failed.")
            results["mobile"] = {"screenshots": mobile_screenshots, "report_text": "Mobile analysis failed.", "report_json": {}, "summary": "Mobile analysis failed."} # Include screenshots even on analysis failure
    else:
        if update_status:
            update_status("Failed to capture mobile scrolling screenshots.")
        else:
            print("Failed to capture mobile scrolling screenshots.")
        results["mobile"] = {"screenshots": [], "report_text": "Failed to capture mobile scrolling screenshots.", "report_json": {}, "summary": "Failed to capture mobile scrolling screenshots."} # Include screenshot as None

    print("\nAudit process finished.")
    return results

# Example usage (for testing purposes)
if __name__ == "__main__":
    test_url = "https://www.google.com" # Replace with a test URL
    test_page_type = "Homepage"
    audit_data = run_audit(test_url, test_page_type)
    print("\n--- FINAL RESULTS ---")
    print(json.dumps(audit_data, indent=4))
