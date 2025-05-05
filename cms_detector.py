import requests
from playwright.sync_api import sync_playwright
from typing import Optional

def fetch_page_content_and_headers(url: str) -> tuple[Optional[str], Optional[dict]]:
    """Fetches HTML content and headers for a given URL using Playwright."""
    html_content = None
    headers = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            response = page.goto(url, wait_until="networkidle", timeout=60000)
            html_content = page.content()
            headers = response.headers if response else None
            browser.close()
            return html_content, headers
    except Exception as e:
        print(f"Error fetching content and headers for {url}: {e}")
        return None, None


def detect_cms_from_content(url: str) -> str:
    """Detects CMS based on HTML content and HTTP headers."""
    html_content, headers = fetch_page_content_and_headers(url)

    if not html_content and not headers:
        return "Detection Failed: No content or headers received."

    # Simple checks for common CMS patterns in HTML
    if html_content:
        if 'wp-content' in html_content or 'wp-admin' in html_content:
            return "WordPress"
        if 'joomla.xml' in html_content or 'Joomla!' in html_content:
            return "Joomla"
        if 'drupal.js' in html_content or 'Drupal' in html_content:
            return "Drupal"
        if 'shopify-assets' in html_content or 'Shopify' in html_content:
            return "Shopify"
        if '<!-- Made with Jekyll' in html_content:
            return "Jekyll"
        # Add more HTML patterns here

    # Simple checks for common CMS patterns in Headers
    if headers:
        # Headers are case-insensitive, so check lowercased keys
        headers_lower = {k.lower(): v for k, v in headers.items()}
        if 'x-powered-by' in headers_lower:
            if 'WordPress' in headers_lower['x-powered-by']:
                return "WordPress"
            if 'Joomla' in headers_lower['x-powered-by']:
                return "Joomla"
            if 'Drupal' in headers_lower['x-powered-by']:
                return "Drupal"
        if 'server' in headers_lower:
             if 'nginx' in headers_lower['server'].lower():
                 # Nginx is a web server, not a CMS, but can host many CMSs.
                 # We might look for other clues if Nginx is detected.
                 pass # Add more checks if needed
        if 'x-generator' in headers_lower:
             if 'WordPress' in headers_lower['x-generator']:
                 return "WordPress"
             if 'Joomla' in headers_lower['x-generator']:
                 return "Joomla"
             if 'Drupal' in headers_lower['x-generator']:
                 return "Drupal"
             if 'Shopify' in headers_lower['x-generator']:
                 return "Shopify"
        # Add more header patterns here

    return "Unknown or Custom CMS"

# Example usage (for testing purposes)
if __name__ == "__main__":
    test_url = "https://www.wordpress.org" # Replace with a test URL
    detected = detect_cms_from_content(test_url)
    print(f"Detected CMS for {test_url}: {detected}")

    test_url_2 = "https://www.shopify.com" # Replace with another test URL
    detected_2 = detect_cms_from_content(test_url_2)
    print(f"Detected CMS for {test_url_2}: {detected_2}")
