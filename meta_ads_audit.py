# meta_ads_audit.py
from apify_client import ApifyClient
import json # Import json for pretty printing in __main__
from datetime import datetime, timedelta # Import datetime for ad fatigue check

def meta_ads_audit(url):
    """
    Audits up to 5 Facebook ads for a given page URL using the Apify client.

    Args:
        url (str): The URL of the Facebook page to audit.

    Returns:
        tuple: (list containing zero to five ad items, error message or None)
    """
    # Initialize the ApifyClient with your API token
    # Consider using st.secrets or environment variables
    client = ApifyClient("apify_api_b2OJWoWIC6a7S9oDt11ZYFR07NfNs00HJWkX") # Use your actual token

    # Prepare the Actor input - Limit to 5 results
    run_input = {
        "startUrls": [{ "url": url }],
        "resultsLimit": 5,  # <<< CHANGE: Limit set to 5 ads >>>
        "activeStatus": "", # Fetch active or inactive (can change to "active" if desired)
        # We assume default for 'isDetailsPerAd' is sufficient (likely false)
    }

    print(f"Starting Actor run for URL: {url} (Limit: 5 ads)...") # Updated print message
    try:
        run = client.actor("JJghSZmShuco4j9gJ").call(run_input=run_input)
        print(f"Actor run finished. Run ID: {run.get('id')}")

        dataset_id = run.get("defaultDatasetId")
        if dataset_id:
            print(f"Fetching results from Dataset ID: {dataset_id} (Limit: 5 items)")
            audited_items = []
            # iterate_items will now fetch at most 5 items because of the input limit
            for item in client.dataset(dataset_id).iterate_items():
                # Perform audit checks on the item
                audit_findings = analyze_ad(item)
                item['audit_findings'] = audit_findings # Add findings to the item
                audited_items.append(item) # Append the audited item

            if not audited_items: # Check if the list is empty (no ads found)
                print("No items found in the dataset.")
                return [], "No ads found in the library for this page." # Return empty list and specific message
            else:
                print(f"Collected and audited {len(audited_items)} item(s).")
                return audited_items, None # Return collected and audited items and no error
        else:
            print("Run finished, but no defaultDatasetId found.")
            # Consider if this case should report an error or just no items
            return [], "Run finished, but could not find results dataset."

    except Exception as e:
        print(f"An error occurred during Apify run: {e}") # Keep print for logging
        # Extract specific Apify error details if possible
        error_details = str(e)
        if "Monthly usage hard limit exceeded" in error_details:
             error_msg = "Apify Error: Monthly usage hard limit exceeded. Please check your Apify plan."
        elif "403" in error_details:
             error_msg = "Apify Error: Forbidden (403). Check API token validity and permissions."
        else:
            error_msg = f"An error occurred during Apify run: {e}"

        print("Please check:")
        print("1. If the API token is correct and has permissions.")
        print("2. If the Actor ID 'JJghSZmShuco4j9gJ' exists and you have access.")
        print("3. Your Apify account usage limits.")
        print("4. The input format required by the specific Actor.")
        return [], error_msg # Return empty list and specific error message

if __name__ == "__main__":
    test_url = "https://www.facebook.com/theofficialoodie" # Example page
    results, error = meta_ads_audit(test_url)
    if error:
        print(f"Audit failed: {error}")
    elif results:
        print(f"Audit successful. Results ({len(results)} ads max):")
        # This will now print up to 5 items
        for item in results:
            print(json.dumps(item, indent=2)) # Pretty print the JSON
    else:
        print("Audit completed with no results and no error message.")

# --- New Functions for Audit Checks ---

def analyze_ad(ad_item):
    """
    Analyzes a single ad item for errors, improvements, and fatigue.

    Args:
        ad_item (dict): The dictionary representing a single ad from Apify.

    Returns:
        dict: A dictionary containing audit findings.
    """
    findings = {
        "errors": [],
        "improvements": [],
        "fatigue_status": "unknown"
    }

    # Placeholder for error and improvement analysis
    findings["errors"].extend(analyze_ad_for_errors(ad_item))
    findings["improvements"].extend(analyze_ad_for_improvements(ad_item))

    # Placeholder for ad fatigue check
    findings["fatigue_status"] = check_ad_fatigue(ad_item)

    return findings

def analyze_ad_for_errors(ad_item):
    """
    Analyzes a single ad item for potential errors and deficiencies in copy.

    Args:
        ad_item (dict): The dictionary representing a single ad from Apify.

    Returns:
        list: A list of identified errors and deficiencies.
    """
    errors = []
    ad_text = ad_item.get("text") or ad_item.get("copy")

    # Check for missing or very short ad text/copy
    if not ad_text or len(ad_text.strip()) < 30: # Increased arbitrary threshold for short text
        errors.append("Deficiency: Ad text is missing or too short. Aim for compelling copy that clearly communicates the value proposition.")
    elif len(ad_text.strip()) > 280: # Arbitrary threshold for long text (consider platform limits)
         errors.append("Deficiency: Ad text may be too long. Consider being more concise for better readability, especially on mobile.")

    # Check for generic or weak call to action within the text (in addition to the CTA button)
    cta_in_text = False
    if ad_text:
        # Simple check for common CTA phrases in the text
        common_cta_phrases = ["click here", "learn more", "shop now", "sign up", "download", "get started"]
        if any(phrase in ad_text.lower() for phrase in common_cta_phrases):
            cta_in_text = True

    if not cta_in_text and not ad_item.get("callToAction"):
         errors.append("Deficiency: Missing call to action in both text and button. A clear CTA is essential to guide user action.")
    elif not cta_in_text and ad_item.get("callToAction"):
         errors.append("Deficiency: Missing call to action within the ad text. Reinforce the CTA from the button in your copy.")
    elif cta_in_text and ad_item.get("callToAction"):
        # Check for generic call to action button
        cta_button = ad_item.get("callToAction", "").lower()
        generic_ctas = ["learn more", "shop now", "sign up", "download"] # Example generic CTAs
        if cta_button in generic_ctas:
             errors.append(f"Deficiency: Generic call to action button '{ad_item.get('callToAction')}'. Consider a more specific and benefit-driven CTA button.")


    # Placeholder for checking headline and description (assuming keys like 'headline', 'description' exist)
    headline = ad_item.get("headline")
    description = ad_item.get("description")

    if not headline or len(headline.strip()) < 10:
        errors.append("Deficiency: Missing or very short headline. A strong headline grabs attention and should be concise and impactful.")
    if not description or len(description.strip()) < 20:
        errors.append("Deficiency: Missing or very short description. The description provides additional context and encourages clicks. Make it compelling.")

    # Check for presence of benefit-driven language (simple keyword check)
    if ad_text and not any(keyword in ad_text.lower() for keyword in ["you", "your", "benefit", "save", "get", "improve"]):
         errors.append("Deficiency: Ad copy may not be benefit-driven. Focus on what the user gains, not just features.")

    # Check for clarity and jargon (placeholder - requires more advanced NLP)
    # errors.append("Potential Deficiency: Ad copy may contain jargon or be unclear. Ensure it's easily understood by your target audience.")

    # Check for missing creative (assuming 'creative' or similar key exists)
    if not ad_item.get("creative"):
        errors.append("Deficiency: Missing ad creative. A strong visual is crucial for engagement.")

    # Placeholder for checking link validity (cannot actually check without another tool)
    if not ad_item.get("link"):
         errors.append("Deficiency: Missing destination URL. Users need a place to go after clicking.")
    # else:
    #      # This would require a browser or HTTP request tool to check if the link is valid/loads
    #      errors.append("Potential Deficiency: Destination URL may be broken or slow loading. Verify the link.")


    return errors

def analyze_ad_for_improvements(ad_item):
    """
    Analyzes a single ad item for potential improvements and opportunities in copy and overall ad.

    Args:
        ad_item (dict): The dictionary representing a single ad from Apify.

    Returns:
        list: A list of suggested improvements and opportunities.
    """
    improvements = []
    ad_text = ad_item.get("text") or ad_item.get("copy")

    # Suggest A/B testing
    improvements.append("Opportunity: Implement A/B testing for different ad creatives, headlines, copy, and calls to action to optimize performance.")

    # Suggest refining targeting (placeholder as targeting data is not available)
    improvements.append("Opportunity: Refine audience targeting to ensure the ad is reaching the most relevant potential customers.")

    # Suggest improving ad creative based on best practices (placeholder)
    improvements.append("Improvement: Ensure ad creative is high-quality, visually appealing, and relevant to the target audience and offer. Consider using video or carousel formats.")

    # Suggest optimizing landing page experience (placeholder as landing page data is not available)
    improvements.append("Improvement: Optimize the landing page experience to be consistent with the ad message and facilitate conversions.")

    # Suggest leveraging social proof if available (placeholder)
    # if ad_item.get("socialProof"): # Assuming a key for likes/shares/comments exists
    #      improvements.append("Improvement: Highlight social proof (likes, shares, comments) in the ad if available and positive.")
    # else:
    #      improvements.append("Opportunity: Encourage engagement to build social proof.")

    # Suggest adding urgency or scarcity
    if ad_text and "limited time" not in ad_text.lower() and "expires" not in ad_text.lower() and "now" not in ad_text.lower():
         improvements.append("Improvement: Consider adding elements of urgency or scarcity to encourage immediate action.")

    # Suggest using stronger emotional triggers
    improvements.append("Improvement: Review ad copy to ensure it uses strong emotional triggers and speaks directly to the audience's pain points or desires.")

    # Suggest using storytelling in copy
    improvements.append("Improvement: Consider using storytelling techniques in your ad copy to connect with your audience on a deeper level.")

    # Suggest using questions to engage the audience
    if ad_text and "?" not in ad_text:
        improvements.append("Improvement: Incorporate questions in your ad copy to engage the audience and encourage them to think or respond.")

    # Suggest highlighting unique selling proposition (USP)
    improvements.append("Improvement: Clearly articulate your unique selling proposition (USP) in the ad copy to differentiate yourself from competitors.")

    # Suggest using power words
    improvements.append("Improvement: Use power words in your ad copy to evoke emotion and drive action.")


    return improvements

def check_ad_fatigue(ad_item):
    """
    Checks a single ad item for potential ad fatigue based on launch date.

    Args:
        ad_item (dict): The dictionary representing a single ad from Apify.

    Returns:
        str: Fatigue status ("unknown", "likely fatiguing", "not likely fatiguing").
    """
    # Implement actual ad fatigue check based on launch date
    # Assuming 'firstSeen' or 'startDate' key contains the launch date value
    launch_date_value = ad_item.get("firstSeen") or ad_item.get("startDate")

    if launch_date_value is not None:
        # Check if the value is a string before attempting string operations
        if isinstance(launch_date_value, str):
            try:
                # Attempt to parse the date string. Adjust format if necessary based on actual data.
                # Common formats: "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%fZ" (ISO 8601)
                # A more robust solution might try multiple formats or use a library like dateutil.
                # For simplicity, assuming ISO 8601 or similar parseable format.
                launch_date = datetime.fromisoformat(launch_date_value.replace('Z', '+00:00')) # Handle 'Z' for UTC

                fatigue_threshold_days = 90 # Example threshold: 90 days
                if datetime.now() - launch_date > timedelta(days=fatigue_threshold_days):
                    return "likely fatiguing"
                else:
                    return "not likely fatiguing"
            except ValueError:
                # If parsing fails, the date format might be different.
                print(f"Warning: Could not parse launch date string: {launch_date_value}. Please check the date format.")
                return "unknown (date parse error)"
        else:
            # Handle cases where the value is not a string (e.g., integer)
            print(f"Warning: Launch date value is not a string: {launch_date_value} (type: {type(launch_date_value).__name__}).")
            return "unknown (invalid date format)"
    else:
        return "unknown (launch date not available)" # Launch date not available
