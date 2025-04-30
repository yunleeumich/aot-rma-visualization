import re
from collections import Counter
from io import StringIO
import pandas as pd

def csv_to_text(file):
    """Convert CSV content into a structured text format for retrieval."""
    df = pd.read_csv(file, encoding="ISO-8859-1")
    
    # Convert column names and first few rows into structured text
    column_names = " | ".join(df.columns)
    data_preview = df.head(5).to_string(index=False)  # Ensure proper row formatting

    return f"Columns: {column_names}\n\nSample Data:\n{data_preview}"

def format_csv_text(raw_text):
    """Formats raw retrieved CSV text into a structured table format."""
    try:
        df = pd.read_csv(StringIO(raw_text))
        
        # Drop unnecessary empty columns if they exist
        df.dropna(how='all', axis=1, inplace=True) 
        
        # Convert to clean table format (display first 5 rows)
        return df.head(5).to_string(index=False)
    
    except Exception as e:
        return ""

def extract_failure_descriptions(text, keyword):
    """Extract relevant failure descriptions from text based on user input."""
    if not keyword.strip():
        return "⚠️ Please enter a keyword to search."

    pattern = rf'(?<=\b{re.escape(keyword)}\b).*'  # Look for keyword occurrences
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    return "\n".join(matches) if matches else "❌ No relevant failures found!"

def extract_dtc_codes(text, keyword, dtc_meanings=None):
    """Extract DTC codes from the section of text that contains the keyword."""
    if not keyword.strip():
        return "⚠️ Please enter a keyword to search."

    # Find all sections that contain the keyword
    sections = re.findall(rf'(?<=\b{re.escape(keyword)}\b).*', text, flags=re.IGNORECASE)

    if not sections:
        return f"❌ No DTC codes found in sections related to '{keyword}'!"

    # Combine all matched sections
    filtered_text = " ".join(sections)

    # Extract DTC codes **only from the matched sections**
    dtc_pattern = r'0x[0-9A-Fa-f]{6}'  # Matches "0x" followed by exactly 6 hex digits
    dtcs = re.findall(dtc_pattern, filtered_text)  # Find all occurrences

    # Count occurrences of each DTC code
    dtc_counts = Counter(dtcs)

    if not dtc_counts:
        return f"❌ No DTC codes found in sections related to '{keyword}'!"

    # Sort DTCs by most occurrences
    sorted_dtcs = sorted(dtc_counts.items(), key=lambda x: x[1], reverse=True)

    # Build output with meanings
    output = [f"📌 **DTC Codes for: '{keyword}'**"]
    for dtc, count in sorted_dtcs:
        meaning = dtc_meanings.get(dtc, "❓ Unknown DTC - No description found") if dtc_meanings else "❓ Unknown DTC - No description found"
        output.append(f"🔹 {dtc}: **{count} times** → {meaning}")

    return "\n".join(output)

def extract_charging_issues(text, keyword):
    """Identify key charging failures (AC vs. DC) only in sections related to the keyword."""
    if not keyword.strip():
        return "⚠️ Please enter a keyword to search."

    # Find all sections that contain the keyword
    sections = re.findall(rf'(?<=\b{re.escape(keyword)}\b).*', text, flags=re.IGNORECASE)

    if not sections:
        return f"❌ No charging issues found in sections related to '{keyword}'!"

    # Combine matched sections
    filtered_text = " ".join(sections)

    # **Updated Regex to Match AC/DC Charging Issues Only in Keyword Sections**
    ac_issue_pattern = r'AC charging.*?(fails|not successful|issue|error|cannot charge)'
    dc_success_pattern = r'DC charging.*?(works fine|no issues|successful)'

    ac_issues = re.findall(ac_issue_pattern, filtered_text, flags=re.IGNORECASE)
    dc_successes = re.findall(dc_success_pattern, filtered_text, flags=re.IGNORECASE)

    ac_summary = f"🔴 **AC Charging Issues for '{keyword}':**\n" + "\n".join(ac_issues) if ac_issues else f"✅ No AC charging issues detected in '{keyword}'."
    dc_summary = f"🟢 **DC Charging Status for '{keyword}':**\n" + "\n".join(dc_successes) if dc_successes else f"⚠️ No DC charging details found in '{keyword}'."

    return ac_summary + "\n\n" + dc_summary

def extract_past_repairs(text, keyword):
    """Find past repair actions linked to correct Unit SNs & Delta PNs based on keyword, ensuring correct section matching."""
    
    if not keyword.strip():
        return "⚠️ Please enter a keyword to search."

    # **Regex Pattern to Capture Unit SN, Delta PN, and Related Section Together**
    unit_block_pattern = re.compile(
        r'Unit SN[: ]*(?P<unit_sn>[0-9A-Za-z-]+).*?'  # Capture Unit SN
        r'Delta PN[: ]*(?P<delta_pn>[0-9A-Za-z-]+)?.*?'  # Capture Delta PN (optional)
        r'(?P<section>.*?)(?=\nUnit SN|\Z)',  # Capture full section until next Unit SN or end of text
        re.DOTALL | re.IGNORECASE
    )

    # **Repair Patterns**
    battery_replacement_pattern = re.compile(r'12V battery.*?(replaced|changed|new battery installed)', re.IGNORECASE)
    ccu_replacement_pattern = re.compile(r'CCU.*?(replacement planned|replaced|updated)', re.IGNORECASE)

    # **Find all sections that contain a Unit SN & Delta PN**
    matches = unit_block_pattern.finditer(text)
    output = []
    seen_units = set()

    for match in matches:
        unit_sn = match.group("unit_sn").strip()
        delta_pn = match.group("delta_pn").strip() if match.group("delta_pn") else "Unknown"
        section_text = match.group("section")

        # **Ensure extraction only occurs within the relevant section**
        if keyword.lower() not in section_text.lower():
            continue  # Skip sections that don't contain the keyword

        # **Find repairs within this section**
        battery_repairs = battery_replacement_pattern.findall(section_text)
        ccu_repairs = ccu_replacement_pattern.findall(section_text)

        # **Ensure each Unit SN is processed only once**
        if (unit_sn, delta_pn) in seen_units:
            continue
        seen_units.add((unit_sn, delta_pn))

        # **Build Output (Only show units that have actual repairs)**
        repair_details = []
        if battery_repairs:
            repair_details.append(f"🔋 {' | '.join(battery_repairs)}")
        if ccu_repairs:
            repair_details.append(f"🖥️ {' | '.join(ccu_repairs)}")

        if repair_details:
            output.append(
                f"🆔 **Unit SN:** {unit_sn}\n"
                f"🔹 **Delta PN:** {delta_pn}\n"
                f"{' '.join(repair_details)}"
            )

    return "\n\n".join(output) if output else f"✅ No relevant repairs found for '{keyword}'."

def extract_key_failure_events(text, keyword):
    """Extract key failure events related to the specified keyword, ensuring proper section matching."""
    
    if not keyword.strip():
        return "⚠️ Please enter a keyword to search."

    # **Regex Pattern to Extract Relevant Sections**
    section_pattern = re.compile(
        rf'Unit SN[: ]*(?P<unit_sn>[0-9A-Za-z-]+).*?'  # Capture Unit SN
        rf'Delta PN[: ]*(?P<delta_pn>[0-9A-Za-z-]+)?.*?'  # Capture Delta PN (optional)
        rf'(?P<section>.*?)(?=\nUnit SN|\Z)',  # Capture full section until next Unit SN or end of text
        re.DOTALL | re.IGNORECASE
    )

    # **Key Failure Event Patterns**
    failure_patterns = [
        r'(battery was.*?discharged)',
        r'(charging process cannot be started)',
        r'(isolation error after replacement)',
        r'(error persists despite repair)',
        r'(overcurrent shut down)',
    ]

    # **Find all sections matching the keyword**
    matches = section_pattern.finditer(text)
    output = []
    seen_units = set()

    for match in matches:
        unit_sn = match.group("unit_sn").strip()
        delta_pn = match.group("delta_pn").strip() if match.group("delta_pn") else "Unknown"
        section_text = match.group("section")

        # **Ensure extraction only occurs within the relevant section**
        if keyword.lower() not in section_text.lower():
            continue  # Skip sections that don't contain the keyword

        # **Extract failure events within this section**
        failure_events = []
        for pattern in failure_patterns:
            failure_events.extend(re.findall(pattern, section_text, flags=re.IGNORECASE))

        # **Ensure each Unit SN is processed only once**
        if (unit_sn, delta_pn, tuple(failure_events)) in seen_units:
            continue
        seen_units.add((unit_sn, delta_pn, tuple(failure_events)))

        # **Build Output (Only show units with failures)**
        if failure_events:
            output.append(
                f"🆔 **Unit SN:** {unit_sn}\n"
                f"🔹 **Delta PN:** {delta_pn}\n"
                f"🔍 {' | '.join(set(failure_events))}"  # Remove duplicates
            )

    return "\n\n".join(output) if output else f"✅ No key failure events found for '{keyword}'." 
