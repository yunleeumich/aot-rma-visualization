import pandas as pd

# 📌 Load CSV and Convert to TXT Format
def convert_csv_to_txt(input_csv, output_txt):
    """Convert CSV file to formatted TXT file."""
    df = pd.read_csv(input_csv, encoding="ISO-8859-1")  # Ensure all values are read as strings
    df.fillna("", inplace=True)  # Replace NaN with empty strings

    with open(output_txt, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            unit_sn = row.get("Unit SN", "")
            delta_pn = row.get("Delta PN", "")
            problem_analysis = row.get("Problem Analysis", "")
            failure_category = row.get("Failure Category", "")
            failure_description = row.get("Failure_Description_Translated", "")

            # 📌 Format the output text
            formatted_text = (
                f'"Unit SN" "{unit_sn}" belongs to "Delta PN" "{delta_pn}" '
                f'has "Problem Analysis:" "{problem_analysis}" and "Failure Category:" "{failure_category}" '
                f'with "{failure_description}" as "Failure Description"\n'
            )
            f.write(formatted_text)

# 📌 Example Usage
input_csv = "df_failure_translated-20250218 - copy.csv"  # Replace with your actual CSV file name
output_txt = "output.txt"

convert_csv_to_txt(input_csv, output_txt)

print(f"✅ Conversion complete! The formatted data is saved in {output_txt}.")
