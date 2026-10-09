import os
import json
import re
import streamlit as st
import pandas as pd
from groq import Groq

st.set_page_config(page_title="PO Comparison Tool", layout="wide")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def clean_val(val):
    """Normalize cell values to clean strings, handling nulls."""
    if pd.isna(val) or val is None:
        return ""
    val_str = str(val).strip()
    return "" if val_str.lower() in ["nan", "none", "null"] else val_str


def parse_llm_json(response_text: str):
    """Safely extracts JSON list or dictionary from LLM markdown output."""
    cleaned = response_text.strip()
    fence_match = re.search(r'```(?:json)?\s*([\{\[].*?[\}\]])\s*```', cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start_obj = cleaned.find('{')
        end_obj = cleaned.rfind('}')
        start_arr = cleaned.find('[')
        end_arr = cleaned.rfind(']')
        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            cleaned = cleaned[start_obj:end_obj + 1]
        elif start_arr != -1:
            cleaned = cleaned[start_arr:end_arr + 1]

    data = json.loads(cleaned)
    if isinstance(data, list):
        res = {}
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                k = str(item.get("id", idx))
                res[k] = item
            else:
                res[str(idx)] = item
        return res
    return data


def batch_llm_semantic_compare(client: Groq, model: str, records: list, batch_size: int = 200) -> dict:
    """
    Sends batches of complete PO JSON objects to Groq for semantic evaluation.
    Evaluates Style (ignoring -STAFF), Size (ignoring cm), Qty (int vs decimal),
    and KT/Colour semantic extraction and equivalence.
    """
    results_map = {}
    if not records:
        return results_map

    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        payload = [
            {
                "id": idx,
                "po_number": item["po_number"],
                "file_a": {
                    "style": item["style_a"],
                    "size": item["size_a"],
                    "qty": item["qty_a"],
                    "kt_col": item["kt_col_a"]
                },
                "file_b": {
                    "style": item["style_b"],
                    "size": item["size_b"],
                    "qty": item["qty_b"],
                    "colour": item["colour_b"],
                    "item_name": item["item_name_b"]
                }
            }
            for idx, item in enumerate(chunk)
        ]

        prompt = f"""You are an automated jewelry purchase order verification system.
Compare the records below between File A and File B and evaluate semantic equivalence for each column:

Evaluation Rules:
1. style_match: Compare file_a.style with file_b.style.
   - Ignore trailing '-STAFF' suffixes (e.g., 'L-ER-61228-200' matches 'L-ER-61228-200-STAFF').
   - Return true if the core style numbers match, otherwise false.

2. size_match: Compare file_a.size with file_b.size.
   - Ignore unit suffix 'cm' (e.g., '45' matches '45cm', '18.5' matches '18.5cm').
   - Both empty/blank values match each other.
   - Return true if sizes match, otherwise false.

3. qty_match: Compare file_a.qty with file_b.qty.
   - Values are equivalent if they represent the same number (e.g., 1 or 1.0 equals 1.00).
   - Return true if quantities match, otherwise false.

4. kt_col_match: Compare Karat and Metal Color.
   - Extract Karat (e.g., 9KT, 10KT, 10CT, 14KT, 14CT, 18KT, SIL, PT) and metal color (YELLOW/Y, WHITE/W, ROSE/PINK/P, YEL/WHITE/YW) from file_b.colour and file_b.item_name.
   - Compare them against file_a.kt_col (formatted as 'Karat - Color', e.g., '10K - Y', '14K - W', '10K - P', '14K - YW', 'S925 - -', 'PT95 - -').
   - Return true if BOTH Karat and Color match, otherwise false.

5. overall_match:
   - Must be true if and only if style_match, size_match, qty_match, and kt_col_match are ALL true. Otherwise false.

Records to evaluate:
{json.dumps(payload, indent=2)}

Respond ONLY with a valid JSON object mapping each integer ID (as a string) to an object containing boolean values:
{{
  "0": {{
    "style_match": true,
    "size_match": true,
    "qty_match": true,
    "kt_col_match": true,
    "overall_match": true
  }}
}}
No commentary, text, or markdown prose outside the JSON."""

        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a JSON-only jewelry verification engine. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.0,
                max_tokens=2500,
            )
            raw_content = response.choices[0].message.content.strip()
            parsed = parse_llm_json(raw_content)

            for idx, item in enumerate(chunk):
                item_res = parsed.get(str(idx), parsed.get(idx, {}))
                results_map[item["po_number"]] = {
                    "style_match": bool(item_res.get("style_match", False)),
                    "size_match": bool(item_res.get("size_match", False)),
                    "qty_match": bool(item_res.get("qty_match", False)),
                    "kt_col_match": bool(item_res.get("kt_col_match", False)),
                    "overall_match": bool(item_res.get("overall_match", False))
                }
        except Exception as e:
            st.warning(f"Batch evaluation warning: {e}")
            for item in chunk:
                results_map[item["po_number"]] = {
                    "style_match": False,
                    "size_match": False,
                    "qty_match": False,
                    "kt_col_match": False,
                    "overall_match": False
                }

    return results_map


# ---------------------------------------------------------
# Streamlit Frontend
# ---------------------------------------------------------

st.title("Jewelry PO Semantic Verification Dashboard")
st.markdown("Upload **File A (MHJ)** and **File B (Dynamics Export)** to evaluate full column matching via LLM.")

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload File A (MHJ.xlsx)", type=["xlsx", "xls"])
with col2:
    file_b = st.file_uploader("Upload File B (Dynamics Export .xlsx)", type=["xlsx", "xls"])

if file_a and file_b:
    if not api_key:
        st.error("Please provide a Groq API key in the sidebar or Streamlit secrets to proceed.")
        st.stop()

    groq_client = Groq(api_key=api_key)

    if st.button("Run Semantic Comparison", type="primary"):
        with st.spinner("Extracting data and sending JSON to LLM..."):
            df_a = pd.read_excel(file_a, sheet_name=0)
            df_b = pd.read_excel(file_b, sheet_name=0)

            # Map Columns by Zero-based Excel Positions
            col_a_po = df_a.columns[8]        # Column I (Sub PO #)
            col_a_style = df_a.columns[4]     # Column E (Style No)
            col_a_qty = df_a.columns[6]       # Column G (Order Qty)
            col_a_size = df_a.columns[9]      # Column J (Size)
            col_a_kt_col = df_a.columns[29]   # Column AD (KT - Col)

            col_b_po = df_b.columns[0]         # Column A (Purchase order)
            col_b_style = df_b.columns[11]     # Column L (Style code)
            col_b_size = df_b.columns[13]      # Column N (Size)
            col_b_colour = df_b.columns[14]    # Column O (Colour)
            col_b_item_name = df_b.columns[15] # Column P (Item name)
            col_b_qty = df_b.columns[16]       # Column Q (Quantity)

            # Filter valid master PO rows from File A
            df_a_filtered = df_a[df_a[col_a_po].notna()].copy()
            df_a_filtered["cleaned_po"] = df_a_filtered[col_a_po].astype(str).str.strip()

            # Group File B for lookup
            df_b["cleaned_po"] = df_b[col_b_po].astype(str).str.strip()
            df_b_grouped = df_b.groupby("cleaned_po")

            matched_records = []
            unmatched_pos = []

            for _, row_a in df_a_filtered.iterrows():
                po_num = row_a["cleaned_po"]
                style_a = clean_val(row_a[col_a_style])
                size_a = clean_val(row_a[col_a_size])
                qty_a = clean_val(row_a[col_a_qty])
                kt_col_a = clean_val(row_a[col_a_kt_col])

                if po_num not in df_b_grouped.groups:
                    unmatched_pos.append({
                        "PO Number": po_num,
                        "Found in File B": False,
                        "Style Match": False,
                        "Size Match": False,
                        "Qty Match": False,
                        "KT/Color Match": False,
                        "Overall Match": False,
                        "File A (Style | Size | Qty | KT-Col)": f"{style_a} | {size_a} | {qty_a} | {kt_col_a}",
                        "File B Values": "NOT FOUND"
                    })
                    continue

                row_b = df_b_grouped.get_group(po_num).iloc[0]
                style_b = clean_val(row_b[col_b_style])
                size_b = clean_val(row_b[col_b_size])
                qty_b = clean_val(row_b[col_b_qty])
                colour_b = clean_val(row_b[col_b_colour])
                item_name_b = clean_val(row_b[col_b_item_name])

                matched_records.append({
                    "po_number": po_num,
                    "style_a": style_a,
                    "size_a": size_a,
                    "qty_a": qty_a,
                    "kt_col_a": kt_col_a,
                    "style_b": style_b,
                    "size_b": size_b,
                    "qty_b": qty_b,
                    "colour_b": colour_b,
                    "item_name_b": item_name_b
                })

            st.info(f"Sending {len(matched_records)} PO records in JSON batches to Groq for semantic evaluation...")

            # Run LLM comparison on all matched records
            llm_evaluations = batch_llm_semantic_compare(
                client=groq_client,
                model="openai/gpt-oss-20b",
                records=matched_records
            )

            # Assemble final table
            final_rows = []
            for item in matched_records:
                po = item["po_number"]
                res = llm_evaluations.get(po, {
                    "style_match": False,
                    "size_match": False,
                    "qty_match": False,
                    "kt_col_match": False,
                    "overall_match": False
                })

                final_rows.append({
                    "PO Number": po,
                    "Found in File B": True,
                    "Style Match": res["style_match"],
                    "Size Match": res["size_match"],
                    "Qty Match": res["qty_match"],
                    "KT/Color Match": res["kt_col_match"],
                    "Overall Match": res["overall_match"],
                    "File A (Style | Size | Qty | KT-Col)": f"{item['style_a']} | {item['size_a']} | {item['qty_a']} | {item['kt_col_a']}",
                    "File B Values": f"{item['style_b']} | {item['size_b']} | {item['qty_b']} | {item['colour_b']} | {item['item_name_b']}"
                })

            final_rows.extend(unmatched_pos)
            result_df = pd.DataFrame(final_rows)

            # Metrics
            total_pos = len(result_df)
            matched_pos = int((result_df["Overall Match"] == True).sum())
            mismatched_pos = total_pos - matched_pos

            m1, m2, m3 = st.columns(3)
            m1.metric("Total POs Evaluated", total_pos)
            m2.metric("All Columns Matched (TRUE)", matched_pos)
            m3.metric("Mismatched / Not Found (FALSE)", mismatched_pos)

            # Summary Table
            st.subheader("PO Matching Results")
            st.dataframe(
                result_df[["PO Number", "Overall Match", "Style Match", "Size Match", "Qty Match", "KT/Color Match", "Found in File B"]],
                use_container_width=True
            )

            with st.expander("Inspect Raw Field Comparisons"):
                st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Results CSV",
                data=csv_data,
                file_name="po_comparison_results.csv",
                mime="text/csv"
            )