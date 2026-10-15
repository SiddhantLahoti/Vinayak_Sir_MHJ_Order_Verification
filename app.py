import os
import json
import re
import streamlit as st
import pandas as pd
from groq import Groq

st.set_page_config(page_title="PO Comparison Tool", layout="wide")

# ---------------------------------------------------------
# Helper Functions & Normalization
# ---------------------------------------------------------

def to_int_safe(val):
    """Safely converts float/string/int representations to an int or None."""
    try:
        if pd.isna(val) or val is None or str(val).strip() == "":
            return None
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None

def compare_as_int(val_a, val_b):
    """Compare two values as integers."""
    ia = to_int_safe(val_a)
    ib = to_int_safe(val_b)
    if ia is None or ib is None:
        return False
    return ia == ib

def format_display(val):
    s = clean_str(val)
    return s if s else "—"

def clean_str(val):
    """Normalize strings by stripping whitespace, carriage returns, and tabs."""
    if pd.isna(val) or val is None:
        return ""
    val_str = str(val).strip()
    return "" if val_str.lower() in ["nan", "none", "null"] else val_str


def clean_style(val):
    """Normalize style code and remove trailing -STAFF."""
    s = clean_str(val)
    s = re.sub(r'[\s-]*staff\s*$', '', s, flags=re.IGNORECASE)
    return s.strip().lower()


def compare_styles(val_a, val_b):
    """Compare style numbers ignoring trailing -STAFF, whitespace, and case."""
    s_a = clean_style(val_a)
    s_b = clean_style(val_b)
    return (s_a == s_b) and (s_a != "")


def compare_generic(val_a, val_b):
    """Compare column values ignoring whitespace and case sensitivity."""
    s_a = clean_str(val_a).lower()
    s_b = clean_str(val_b).lower()
    return (s_a == s_b) and (s_a != "")


def clean_size(val):
    """Normalize sizes by stripping 'cm' units and surrounding whitespace."""
    s = clean_str(val)
    s = re.sub(r'\s*cm\b', '', s, flags=re.IGNORECASE)
    return s.strip().lower()


def compare_sizes(val_a, val_b):
    """Compare ring/jewelry sizes ignoring 'cm' unit annotations."""
    s_a = clean_size(val_a)
    s_b = clean_size(val_b)
    return s_a == s_b


def compare_quantities(val_a, val_b):
    """Compare integer order qty with decimal purchase order qty."""
    try:
        if pd.isna(val_a) or pd.isna(val_b):
            return False
        return abs(float(val_a) - float(val_b)) < 0.001
    except (ValueError, TypeError):
        return False


def parse_llm_json(response_text: str) -> dict:
    """Safely extracts and parses JSON dictionary or array from LLM output."""
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
        for i, item in enumerate(data):
            if isinstance(item, dict):
                k = str(item.get("id", i))
                v = item.get("match", item.get("result", item.get("matched", False)))
                res[k] = v
            else:
                res[str(i)] = item
        return res
    return data


def batch_verify_kt_and_color_llm(client: Groq, model: str, unique_triplets: list, batch_size: int = 15) -> dict:
    """Batches unique combinations to verify Karat/Metal and Color using Groq LLM."""
    results_map = {}
    if not unique_triplets:
        return results_map

    for i in range(0, len(unique_triplets), batch_size):
        chunk = unique_triplets[i:i + batch_size]
        payload = [
            {
                "id": str(idx),
                "file_a_spec": t[0],
                "file_b_color": t[1],
                "file_b_item_name": t[2]
            }
            for idx, t in enumerate(chunk)
        ]

        prompt = f"""You are an expert jewelry specification auditor.
Compare the metal specification in 'file_a_spec' with the item details in 'file_b_color' and 'file_b_item_name'.

CRITICAL INSTRUCTIONS:
1. METAL / PURITY IDENTIFICATION & TWO-TONE COMBINATIONS:
   - Identify the metal alloy in 'file_b_item_name' (e.g., 9KT, 10KT, 14KT, 18KT, or SIL / SILVER / S925, or PT / PLATINUM / PT95).
   - TWO-TONE COMBINATIONS: Codes like '10K92' or '10K925' in 'file_a_spec' represent two metals: 10KT Gold + 925 Sterling Silver. This MATCHES combinations in File B like 'SIL/10KT', '10KT/SIL', 'SIL/10K', or '10KT & SIL'.
   - SILVER EQUIVALENCE: 'S925', '925', 'SIL', and 'SILVER' all refer to the exact same metal.
   - SINGLE METALS: Match '14K' to '14KT', '10K' to '10KT', '9K' to '9KT', 'S925' to 'SIL'/'SILVER', 'PT95' to 'PT'/'PLATINUM'.
   - DO NOT confuse diamond/gemstone carat weight with metal purity! Numbers followed by 'TDW', 'CTW', or diamond weights (like '4.00TDW', '.15TDW', '1.00CT') are stone weights, NOT the metal karat.

2. COLOR & TWO-TONE IDENTIFICATION:
   - File A encodes color after the hyphen (e.g., '- Y' = Yellow, '- W' = White, '- P' = Rose/Pink, '- YW' = Yellow/White, '- -' = unplated/natural).
   - File B provides color in 'file_b_color' (YELLOW, WHITE, ROSE, SILVER, SIL/YELLOW, YELLOW/SIL, etc.).
   - SILVER IS WHITE: Silver / S925 is naturally white metal. Therefore, 'S925 - W' perfectly matches 'SILVER' or 'WHITE'.
   - TWO-TONE COLOR: '- YW' stands for Yellow & White. This MATCHES 'SIL/YELLOW', 'YELLOW/SIL', 'YELLOW/WHITE', or 'TWO TONE'.
   - If File A specifies a single color ('Y', 'W', 'P'), both metal AND color must match.
   - If File A specifies unplated ('- -'), only the metal must match.

Data to verify:
{json.dumps(payload, indent=2)}

Return ONLY a flat JSON object mapping string IDs to booleans (true if both metal and color match, false otherwise):
{{"0": true, "1": false}}"""

        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a jewelry QA verifier. Return ONLY a valid JSON object mapping string IDs to true/false booleans. Do not include markdown codeblocks or explanation."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.0,
                max_tokens=1500,
            )
            raw_content = response.choices[0].message.content.strip()
            parsed = parse_llm_json(raw_content)

            if isinstance(parsed, dict) and not any(str(k) in parsed for k in range(len(chunk))):
                for val in parsed.values():
                    if isinstance(val, dict):
                        parsed = val
                        break

            for idx, triplet in enumerate(chunk):
                item_val = parsed.get(str(idx), parsed.get(idx, False))
                if isinstance(item_val, bool):
                    is_match = item_val
                elif isinstance(item_val, str):
                    is_match = item_val.strip().lower() in ["true", "1", "yes"]
                elif isinstance(item_val, dict):
                    is_match = bool(item_val.get("match", item_val.get("result", False)))
                else:
                    is_match = False

                results_map[triplet] = is_match

        except Exception as e:
            st.error(f"Groq API Error on batch {i}: {e}")
            for triplet in chunk:
                results_map[triplet] = False

    return results_map


# ---------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------

st.title("MH Order Verification")
st.markdown("Compare **File A (MHJ)** with **File B (Dynamics Export)**.")

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload File A (MHJ.xlsx)", type=["xlsx", "xls"])
with col2:
    file_b = st.file_uploader("Upload File B (Dynamics Export .xlsx)", type=["xlsx", "xls"])

if file_a and file_b:
    if not api_key:
        st.error("Please provide a Groq API key in the sidebar or Streamlit secrets.")
        st.stop()

    groq_client = Groq(api_key=api_key)

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Processing files..."):
            df_a = pd.read_excel(file_a, sheet_name=0)
            df_b = pd.read_excel(file_b, sheet_name=0)

            # Map Columns by Zero-based Index
            col_a_po = df_a.columns[9]       # Column J (Sub PO #)
            col_a_style = df_a.columns[10]   # Column K (Style No)
            col_a_g = df_a.columns[6]        # Column G (Print only, no comparison)
            col_a_size = df_a.columns[12]    # Column M (Size)
            col_a_l = df_a.columns[11]       # Column L (Suffix / ID)
            col_a_qty = df_a.columns[14]     # Column O (Qty)
            col_a_kt_col = df_a.columns[13]  # Column N (KT - Col)

            col_b_po = df_b.columns[0]         # Column A (Purchase order)
            col_b_style = df_b.columns[10]     # Column K (Style code)
            col_b_l = df_b.columns[11]         # Column L (Suffix / ID)
            col_b_size = df_b.columns[12]      # Column M (Size)
            col_b_colour = df_b.columns[13]    # Column N (Colour)
            col_b_item_name = df_b.columns[14] # Column O (Item name)
            col_b_qty = df_b.columns[15]       # Column P (Quantity)

            # Filter valid PO rows from File A
            df_a_filtered = df_a[df_a[col_a_po].notna()].copy()
            df_a_filtered["cleaned_po"] = df_a_filtered[col_a_po].astype(str).str.strip()

            # Group File B by cleaned PO for rapid lookup
            df_b["cleaned_po"] = df_b[col_b_po].astype(str).str.strip()
            df_b_grouped = df_b.groupby("cleaned_po")

            rows_to_process = []
            unique_triplets_set = set()

            for _, row_a in df_a_filtered.iterrows():
                po_num = row_a["cleaned_po"]
                style_a = row_a[col_a_style]
                col_g_a = row_a[col_a_g]
                col_l_a = row_a[col_a_l]
                size_a = row_a[col_a_size]
                qty_a = row_a[col_a_qty]
                kt_col_a = clean_str(row_a[col_a_kt_col])

                if po_num not in df_b_grouped.groups:
                    rows_to_process.append({
                        "po_num": po_num,
                        "found": False,
                        "style_a": style_a, "col_g_a": col_g_a, "col_l_a": col_l_a,
                        "size_a": size_a, "qty_a": qty_a, "kt_col_a": kt_col_a,
                        "style_b": "", "col_l_b": "", "size_b": "", "qty_b": "",
                        "colour_b": "", "item_name_b": "",
                        "match_style": False, "match_col_l": False,
                        "match_qty": False, "match_size": False,
                        "triplet": None
                    })
                    continue

                row_b = df_b_grouped.get_group(po_num).iloc[0]
                style_b = row_b[col_b_style]
                col_l_b = row_b[col_b_l]
                size_b = row_b[col_b_size]
                qty_b = row_b[col_b_qty]
                colour_b = clean_str(row_b[col_b_colour])
                item_name_b = clean_str(row_b[col_b_item_name])

                triplet = (kt_col_a, colour_b, item_name_b)
                unique_triplets_set.add(triplet)

                rows_to_process.append({
                    "po_num": po_num,
                    "found": True,
                    "style_a": style_a, "col_g_a": col_g_a, "col_l_a": col_l_a,
                    "size_a": size_a, "qty_a": qty_a, "kt_col_a": kt_col_a,
                    "style_b": style_b, "col_l_b": col_l_b, "size_b": size_b, "qty_b": qty_b,
                    "colour_b": colour_b, "item_name_b": item_name_b,
                    "match_style": compare_styles(style_a, style_b),
                    "match_col_l": compare_as_int(col_l_a, col_l_b),
                    "match_qty": compare_as_int(qty_a, qty_b),
                    "match_size": compare_sizes(size_a, size_b),
                    "triplet": triplet
                })

            unique_triplets = list(unique_triplets_set)
            st.info(f"Evaluating {len(rows_to_process)} POs ({len(unique_triplets)} unique specifications sent in batch to Groq)...")

            llm_results_map = batch_verify_kt_and_color_llm(
                client=groq_client,
                model="qwen/qwen3.8-27b",
                unique_triplets=unique_triplets
            )

            final_results = []
            for item in rows_to_process:
                if not item["found"]:
                    final_results.append({
                        "PO Number": item["po_num"],
                        "Status": "⚠️ Not in File B",
                        "Col G (MHJ)": format_display(item['col_g_a']),
                        "Style (A ➔ B)": f"{format_display(item['style_a'])} ➔ NOT FOUND",
                        "Col L (A ➔ B)": f"{format_display(item['col_l_a'])} ➔ NOT FOUND",
                        "Qty (A ➔ B)": f"{format_display(item['qty_a'])} ➔ NOT FOUND",
                        "Size (A ➔ B)": f"{format_display(item['size_a'])} ➔ NOT FOUND",
                        "KT & Color (A ➔ B)": f"{format_display(item['kt_col_a'])} ➔ NOT FOUND",
                        "Style Match": False,
                        "Col L Match": False,
                        "Qty Match": False,
                        "Size Match": False,
                        "KT/Color Match": False,
                        "Overall Match": False,
                    })
                else:
                    match_kt_color = llm_results_map.get(item["triplet"], False)

                    # Clean values for display
                    g_a = format_display(item['col_g_a'])
                    s_a, s_b = format_display(item['style_a']), format_display(item['style_b'])

                    # Format integer values cleanly (eliminates '.0' from the table)
                    l_a_int = to_int_safe(item['col_l_a'])
                    l_b_int = to_int_safe(item['col_l_b'])
                    disp_l_a = str(l_a_int) if l_a_int is not None else format_display(item['col_l_a'])
                    disp_l_b = str(l_b_int) if l_b_int is not None else format_display(item['col_l_b'])

                    q_a_int = to_int_safe(item['qty_a'])
                    q_b_int = to_int_safe(item['qty_b'])
                    disp_q_a = str(q_a_int) if q_a_int is not None else format_display(item['qty_a'])
                    disp_q_b = str(q_b_int) if q_b_int is not None else format_display(item['qty_b'])

                    overall_match = (
                        item["match_style"]
                        and item["match_col_l"]
                        and item["match_size"]
                        and item["match_qty"]
                        and match_kt_color
                    )

                    sz_a, sz_b = format_display(item['size_a']), format_display(item['size_b'])
                    kt_a = format_display(item['kt_col_a'])
                    b_desc = f"{format_display(item['colour_b'])} ({format_display(item['item_name_b'])})"

                    final_results.append({
                        "PO Number": item["po_num"],
                        "Status": "✅ Matched" if overall_match else "❌ Mismatched",
                        "Col G (MHJ)": g_a,
                        "Style (A ➔ B)": f"{'✅' if item['match_style'] else '❌'} {s_a} ➔ {s_b}",
                        "Col L (A ➔ B)": f"{'✅' if item['match_col_l'] else '❌'} {disp_l_a} ➔ {disp_l_b}",
                        "Qty (A ➔ B)": f"{'✅' if item['match_qty'] else '❌'} {disp_q_a} ➔ {disp_q_b}",
                        "Size (A ➔ B)": f"{'✅' if item['match_size'] else '❌'} {sz_a} ➔ {sz_b}",
                        "KT & Color (A ➔ B)": f"{'✅' if match_kt_color else '❌'} {kt_a} ➔ {b_desc}",
                        "Style Match": item["match_style"],
                        "Col L Match": item["match_col_l"],
                        "Qty Match": item["match_qty"],
                        "Size Match": item["match_size"],
                        "KT/Color Match": match_kt_color,
                        "Overall Match": overall_match,
                    })

            result_df = pd.DataFrame(final_results)
            st.session_state["result_df"] = result_df

            # Display Summary Metrics
            total_pos = len(result_df)
            matched_pos = int((result_df["Overall Match"] == True).sum())
            unmatched_pos = total_pos - matched_pos

            m1, m2, m3 = st.columns(3)
            m1.metric("Total POs Evaluated", total_pos)
            m2.metric("All Columns Matched (TRUE)", matched_pos)
            m3.metric("Mismatched / Not Found (FALSE)", unmatched_pos)

    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]

        st.subheader("Verification Breakdown")

        filter_choice = st.radio(
            "Filter View:",
            ["All POs", "Only Matched", "Only Mismatched", "Only Missing (Not in File B)"],
            horizontal=True
        )

        if filter_choice == "Only Matched":
            view_df = result_df[result_df["Overall Match"] == True]
        elif filter_choice == "Only Mismatched":
            view_df = result_df[result_df["Status"].str.contains("Mismatched", na=False)]
        elif filter_choice == "Only Missing (Not in File B)":
            view_df = result_df[result_df["Status"].str.contains("Not in File B", na=False)]
        else:
            view_df = result_df

        st.dataframe(
            view_df[[
                "PO Number", "Status", "Col G (MHJ)", "Style (A ➔ B)",
                "Col L (A ➔ B)", "Qty (A ➔ B)", "Size (A ➔ B)", "KT & Color (A ➔ B)"
            ]],
            hide_index=True,
            width="stretch"
        )