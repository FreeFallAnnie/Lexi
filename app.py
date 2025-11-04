### Code
from oauth2client.service_account import ServiceAccountCredentials
from gtts import gTTS
import gspread
import os
import streamlit as st
import pyttsx3
import base64
import json
import tempfile
import random
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=api_key)

# ---------- AGENT REGISTRY (Live Tab + Logic) ----------
from datetime import datetime
import pandas as pd

def get_item_combos(df, item_id):
    """
    Shared helper: find Combo_1, Combo_2, Combo_3... and their linked IDC1/IDC2/IDC3.
    Returns a list of dicts with both combo text and item ID.
    """
    df.columns = [c.strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if c in ["item id", "id"]), None)
    if not id_col:
        return []

    item_row = df[df[id_col].astype(str) == str(item_id)]
    if item_row.empty:
        return []

    combo_cols = [c for c in df.columns if c.startswith("combo")]
    idc_cols = [c for c in df.columns if c.startswith("idc")]

    combos = []
    for i, combo_col in enumerate(combo_cols, start=1):
        combo_text = str(item_row.iloc[0][combo_col]).strip() if combo_col in item_row.columns else ""
        idc_col = f"idc{i}"
        combo_id = str(item_row.iloc[0][idc_col]).strip() if idc_col in item_row.columns else ""
        if combo_text:
            combos.append({"combo": combo_text, "id": combo_id})

    if combos:
        import random
        random.shuffle(combos)
        combos = combos[:1]

    return combos

def joy_prep(df, item_id):
    """
    For Joy agent:
    Uses PI1/PI2/PI3 columns to find which combinations (Combo_1, Combo_2, Combo_3)
    were positive interactions. Returns both combo text and linked IDC values.
    """
    df.columns = [c.strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if c in ["item id", "id"]), None)
    if not id_col:
        return {"combos": [], "combo_ids": []}

    df[id_col] = df[id_col].astype(str)
    item_id = str(item_id)
    item_rows = df[df[id_col] == item_id]
    if item_rows.empty:
        return {"combos": [], "combo_ids": []}

    row = item_rows.iloc[0]
    combo_cols = [c for c in df.columns if c.startswith("combo")]
    idc_cols = [c for c in df.columns if c.startswith("idc")]
    pi_cols = [c for c in df.columns if c.startswith("pi")]

    combos = []
    combo_ids = []

    # Match PI columns with Combo and IDC columns (by number suffix)
    for i, pi_col in enumerate(pi_cols, start=1):
        pi_val = str(row.get(pi_col, "")).strip().lower()
        combo_col = f"combo_{i}"
        idc_col = f"idc{i}"

        if pi_val in ["true", "yes", "1"]:
            combo_val = str(row.get(combo_col, "")).strip()
            id_val = str(row.get(idc_col, "")).strip()
            if combo_val:
                combos.append(combo_val)
                combo_ids.append(id_val)

    # Fallback: include all combos if none were positively marked
    if not combos:
        for i, combo_col in enumerate(combo_cols, start=1):
            combo_val = str(row.get(combo_col, "")).strip()
            id_val = str(row.get(f"idc{i}", "")).strip()
            if combo_val:
                combos.append(combo_val)
                combo_ids.append(id_val)

    return {"combos": combos, "combo_ids": combo_ids}

def environment_prep(df, item_id):
    combo_data = get_item_combos(df, item_id)
    combos = [c["combo"] for c in combo_data]
    combo_ids = [c["id"] for c in combo_data]

    df.columns = [c.strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if c in ["item id", "id"]), None)
    cost_col = "cost"
    worn_col = "times worn"

    if not all([id_col, cost_col, worn_col]) or id_col not in df.columns:
        return {"combos": combos, "combo_ids": combo_ids, "cost_per_wear": None}

    row = df[df[id_col].astype(str) == str(item_id)]
    if row.empty:
        return {"combos": combos, "combo_ids": combo_ids, "cost_per_wear": None}

    cost = pd.to_numeric(row.iloc[0][cost_col], errors="coerce")
    worn = pd.to_numeric(row.iloc[0][worn_col], errors="coerce")
    cpw = round(cost / worn, 2) if cost and worn else None

    return {"combos": combos, "combo_ids": combo_ids, "cost_per_wear": cpw}

def retire_reuse_prep(df, item_id):
    """
    Retire & Reuse Agent:
    Calculates how many days it's been since the item was last worn,
    comparing 'Last Worn Date' (Wardrobe sheet, DD/MM/YYYY)
    to the latest detection timestamp (from Detections sheet, YYYY-MM-DD).
    """
    combos = get_item_combos(df, item_id)

    # Normalize Wardrobe columns
    df.columns = [c.strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if c in ["item id", "id"]), None)
    last_worn_col = next((c for c in df.columns if "last worn" in c), None)

    if not id_col or not last_worn_col:
        return {"combos": combos, "days_since_last_wear": None}

    # Find selected item in Wardrobe
    row = df[df[id_col].astype(str) == str(item_id)]
    if row.empty:
        return {"combos": combos, "days_since_last_wear": None}

    # Parse Last Worn Date (Australian format)
    last_worn_str = str(row.iloc[0][last_worn_col]).strip()
    last_worn_date = pd.to_datetime(last_worn_str, errors="coerce", dayfirst=True)

    if pd.isna(last_worn_date):
        return {"combos": combos, "days_since_last_wear": None}

    # Load Detections sheet and find latest detection timestamp
    try:
        sheet_detections = gs_client.open("lexi_live").worksheet("detections")
        records_detections = sheet_detections.get_all_records()
        df_detect = pd.DataFrame(records_detections)
        df_detect.columns = [c.strip().lower() for c in df_detect.columns]

        id_detect_col = next((c for c in df_detect.columns if "item id" in c), None)
        ts_col = next((c for c in df_detect.columns if "timestamp" in c), None)

        if id_detect_col and ts_col:
            # Timestamps from Detections are already in ISO format (YYYY-MM-DD)
            df_detect[ts_col] = pd.to_datetime(df_detect[ts_col], errors="coerce")
            latest_timestamp = df_detect.loc[
                df_detect[id_detect_col].astype(str) == str(item_id), ts_col
            ].max()
        else:
            latest_timestamp = datetime.now()

        if pd.isna(latest_timestamp):
            latest_timestamp = datetime.now()

    except Exception:
        latest_timestamp = datetime.now()

    # Compute days since last worn
    try:
        days_since = int((latest_timestamp - last_worn_date).days)
    except Exception:
        days_since = None

    return {
        "combos": combos,
        "days_since_last_wear": days_since,
    }


def my_own_words_prep(item_id):
    """
    My Own Words Agent:
    - Looks at the 'form responses' sheet.
    - Uses the first number in 'Item_IDs' for matching.
    - Collects ALL user comments related to that item.
    """
    try:
        sheet_feedback = gs_client.open("lexi_live").worksheet("form responses")
        records_feedback = sheet_feedback.get_all_records()
        df_fb = pd.DataFrame(records_feedback)

        # Normalize columns
        df_fb.columns = [c.strip().lower() for c in df_fb.columns]

        if not all(col in df_fb.columns for col in ["comment", "item_ids"]):
            return {"user_comments": []}

        # Extract first number from the Item_IDs string
        def extract_first_id(ids_str):
            try:
                ids = [i.strip() for i in str(ids_str).split(",") if i.strip()]
                return ids[0] if ids else None
            except Exception:
                return None

        df_fb["primary_id"] = df_fb["item_ids"].apply(extract_first_id)

        # Match the first ID to the selected item_id
        mask = df_fb["primary_id"].astype(str) == str(item_id)
        relevant_comments = df_fb.loc[mask, "comment"].dropna().tolist()

        return {"user_comments": relevant_comments}

    except Exception as e:
        st.warning(f"My Own Words prep failed: {e}")
        return {"user_comments": []}



AGENT_SPECS = {
    "Joy": {
        "prompt": "What brings me joy?",
        "sheet": "Wardrobe",
        "prep": joy_prep,
        "style_prompt": (
            "Focus on emotional resonance, sensory language, and self-expression. "
            "Use warm, positive adjectives that celebrate how the outfit makes the user feel."
        ),
    },
    "Cost-Per-Wear": {
        "prompt": "What is the Cost Per Wear for this peice?",
        "sheet": "Wardrobe",
        "prep": environment_prep,
        "style_prompt": (
            "Speak like a pragmatic style strategist. Mention value, longevity, and smart wardrobe economics. "
            "Tone: calm, confident, slightly analytical, like a sustainability coach."
        ),
    },
    "Old Favourite": {
        "prompt": "Which old favourites am I picking up again?",
        "sheet": "Wardrobe",
        "prep": retire_reuse_prep,
        "style_prompt": (
            "Encourage rediscovery and sentimentality. Reference nostalgia, memory, and gentle renewal. "
            "Tone: kind, reflective, encouraging a second life for beloved items."
        ),
    },
    "My Own Words": {
        "prompt": "Reflect what the user has previously said about this item in their own words.",
        "sheet": "form responses",
        "prep": None,
        "style_prompt": (
            "Use the user’s tone, mirroring their vocabulary and emotional phrasing. "
            "Prioritize personal reflection over suggestion."
        ),
    },
}



def query_agent(agent_name):
    """
    Fetch agent-specific data from Google Sheets, filter for selected item,
    process it, and generate a GPT suggestion.
    """
    spec = AGENT_SPECS.get(agent_name)
    if not spec:
        st.error(f"Unknown agent: {agent_name}")
        return None

    # Get selected item from session state
    item_id = st.session_state.get("selected_item_id", None)
    if not item_id:
        st.warning("No item selected yet (camera or gallery).")
        return None

    # 1️⃣ Load the correct sheet for this agent
    try:
        sheet = gs_client.open("lexi_live").worksheet(spec["sheet"])
        records = sheet.get_all_records()
        df = pd.DataFrame(records)
    except Exception as e:
        st.error(f"Could not load data from sheet '{spec['sheet']}': {e}")
        return None

    # 2️⃣ Process data for the selected item
    if spec["prep"] is not None:
        processed = spec["prep"](df, item_id)
    else:
        processed = {}

    # Skip combo logic entirely for "My Own Words"
    if agent_name == "My Own Words":
        processed = my_own_words_prep(item_id)
        combos = []  # Not relevant
        combo_ids = []
        extra_info = None
    else:
        # Normal agent combo extraction
        combos = []
        combo_ids = []
        extra_info = None

        if isinstance(processed, dict):
            combos = processed.get("combos", [])
            combo_ids = processed.get("combo_ids", [])
            extra_info = processed.get("cost_per_wear")
        elif isinstance(processed, list):
            combos = processed

        else:
            combos = []
            extra_info = None

        if not combos:
            st.warning(f"No combos found for item {item_id}.")
            return None


    # 3️⃣ Build the prompt dynamically for all agents
    combo_text = "\n".join([f"- {c}" for c in combos])
    agent_prompt = spec["prompt"]

    # Add any specialty context
    if agent_name == "Cost-Per-Wear" and extra_info:
        context_line = f"The cost per wear for this single peice of clothing is approximately ${extra_info:.2f}, which you may like to consider."
    elif agent_name == "Joy":
        context_line = "Focus on what makes these combinations joyful as it has been previously noted that you enjoyed wearing this combination."
    elif agent_name == "Retire & Reuse":
        days = processed.get("days_since_last_wear")
        if days is not None:
            context_line = f"It’s been about {days} days since you last wore this piece — consider if it’s time to bring it back."
        else:
            context_line = "Think about when you last wore this piece, and whether it's time to revive it."
    elif agent_name == "My Own Words":
        processed = my_own_words_prep(item_id)
        user_comments = processed.get("user_comments", [])

        if user_comments:
            joined_comments = " ".join(user_comments)
            context_line = (
                f"These are the user's own reflections about this piece:\n\n"
                f"\"{joined_comments}\"\n\n"
                f"Respond as if reminding them of their own feelings and thoughts — "
                f"use their language and tone naturally."
            )
        else:
            context_line = (
                "The user hasn’t written any reflections for this item yet. "
                "Encourage them to add their thoughts next time."
            )


    # --- Unified contextual prompt builder ---
    item_source = st.session_state.get("selected_source", "unknown source")

    # Build item context (pull from Wardrobe sheet if available)
    try:
        sheet_wardrobe = gs_client.open("lexi_live").worksheet("Wardrobe")
        df_wardrobe = pd.DataFrame(sheet_wardrobe.get_all_records())
        df_wardrobe.columns = [c.strip().lower() for c in df_wardrobe.columns]

        row = df_wardrobe[df_wardrobe["item id"].astype(str) == str(item_id)]
        if not row.empty:
            item_name = str(row.iloc[0].get("item name", "an item")).strip()
            item_colour = str(row.iloc[0].get("colour", row.iloc[0].get("color", ""))).strip()
            item_type = str(row.iloc[0].get("category", row.iloc[0].get("type", ""))).strip()
            item_desc = f"{item_colour} {item_name}".strip()
            if item_type:
                item_desc += f" ({item_type})"
        else:
            item_desc = f"Item {item_id}"
    except Exception:
        item_desc = f"Item {item_id}"

    # Construct shared context block
    context_parts = [
        f"You are Lexi, a wardrobe AI that speaks in a friendly, concise, and emotionally aware tone.",
        f"You are currently acting through the **{agent_name}** agent, whose perspective is: {agent_prompt}",
        f"The user selected {item_desc} via {item_source}.",
    ]

    # Include combos, user reflections, or metrics depending on the agent
    if combos:
        combo_context = "\n".join([f"- {c}" for c in combos])
        context_parts.append(f"The wardrobe database suggests these outfit combinations:\n{combo_context}")

    if agent_name == "My Own Words":
        user_comments = processed.get("user_comments", [])
        if user_comments:
            context_parts.append(
                f"The user has previously written about this item in their own words:\n"
                f"---\n{'. '.join(user_comments)}\n---\n"
                f"Reflect back their tone and remind them of what they expressed."
            )
        else:
            context_parts.append("No prior reflections found for this item.")

    if extra_info:
        context_parts.append(f"Additional insight: Cost per wear is approximately ${extra_info:.2f}.")

    if agent_name == "Old Favourite":
        days = processed.get("days_since_last_wear")
        if days is not None:
            context_parts.append(f"It’s been around {days} days since the user last wore this piece.")

    # Combine everything into a unified instruction
    full_prompt = "\n\n".join(context_parts) + (
        "\n\nYour task: Generate a cohesive 3-line reflection that blends the agent's lens, "
        "the user's data (combos, history, or comments), and prioritise the selected item context. "
)


    # 4️⃣ Query GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are Lexi's wardrobe assistant."},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.8,
    )

    suggestion_text = response.choices[0].message.content.strip()

    # Append extra info visibly and vocally
    if extra_info:
        suggestion_text += f"\n\n Estimated Cost per Wear: ${extra_info:.2f}"

    # ✅ return correct combo IDs (IDC1, IDC2, IDC3)
    return {
        "suggestion": suggestion_text,
        "combo_ids": combo_ids,
    }



# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="LEXI – Wardrobe Curator", page_icon="🟡", layout="wide")
st.markdown("<h1 style='text-align: center; font-size: 80px;'>LEXI</h1>", unsafe_allow_html=True)

# ---------- GOOGLE SHEET CONNECTION ----------

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# st.secrets["google"]
google_secrets = dict(st.secrets["google"])

creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["google"], scope)
gs_client = gspread.authorize(creds)

sheet = gs_client.open("lexi_live").worksheet("detections")
data = sheet.get_all_records()

import pandas as pd
df = pd.DataFrame(data)
### st.dataframe(df)

# ---------- CAMERA & GALLERY SELECTION ----------
def get_last_item_id(sheet, column_index=2):
    try:
        col_values = sheet.col_values(column_index)
        if not col_values:
            return None
        for value in reversed(col_values):
            if str(value).strip():
                return value.strip()
        return None
    except Exception as e:
        st.error(f"Error fetching latest item: {e}")
        return None

def set_selected_item(source_label, sheet):
    item_id = get_last_item_id(sheet)
    if item_id:
        st.session_state["selected_item_id"] = item_id
        st.session_state["selected_source"] = source_label
        st.toast(f"{source_label} item selected: {item_id}")
    else:
        st.warning(f"No items found from {source_label} sheet.")

# Sheets
sheet_camera = gs_client.open("lexi_live").worksheet("detections")
sheet_gallery = None  # placeholder

# ---------- Responsive Top Section ----------
st.markdown(
    """
    <style>
        /* Base layout for all screen sizes */
        .top-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            margin-bottom: 1rem;
        }

        .button-row {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 0.5rem;
            flex-wrap: wrap;
        }

        /* Desktop layout (side-by-side) */
        @media (min-width: 768px) {
            .top-section {
                flex-direction: row;
                justify-content: center;
                align-items: center;
            }
            .button-row {
                flex-wrap: nowrap;
                margin-top: 0;
            }
        }

        /* Optional: make buttons flexible on small screens */
        .button-row button {
            flex: 1;
            min-width: 120px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Layout structure ---
st.markdown('<div class="top-section">', unsafe_allow_html=True)

# Centered title on all screens
st.markdown("<h3>Curating Outfits Everyday</h3>", unsafe_allow_html=True)

# Buttons row — below on mobile, beside on desktop
st.markdown('<div class="button-row">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Choose from Camera", key="camera_button"):
        set_selected_item("Camera", sheet_camera)

with col2:
    if st.button("Choose from Gallery", key="gallery_button"):
        st.session_state["gallery_open"] = not st.session_state.get("gallery_open", False)

st.markdown('</div></div>', unsafe_allow_html=True)


# --- Step 2: Only show the gallery dropdown when toggled open ---
if st.session_state.get("gallery_open", False):
    st.markdown("### Select from Gallery")
    try:
        # Load the Wardrobe sheet
        sheet_gallery = gs_client.open("lexi_live").worksheet("Wardrobe")
        records_gallery = sheet_gallery.get_all_records()
        df_gallery = pd.DataFrame(records_gallery)

        # Normalize columns
        df_gallery.columns = [c.strip().lower() for c in df_gallery.columns]

        # Confirm required columns exist
        required_cols = ["item name", "item id"]
        for col in ["colour", "color"]:
            if col in df_gallery.columns:
                required_cols.append(col)
                break

        if all(col in df_gallery.columns for col in ["item name", "item id"]):
            # Determine colour column name dynamically
            colour_col = "colour" if "colour" in df_gallery.columns else "color" if "color" in df_gallery.columns else None

            # Build display names: "Colour – Item Name"
            if colour_col:
                df_gallery["display_name"] = df_gallery[colour_col].fillna("") + " – " + df_gallery["item name"].fillna("")
            else:
                df_gallery["display_name"] = df_gallery["item name"].fillna("")

            # Drop duplicates and empty rows
            item_display_list = df_gallery["display_name"].dropna().unique().tolist()

            # Show dropdown
            selected_display = st.selectbox("Choose an item from your gallery:", item_display_list)

            if st.button("Confirm Selection", key="gallery_select_button"):
                selected_row = df_gallery[df_gallery["display_name"] == selected_display]
                if not selected_row.empty:
                    item_id = str(selected_row.iloc[0]["item id"])
                    st.session_state["selected_item_id"] = item_id
                    st.session_state["selected_source"] = "Gallery"
                    st.toast(f"Gallery item selected: {selected_display} ({item_id})")
                    st.session_state["gallery_open"] = False
                else:
                    st.warning("No matching Item ID found for that name.")
        else:
            st.warning("Wardrobe sheet must include 'Item Name' and 'Item ID' columns.")

    except Exception as e:
        st.warning(f"Gallery connection failed: {e}")

# Persistent display
if "selected_item_id" in st.session_state:
    src = st.session_state.get("selected_source", "Unknown Source")
    st.info(f"Selected item: **{st.session_state['selected_item_id']}** from {src}")
else:
    st.caption("No item selected yet — choose from Camera or Gallery.")



# ---------- WARDROBE AGENT INTERACTION ----------
# Define agents
selected_agent = st.selectbox(
    "Choose your AI Assistant:",
    options=list(AGENT_SPECS.keys()),
    key="agent_selector"
)
if selected_agent:
    st.markdown(f"_Prompt seed:_ **{AGENT_SPECS[selected_agent]['prompt']}**")


# Suggestion generator button
if st.button("Fetch Outfit", use_container_width=True):
    with st.spinner(f"LEXI ({selected_agent}) is curating your outfit..."):
        result = query_agent(selected_agent)
        if result:
            st.session_state["suggestion"] = result["suggestion"]
            st.session_state["combo_ids"] = result.get("combo_ids", [])

            # --- Show Lexi Suggests section immediately ---
            st.markdown("### Lexi Suggests")
            suggestion_text = st.session_state["suggestion"]
            st.write(suggestion_text)

            # --- Convert suggestion to speech using gTTS ---
            spoken_text = st.session_state["suggestion"]

            import re

            # Handle all forms like $25, $25.50, $25,000 etc.
            def pronounce_dollars(text):
                # Match $ followed by digits (and optional commas or decimal)
                def replacer(match):
                    amount = match.group(1)
                    return f"{amount} dollars"  # say “25 dollars”
                # Convert things like "$25.50" → "25.50 dollars"
                text = re.sub(r"\$([0-9.,]+)", replacer, text)
                # Replace any remaining stray "$" just in case
                text = text.replace("$", " dollars ")
                return text

            spoken_text = pronounce_dollars(st.session_state["suggestion"])


            from openai import OpenAI
            import base64, tempfile, re

            spoken_text = st.session_state["suggestion"]

            # Convert currency expressions for clarity
            spoken_text = re.sub(r"\$([0-9.,]+)", r"\1 dollars", spoken_text)

            # Optional: tone cue for more expressive results
            spoken_text = (
                f"Speak with a natural Australian female rhythm and tone, "
                f"sounding friendly and confident: {spoken_text}"
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                # ✅ Correct v1.x syntax — no 'format' argument
                response = client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice="sage",
                    input=spoken_text
                )
                response.stream_to_file(tmp_file.name)

                # --- Embed MP3 into Streamlit ---
                with open(tmp_file.name, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    encoded = base64.b64encode(audio_bytes).decode()

                # Use a unique key each time to force reload of audio
                audio_key = f"lexi_audio_{datetime.now().timestamp()}"

                audio_html = f"""
                    <audio key="{audio_key}" autoplay controls style="width: 100%;">
                        <source src="data:audio/mp3;base64,{encoded}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)


## Feedback form

with st.form("feedback_form"):
    user_comment = st.text_area("What are you thoughts, suggestions of feedback after seeing Lexi?", height=100)
    submit_feedback = st.form_submit_button("Submit Feedback")

    if submit_feedback:
        try:
            sheet_feedback = gs_client.open("lexi_live").worksheet("form responses")
            # Determine which Item_IDs to store
            combo_ids = st.session_state.get("combo_ids", [])
            if combo_ids:
                item_ids_to_save = ", ".join(combo_ids)
            else:
                # Fallback: just use the selected item being discussed
                item_ids_to_save = str(st.session_state.get("selected_item_id", ""))

            feedback_entry = [
                str(pd.Timestamp.now()),
                selected_agent,
                st.session_state.get("suggestion", ""),
                item_ids_to_save,  # ensures Item_IDs is never blank
                user_comment
            ]

            sheet_feedback.append_row(feedback_entry)

            st.success("Feedback saved!")
            # Reset interface after successful feedback
            st.session_state.pop("selected_item_id", None)
            st.session_state.pop("combo_ids", None)
            st.session_state.pop("suggestion", None)

            # Clear the text area (Streamlit form input)
            st.rerun()

        except Exception as e:
            st.warning(f"Feedback not saved: {e}")






