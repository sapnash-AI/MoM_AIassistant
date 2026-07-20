import streamlit as st
from google import genai
from google.genai import types

# --- Page Config & Styling ---
st.set_page_config(page_title="Multilingual Meeting Minutes AI", page_icon="🎙️", layout="wide")

st.title("🎙️ PWS-Audio Meeting Minutes & Action Points")
st.caption("Record live audio in English, Hindi, or a mix of both. Raw transcript matches speech, while insights are delivered in English.")

# --- API Configuration via Streamlit Secrets ---
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
    api_key = st.secrets["gemini"]["api_key"]
else:
    st.error("🔑 **API Key Missing:** Please configure your `GEMINI_API_KEY` inside your Streamlit Secrets.")
    st.stop()

# Initialize the modern Google Gen AI client
client = genai.Client(api_key=api_key)

# --- Initialize Session States ---
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "action_points" not in st.session_state:
    st.session_state.action_points = ""
if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

# --- Layout Columns ---
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("🔊 Audio Input & Processing")
    
    # Native Streamlit audio recorder
    audio_file = st.audio_input("Record your live discussion here (English / Hindi / Hinglish)")
    
    # Only run the pipeline if there is a new audio file recording
    if audio_file is not None and audio_file != st.session_state.last_processed_audio:
        with st.spinner("Gemini is listening and translating the discussion..."):
            try:
                audio_bytes = audio_file.read()
                
                # Rigid instructions forcing English output for Summary and Action points
                system_instruction = (
                    "You are an expert executive assistant fluent in both English and Hindi. "
                    "Analyze the audio provided. The audio might be in English, Hindi, or a code-switched mix of both (Hinglish). "
                    "\n\n"
                    "Provide your analysis exactly inside the designated tags:\n"
                    "1. Under the label [TRANSCRIPT], write down an accurate word-for-word transcript. If the user spoke Hindi, use Devanagari script. If mixed, write it exactly as spoken.\n"
                    "2. Under the label [SUMMARY], translate the primary themes if necessary and write a concise paragraph summary of the key talking points strictly in English.\n"
                    "3. Under the label [ACTIONS], highlight action items with bolded names and responsibilities strictly in English. If dates or deadlines were discussed, include them cleanly."
                )
                
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3, # Lower temperature for less creative variance and stricter compliance
                )
                
                # Process the native audio file
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type="audio/wav"
                        ),
                        "Process this meeting audio according to your custom language instructions."
                    ],
                    config=config
                )
                
                full_text = response.text
                st.session_state.last_processed_audio = audio_file
                
                # Parse the response text out into its respective blocks safely
                if "[TRANSCRIPT]" in full_text and "[SUMMARY]" in full_text and "[ACTIONS]" in full_text:
                    parts_ts = full_text.split("[TRANSCRIPT]")
                    parts_sum = parts_ts[1].split("[SUMMARY]")
                    parts_act = parts_sum[1].split("[ACTIONS]")
                    
                    st.session_state.transcript = parts_sum[0].strip()
                    st.session_state.summary = parts_act[0].strip()
                    st.session_state.action_points = parts_act[1].strip()
                else:
                    # Fallback parsing if layout markers break down
                    st.session_state.transcript = "Please see full translated output on the right column."
                    st.session_state.summary = full_text
                    st.session_state.action_points = "Check summary details above."
                
            except Exception as e:
                st.error(f"Error during Gemini processing: {e}")
                st.stop()

    # Display the raw text transcript if generated
    if st.session_state.transcript:
        st.markdown("### 📝 Text Transcript (Original Spoken Language)")
        st.text_area("Full text generated from speech:", st.session_state.transcript, height=250)

with col_output:
    st.subheader("💡 Gemini Post-Meeting Analysis (English)")
    
    if st.session_state.summary or st.session_state.action_points:
        st.markdown("### 📌 Discussion Summary")
        st.info(st.session_state.summary if st.session_state.summary else "Awaiting processing...")
        
        st.markdown("### 🚀 Action Items & Deliverables")
        st.success(st.session_state.action_points if st.session_state.action_points else "Awaiting processing...")
        
        # Add a download feature for documentation
        meeting_notes = (
            f"# Meeting Minutes\n\n## Transcript\n{st.session_state.transcript}\n\n"
            f"## Summary (English)\n{st.session_state.summary}\n\n## Action Items (English)\n{st.session_state.action_points}"
        )
        st.download_button(
            label="💾 Export Meeting Markdown",
            data=meeting_notes,
            file_name="meeting_minutes.md",
            mime="text/markdown"
        )
    else:
        st.info("Your English insights, summaries, and highlighted action items will show up right here once you record audio.")
