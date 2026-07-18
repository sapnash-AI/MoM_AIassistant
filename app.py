import streamlit as st
from google import genai
from google.genai import types

# --- Page Config & Styling ---
st.set_page_config(page_title="Meeting Minutes AI (Gemini)", page_icon="🎙️", layout="wide")

st.title("🎙️PWS- Meeting Summariser & Action Point Generator-Pilot Project")
st.caption("Record live audio and leverage Gemini's native audio intelligence to transcribe, summarize, and highlight targets.")

# --- API Configuration via Streamlit Secrets ---
# Looks for GEMINI_API_KEY in Streamlit Community Cloud secrets or your local secrets.toml file
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "gemini" in st.secrets and "api_key" in st.secrets["gemini"]: # Fallback structure style
    api_key = st.secrets["gemini"]["api_key"]
else:
    st.error("🔑 **API Key Missing:** Please configure your `GEMINI_API_KEY` inside your Streamlit Secrets.")
    st.markdown("""
    ### 🛠️ How to fix this:
    * **Locally:** Create a directory named `.streamlit/` in your project folder, add a file named `secrets.toml`, and write: `GEMINI_API_KEY = "your_actual_api_key_here"`
    * **On Streamlit Community Cloud:** Go to your App Dashboard -> App Settings -> Secrets, and add:
    ```toml
    GEMINI_API_KEY = "your_actual_api_key_here"
    ```
    """)
    st.stop()

# Initialize the modern Google Gen AI client with the secure key
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
    audio_file = st.audio_input("Record your live discussion here")
    
    # Only run the pipeline if there is a new audio file recording
    if audio_file is not None and audio_file != st.session_state.last_processed_audio:
        with st.spinner("Gemini is listening and analyzing the audio..."):
            try:
                audio_bytes = audio_file.read()
                
                # Setup structured generation instructions using a System Prompt
                config = types.GenerateContentConfig(
                    system_instruction=(
                        "You are an expert executive assistant. Analyze the audio provided. "
                        "First, write down an accurate word-for-word transcript. "
                        "Second, write a concise paragraph summary of the key talking points. "
                        "Third, highlight action items with bolded names and responsibilities. "
                        "Separate sections cleanly using the labels [TRANSCRIPT], [SUMMARY], and [ACTIONS]."
                    ),
                    temperature=0.4,
                )
                
                # Gemini accepts binary audio data inline via types.Part.from_bytes
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type="audio/wav"
                        ),
                        "Process this meeting audio according to your system instructions."
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
                    # Fallback parsing if structure deviates slightly
                    st.session_state.transcript = "Please see full output on the right column."
                    st.session_state.summary = full_text
                    st.session_state.action_points = "Check summary details above."
                
            except Exception as e:
                st.error(f"Error during Gemini processing: {e}")
                st.stop()

    # Display the raw text transcript if generated
    if st.session_state.transcript:
        st.markdown("### 📝 Text Transcript")
        st.text_area("Full text generated from speech:", st.session_state.transcript, height=250)

with col_output:
    st.subheader("💡 Gemini Post-Meeting Analysis")
    
    if st.session_state.summary or st.session_state.action_points:
        st.markdown("### 📌 Discussion Summary")
        st.info(st.session_state.summary if st.session_state.summary else "Awaiting processing...")
        
        st.markdown("### 🚀 Action Items & Deliverables")
        st.success(st.session_state.action_points if st.session_state.action_points else "Awaiting processing...")
        
        # Add a download feature for documentation
        meeting_notes = (
            f"# Meeting Minutes\n\n## Transcript\n{st.session_state.transcript}\n\n"
            f"## Summary\n{st.session_state.summary}\n\n## Action Items\n{st.session_state.action_points}"
        )
        st.download_button(
            label="💾 Export Meeting Markdown",
            data=meeting_notes,
            file_name="meeting_minutes.md",
            mime="text/markdown"
        )
    else:
        st.info("Your AI insights, summaries, and highlighted items will show up right here once you record and save audio.")
