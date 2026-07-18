import streamlit as st
from openai import OpenAI
import os

# --- Page Config & Styling ---
st.set_page_config(page_title="Meeting Minutes AI", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Audio Meeting Minutes & Action Points")
st.caption("Record live audio, transcribe instantly, and let AI summarize your key deliverables.")

# --- API Configuration ---
# Sidebar for secure API management
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Grab your key from platform.openai.com")
    
    # Simple instructions inside the sidebar
    st.markdown("---")
    st.markdown("""
    ### 💡 How to use:
    1. Paste your OpenAI API Key.
    2. Click **Record** on the microphone widget below.
    3. Speak your meeting notes or discussion details.
    4. Click **Stop** to let the AI process the transcription, generate summaries, and highlight action items!
    """)

# Check for API key before continuing
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# --- Initialize Session States ---
# We store results in state so they don't disappear when Streamlit re-renders
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "action_points" not in st.session_state:
    st.session_state.action_points = ""

# --- Layout Columns ---
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("🔊 Audio Input & Processing")
    
    # Native Streamlit audio recorder (sets 16kHz sample rate optimal for speech recognition)
    audio_file = st.audio_input("Record your live discussion here")
    
    if audio_file is not None:
        # Show a processing spinner
        with st.spinner("Processing your audio transcription..."):
            try:
                # Read the bytes buffer from the Streamlit UploadedFile object
                audio_bytes = audio_file.read()
                
                # Whisper requires a named file format, so we pass standard metadata name
                # using the client.audio.transcriptions API
                transcription_response = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=("live_meeting.wav", audio_bytes, "audio/wav")
                )
                
                st.session_state.transcript = transcription_response.text
                st.success("🎉 Transcription complete!")
                
            except Exception as e:
                st.error(f"Error during transcription: {e}")
                st.stop()

        # Generate Insights if a transcript exists
        if st.session_state.transcript:
            with st.spinner("Analyzing discussion & building action plan..."):
                try:
                    # Formulate system instructions to structure the summary
                    system_prompt = (
                        "You are an expert executive assistant. Take the meeting transcript provided "
                        "and generate two distinct outputs: \n"
                        "1. A concise, paragraph-based summary of the main discussion points.\n"
                        "2. A clear bulleted list of actionable items or deliverables, explicitly bolding "
                        "names, responsibilities, or deadlines if mentioned."
                    )
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Transcript:\n{st.session_state.transcript}"}
                        ],
                        temperature=0.5
                    )
                    
                    full_analysis = response.choices[0].message.content
                    
                    # Rudimentary separation of Summary and Action items for split UI display
                    if "action" in full_analysis.lower():
                        # Try to split if the AI outputs structured headers
                        parts = full_analysis.split("\n\n")
                        st.session_state.summary = parts[0]
                        st.session_state.action_points = "\n\n".join(parts[1:])
                    else:
                        st.session_state.summary = full_analysis
                        st.session_state.action_points = "No specific action items detected."
                        
                except Exception as e:
                    st.error(f"AI Analytics error: {e}")

    # Display the raw text transcript if generated
    if st.session_state.transcript:
        st.markdown("### 📝 Raw Transcript")
        st.text_area("Full text generated from speech:", st.session_state.transcript, height=250)

with col_output:
    st.subheader("💡 AI Post-Meeting Analysis")
    
    if st.session_state.summary or st.session_state.action_points:
        st.markdown("### 📌 Discussion Summary")
        st.info(st.session_state.summary if st.session_state.summary else "Awaiting processing...")
        
        st.markdown("### 🚀 Action Items & Deliverables")
        # Visual styling callout box for clarity
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
        st.light("Your AI insights, summaries, and highlighted items will show up right here once you record and save audio.")
