import base64
import json
import os
import requests
import streamlit as st

DATA_FILE = "student_memory.json"

# Page configuration
st.set_page_config(
    page_title="Personal AI Tutor", page_icon="🎓", layout="wide"
)


# Load and save memory functions
def load_memory():
    if not os.path.exists(DATA_FILE):
        return {
            "subjects": {},
            "test_scores": [],
            "system_prompt": (
                "You are an encouraging and knowledgeable personal AI tutor."
            ),
        }
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
        if "system_prompt" not in data:
            data["system_prompt"] = (
                "You are an encouraging and knowledgeable personal AI tutor."
            )
        return data


def save_memory(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


# Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = load_memory()


def get_api_key():
    """Safely fetch API Key from environment variables or Streamlit secrets."""
    api_key = os.getenv("SARVAM_API_KEY")
    try:
        if "SARVAM_API_KEY" in st.secrets:
            api_key = st.secrets["SARVAM_API_KEY"]
    except Exception:
        pass  # Fallback if secrets.toml file does not exist locally
    return api_key


def call_sarvam_ai(prompt):
    """Generates AI response using Sarvam LLM (sarvam-105b)."""
    api_key = get_api_key()
    if not api_key:
        return (
            "⚠️ Error: Sarvam API Key not found. Please set it in your environment"
            " variables or Streamlit Secrets."
        )

    url = "https://api.sarvam.ai/v1/chat/completions"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sarvam-105b",
        "messages": [
            {
                "role": "system",
                "content": st.session_state.memory.get(
                    "system_prompt",
                    "You are an encouraging and knowledgeable personal AI tutor.",
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()

        if response.status_code == 200 and "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        elif "error" in res_json:
            return f"⚠️ API Error ({response.status_code}): {res_json['error'].get('message', res_json)}"
        else:
            return f"Unexpected API response: {res_json}"
    except Exception as e:
        return f"Error communicating with Sarvam API: {e}"


def sarvam_stt(audio_bytes, language_code="en-IN"):
    """Transcribes audio using Sarvam AI Speech-to-Text API (saaras:v1)."""
    api_key = get_api_key()
    if not api_key:
        return None, "API Key missing."

    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": api_key}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {
        "model": "saaras:v1",
        "language_code": language_code,
        "with_timestamps": "false",
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        res_json = response.json()
        if response.status_code == 200 and "transcript" in res_json:
            return res_json["transcript"], None
        return None, f"STT API error ({response.status_code}): {res_json}"
    except Exception as e:
        return None, f"STT Request Exception: {e}"


def sarvam_tts(text, target_language_code="en-IN", speaker="anushka"):
    """Converts text response to audio using Sarvam AI Text-to-Speech API (bulbul:v3)."""
    api_key = get_api_key()
    if not api_key:
        return None, "API Key missing."

    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": [text[:500]],  # Truncate to fit single batch constraints
        "target_language_code": target_language_code,
        "speaker": speaker,
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v3",
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        res_json = response.json()
        if (
            response.status_code == 200
            and "audios" in res_json
            and len(res_json["audios"]) > 0
        ):
            audio_base64 = res_json["audios"][0]
            return base64.b64decode(audio_base64), None
        return None, f"TTS API error ({response.status_code}): {res_json}"
    except Exception as e:
        return None, f"TTS Request Exception: {e}"


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎓 AI Tutor Navigation")

# Audio Controls in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🎙️ Voice Settings")
enable_tts = st.sidebar.checkbox("Enable Voice Responses (TTS)", value=True)
selected_lang = st.sidebar.selectbox(
    "Language",
    ["en-IN", "hi-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN", "gu-IN", "mr-IN"],
    index=0,
)
selected_speaker = st.sidebar.selectbox(
    "Speaker Voice",
    ["aditya"],
    index=0,
)
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Log Daily Learning",
        "Track Test Scores",
        "What to Learn Next?",
        "Practice Generator",
        "Prompt Customization (Settings)",
    ],
)

# --- 1. DASHBOARD ---
if menu == "Dashboard":
    st.title("Welcome to Your Personal AI Tutor 🚀")
    st.markdown(
        "Track your progress, monitor test scores, find weak spots, and practice"
        " with custom AI-generated problems."
    )

    memory = st.session_state.memory
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Tracked Subjects")
        if memory["subjects"]:
            for subj, subtopics in memory["subjects"].items():
                with st.expander(f"Subject: {subj}"):
                    for sub, details in subtopics.items():
                        st.write(
                            f"- **{sub}**: Confidence Score = {details['confidence']}/5"
                        )
        else:
            st.info("No learning data logged yet. Go to 'Log Daily Learning'.")

    with col2:
        st.subheader("📊 Recent Test Scores")
        if memory["test_scores"]:
            for test in memory["test_scores"][-5:]:
                st.write(
                    f"- **{test['subject']} ({test['topic']})**: {test['score']}/"
                    f"{test['total']}"
                )
        else:
            st.info("No test scores tracked yet. Go to 'Track Test Scores'.")

# --- 2. LOG DAILY LEARNING ---
elif menu == "Log Daily Learning":
    st.title("📝 Log Your Daily Learning")

    with st.form("daily_form"):
        subject = st.text_input(
            "Subject (e.g., Mathematics, History, Python)"
        ).strip()
        subtopic = st.text_input(
            "Subtopic learned today (e.g., Recursion, World War II)"
        ).strip()
        confidence = st.slider(
            "Rate your confidence in this subtopic (1 = Lost, 5 = Mastered)",
            1,
            5,
            3,
        )

        submitted = st.form_submit_button("Save Entry")
        if submitted and subject and subtopic:
            memory = st.session_state.memory
            if subject not in memory["subjects"]:
                memory["subjects"][subject] = {}

            memory["subjects"][subject][subtopic] = {"confidence": confidence}
            save_memory(memory)
            st.success(
                f"✅ Saved! '{subtopic}' recorded under {subject} with confidence"
                f" {confidence}/5."
            )
        elif submitted:
            st.warning("Please fill in both subject and subtopic fields.")

# --- 3. TRACK TEST SCORES ---
elif menu == "Track Test Scores":
    st.title("📊 Test Score Tracking")
    st.markdown(
        "Record your exam or quiz scores to help the AI spot your recurring weak"
        " areas."
    )

    with st.form("test_form"):
        test_subject = st.text_input("Subject (e.g., Physics)").strip()
        test_topic = st.text_input(
            "Chapter / Topic (e.g., Electromagnetism)"
        ).strip()
        score = st.number_input("Score Obtained", min_value=0.0, step=1.0)
        total_marks = st.number_input(
            "Total Marks", min_value=1.0, value=100.0, step=1.0
        )
        mistakes_notes = st.text_area(
            "What mistakes did you make? (Optional notes)"
        ).strip()

        test_submitted = st.form_submit_button("Save Test Score")
        if test_submitted and test_subject and test_topic:
            memory = st.session_state.memory
            test_record = {
                "subject": test_subject,
                "topic": test_topic,
                "score": score,
                "total": total_marks,
                "notes": mistakes_notes,
            }
            memory["test_scores"].append(test_record)
            save_memory(memory)
            st.success(
                f"✅ Test score for {test_topic} ({score}/{total_marks}) saved"
                " successfully!"
            )
        elif test_submitted:
            st.warning("Please fill in the subject and topic fields.")

    # Display Test History table
    if st.session_state.memory["test_scores"]:
        st.subheader("Your Test History")
        for idx, t in enumerate(
            reversed(st.session_state.memory["test_scores"])
        ):
            st.markdown(
                f"**{t['subject']} - {t['topic']}**: {t['score']}/{t['total']}"
                f" (Notes: {t.get('notes', 'None')})"
            )

# --- 4. WHAT TO LEARN NEXT ---
elif menu == "What to Learn Next?":
    st.title("🎯 Smart Study Recommendation")
    st.markdown(
        "Let your AI tutor analyze your confidence levels and test scores to guide"
        " your next study session."
    )

    if st.button("Generate Recommendation"):
        memory = st.session_state.memory
        if not memory["subjects"] and not memory["test_scores"]:
            st.warning(
                "Please log some learning data or test scores first before asking for"
                " recommendations."
            )
        else:
            with st.spinner("Analyzing your learning profile..."):
                prompt = (
                    "Here is the student's mastery record (confidence scale 1-5):\n"
                    f"{json.dumps(memory['subjects'], indent=2)}\n\n"
                    "And here are their recent test scores and mistakes:\n"
                    f"{json.dumps(memory['test_scores'], indent=2)}\n\n"
                    "Analyze these. Recommend what the student should focus on learning"
                    " next, prioritizing weak subtopics and test errors. Provide an"
                    " encouraging rationale."
                )
                response = call_sarvam_ai(prompt)
                st.subheader("AI Recommendation:")
                st.write(response)

                if enable_tts and response and not response.startswith("⚠️"):
                    with st.spinner("Generating speech audio..."):
                        audio_bytes, err = sarvam_tts(
                            response,
                            target_language_code=selected_lang,
                            speaker=selected_speaker,
                        )
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/wav", autoplay=True)
                        elif err:
                            st.error(err)

# --- 5. PRACTICE GENERATOR ---
elif menu == "Practice Generator":
    st.title("💡 Custom Practice Problem Generator")
    st.markdown(
        "Generate custom problems tailored precisely to your weak areas or"
        " upcoming tests."
    )

    st.markdown("### 🎙️ Audio Input (Optional)")
    audio_input = st.audio_input("Record your question or topic:")

    transcribed_area = ""
    if audio_input is not None:
        with st.spinner("Transcribing your spoken prompt via Sarvam STT..."):
            raw_bytes = audio_input.read()
            transcription, err = sarvam_stt(raw_bytes, language_code=selected_lang)
            if transcription:
                st.success(f"🗣️ **Transcribed Text:** {transcription}")
                transcribed_area = transcription
            elif err:
                st.error(err)

    weak_area = st.text_input(
        "Enter weak topic or concept (or use audio above)", value=transcribed_area
    ).strip()
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])

    if st.button("Generate Practice Problems"):
        if weak_area:
            with st.spinner("Generating custom problem set..."):
                prompt = (
                    f"Generate 3 practice problems for the topic: '{weak_area}' at"
                    f" '{difficulty}' difficulty level. Do not provide the answers"
                    " immediately, but give hints if needed."
                )
                response = call_sarvam_ai(prompt)
                st.subheader("Your Custom Problem Set:")
                st.write(response)

                if enable_tts and response and not response.startswith("⚠️"):
                    with st.spinner("Generating voice response..."):
                        audio_bytes, err = sarvam_tts(
                            response,
                            target_language_code=selected_lang,
                            speaker=selected_speaker,
                        )
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/wav", autoplay=True)
                        elif err:
                            st.error(err)
        else:
            st.warning("Please enter or record a topic to generate practice problems.")

# --- 6. PROMPT CUSTOMIZATION ---
elif menu == "Prompt Customization (Settings)":
    st.title("⚙️ AI Tutor Prompt Customization")
    st.markdown(
        "Customize how your AI tutor behaves, talks, and teaches you by editing"
        " its system prompt."
    )

    current_prompt = st.session_state.memory.get(
        "system_prompt",
        "You are an encouraging and knowledgeable personal AI tutor.",
    )

    new_prompt = st.text_area(
        "System Prompt (AI Persona Guidelines)", value=current_prompt, height=150
    )

    if st.button("Update AI Persona"):
        st.session_state.memory["system_prompt"] = new_prompt
        save_memory(st.session_state.memory)
        st.success("✅ AI Tutor persona successfully updated!")