import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Next Word Prediction System",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.5rem;
}

.block-container {
    max-width: 1200px;
}

.stButton > button {
    width:100%;
    border-radius:12px;
    height:48px;
    font-weight:600;
}

.feature-box {
    padding:15px;
    border-radius:15px;
    background:#f8fafc;
    border:1px solid #e2e8f0;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD ASSETS
# =====================================================

@st.cache_resource
def load_assets():

    model = load_model("next_word_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("config.json", "r") as f:
        config = json.load(f)

    return model, tokenizer, config


model, tokenizer, config = load_assets()

sequence_length = config["sequence_length"]
total_words = len(tokenizer.word_index) + 1

index_to_word = {
    idx: word
    for word, idx in tokenizer.word_index.items()
}

# =====================================================
# SESSION STATE
# =====================================================

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

if "pending_update" not in st.session_state:
    st.session_state.pending_update = None

# =====================================================
# APPLY PENDING UPDATE
# =====================================================

if st.session_state.pending_update is not None:
    st.session_state.text_input = st.session_state.pending_update
    st.session_state.pending_update = None

# =====================================================
# FUNCTIONS
# =====================================================

def get_top_predictions(text, top_n=3):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return []

    token_list = token_list[-sequence_length:]

    padded = np.pad(
        token_list,
        (sequence_length - len(token_list), 0),
        mode="constant"
    )

    padded = np.array([padded])

    pred = model.predict(
        padded,
        verbose=0
    )[0]

    top_indices = np.argsort(pred)[-top_n:][::-1]

    words = []

    for idx in top_indices:

        word = index_to_word.get(idx)

        if word:
            words.append(word)

    return words


def autocomplete(prefix, limit=5):

    prefix = prefix.lower().strip()

    if not prefix:
        return []

    matches = [
        word
        for word in tokenizer.word_index.keys()
        if word.startswith(prefix)
    ]

    return sorted(matches)[:limit]


def replace_last_word(text, new_word):

    words = text.split()

    if not words:
        return new_word

    words[-1] = new_word

    return " ".join(words)


def append_word(text, word):

    if not text.strip():
        return word

    return text + " " + word

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero">
<h1>🤖 Next Word Prediction System</h1>
<p>
Deep Learning Based Predictive Text Generation Using LSTM Networks
</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📊 Model Information")

    


    st.info("""
Features

• Prefix Autocomplete

• Next Word Prediction

• Interactive Suggestions

• LSTM Deep Learning Model
""")

# =====================================================
# INPUT
# =====================================================

st.subheader("✍ Start Typing")

user_text = st.text_input(
    "",
    key="text_input",
    placeholder="Type something..."
)

# =====================================================
# AUTOCOMPLETE
# =====================================================

if user_text.strip():

    last_word = user_text.split()[-1]

    auto_words = autocomplete(last_word)

    if auto_words:

        st.subheader("🔎 Autocomplete")

        cols = st.columns(len(auto_words))

        for i, word in enumerate(auto_words):

            if cols[i].button(
                word,
                key=f"auto_{i}"
            ):

                st.session_state.pending_update = replace_last_word(
                    user_text,
                    word
                )

                st.rerun()

# =====================================================
# NEXT WORD SUGGESTIONS
# =====================================================

if user_text.strip():

    st.subheader("🎯 Next Word Suggestions")

    predictions = get_top_predictions(
        user_text,
        top_n=3
    )

    if predictions:

        cols = st.columns(len(predictions))

        for i, word in enumerate(predictions):

            if cols[i].button(
                word,
                key=f"pred_{i}"
            ):

                st.session_state.pending_update = append_word(
                    user_text,
                    word
                )

                st.rerun()

# =====================================================
# INFO SECTION
# =====================================================

st.markdown("---")

st.markdown("""
<div class="feature-box">

<h4>How It Works</h4>

<ul>
<li>Type text in the input field.</li>
<li>Autocomplete suggests matching words.</li>
<li>Click autocomplete to replace the current word.</li>
<li>LSTM predicts the next most likely words.</li>
<li>Click a prediction to continue the sentence.</li>
</ul>

</div>
""", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "LSTM Next Word Prediction System"
)