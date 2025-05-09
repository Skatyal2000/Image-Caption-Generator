import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences
from pickle import load
from gtts import gTTS
import os
import uuid
from dotenv import load_dotenv
import gdown
from pickle import load as pickle_load

@st.cache_resource
def load_captioning_model():
    model_path = "fnl_epoch_45.h5"
    tokenizer_path = "tokenizer.p"

    # ✅ Direct download URL from Hugging Face (NOT blob link)
    model_url = "https://huggingface.co/skatyal1931/CaptionModel/resolve/main/fnl_epoch_45.h5"

    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            gdown.download(model_url, model_path, quiet=False)

    # Load model and tokenizer
    model = load_model(model_path)
    tokenizer = pickle_load(open(tokenizer_path, "rb"))

    # Load feature extractor
    xception_model = Xception(include_top=False, pooling='avg')

    return model, tokenizer, xception_model

# @st.cache_resource
# def load_model_from_url():
#     model_path = "fnl_epoch_45.h5"
#     model_url = st.secrets["model_url"]

#     if not os.path.exists(model_path):
#         with st.spinner("Downloading model from Google Drive..."):
#             urllib.request.urlretrieve(model_url, model_path)

#     return load_model(model_path)


def extract_features(image, model):
    image = image.resize((299, 299)).convert('RGB')
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length=32):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text.replace("start ", "").replace(" end", "")

def text_to_speech(text, filename):
    tts = gTTS(text)
    tts.save(filename)
    return filename

# App UI
st.title("Image Captioning with Audio Output")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        try:
            model, tokenizer, xception_model = load_captioning_model()
            feature = extract_features(image, xception_model)
            caption = generate_desc(model, tokenizer, feature)
            st.subheader("Generated Caption")
            st.write(caption)
        except Exception as e:
            st.error(f"Something went wrong while generating the caption: {e}")


    if st.button("🔊 Listen to Caption"):
        filename = f"temp_audio_{uuid.uuid4()}.mp3"
        audio_path = text_to_speech(caption, filename)
        audio_file = open(audio_path, "rb")
        st.audio(audio_file.read(), format="audio/mp3")
        audio_file.close()
        os.remove(filename)
