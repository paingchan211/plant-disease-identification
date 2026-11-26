# Streamlit app: upload a leaf photo to see the crop and the disease, the top-5 predictions and the most similar
# training photos. It runs the report's final model (model.py): Branch 1 (fine-tuned BioCLIP + synthetic embeddings)
# ensembled with Branch 2 (MaPLe, 3 seeds): S = 0.4 * S1 + 0.6 * S2 + 7.414 * [pair is unseen].
# The model files are too big for GitHub, so they are downloaded from the Hugging Face repo in the HF_REPO secret.
# Run locally with:  streamlit run streamlit_app.py
import io
import os
import zipfile
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

from model import HERE, FinalModel

FILES = ['branch1_a.pt', 'branch1_b.pt', 'heads.pt', 'thumbs.zip']
# Test results of this exact model (cVAE seed 42); the report gives the mean of 3 seeds
TEST = dict(seen=67.9, unseen=87.4, hm=76.4, seen5=91.8, unseen5=99.6)
nice = lambda s: s.replace('_', ' ')


def secret(name):
    try:
        return st.secrets.get(name, os.environ.get(name))
    except FileNotFoundError:   # no secrets.toml: fall back to environment variables
        return os.environ.get(name)


@st.cache_resource(show_spinner=False)
def model_folder():
    repo = secret('HF_REPO')
    # ponytail: a local folder also works, for testing before the upload
    return Path(repo) if Path(repo).is_dir() else Path(snapshot_download(repo, allow_patterns=FILES, token=secret('HF_TOKEN')))


# Streamlit reruns the script on every click, so cache the model to load it only once
@st.cache_resource(show_spinner='Loading the model. The first start downloads about 750 MB, so it can take a few minutes.')
def load_model():
    return FinalModel(model_folder())   # runs on CPU, Streamlit Cloud has no GPU


@st.cache_resource(show_spinner=False)
def load_thumbs():
    with zipfile.ZipFile(model_folder() / 'thumbs.zip') as z:
        return {n: z.read(n) for n in z.namelist()}


@st.cache_data(max_entries=64, show_spinner=False)
def predict(data: bytes):
    """Probabilities for all 333 crop-disease pairs and the 6 most similar training photos."""
    p, _, _, sim = load_model().predict(data)
    nn_sim, nn_idx = sim.topk(6)
    return p.tolist(), nn_idx.tolist()


def show_percent(p):
    """Formats a probability, showing '>99%' instead of 100%."""
    return '>99%' if p >= 0.995 else f'{p:.0%}'


# Example photos from the test set (two rows of four). The first four are pairs the model never saw in training.
EXAMPLES = [('CDDM_tomato_early_blight_89125.jpg', 'Tomato early blight'),
            ('MLBD_mango_anthracnose_423.jpg', 'Mango anthracnose'),
            ('CDRD_cucumber_downy_mildew_474.jpg', 'Cucumber downy mildew'),
            ('CDDM_bell_pepper_bacterial_spot_41490.jpg', 'Pepper bacterial spot'),
            ('MTID_maize_streak_virus_14061.jpg', 'Corn streak virus'),
            ('LSAN_apple_healthy_1349.JPG', 'Healthy apple'),
            ('FP_corn_brown_spots_1775.jpg', 'Tricky: Corn brown spot'),
            ('PDN_cherry_aphis_spp_1562.jpg', 'Tricky: Cherry aphids')]


@st.cache_data
def thumbnail(file):
    """Square crop so all thumbnails are the same size."""
    return ImageOps.fit(Image.open(HERE / 'examples' / file).convert('RGB'), (300, 300))


# CSS for a bigger, centred upload box with a dashed border and a hover highlight.
# It uses Streamlit's internal names, so the Streamlit version is pinned in requirements.txt.
CSS = """
<style>
[data-testid="stFileUploaderDropzone"] {
    min-height: 170px; flex-direction: column; justify-content: center; align-items: center; gap: 10px;
    border: 2px dashed rgba(128, 128, 128, 0.45); transition: border-color 0.2s, background-color 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2e7d32; background-color: rgba(46, 125, 50, 0.08); cursor: pointer;
}
[data-testid="stFileUploaderDropzone"]::before {content: 'Drag and drop a leaf photo here'; font-weight: 600;}
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChip"])::before {content: 'Drop another photo to replace it';}
[data-testid="stFileUploaderDropzoneInstructions"] {text-align: center;}
</style>
"""

# ---------------- the page ----------------
st.set_page_config(page_title='Plant Disease Identifier', page_icon='🌿', layout='wide')
st.markdown(CSS, unsafe_allow_html=True)
st.title('Plant Disease Identifier')
st.write('Upload a photo of a leaf and the model will name the crop and the disease, out of 333 crop and disease '
         'pairs in DPD-Small. It can also recognise 10 pairs that it never saw in training.')

if not secret('HF_REPO'):
    st.error('Set HF_REPO (your Hugging Face model repo, e.g. "username/plant-disease-model") in the app secrets.')
    st.stop()

# Kept between reruns: which photo is shown ('upload' or an example file), and a key used to reset the upload box
st.session_state.setdefault('source', None)
st.session_state.setdefault('upload_key', 0)


def pick_example(file):
    """Shows the clicked example and clears the upload box."""
    st.session_state.source = file
    st.session_state.upload_key += 1   # a new key gives an empty upload box


# Input on the left, results on the right. height='stretch' keeps both boxes the same height.
left, right = st.columns(2, gap='medium')
with left, st.container(border=True, height='stretch'):
    uploaded = st.file_uploader('Your leaf photo', type=['jpg', 'jpeg', 'png', 'webp'],
                                key=f'upload{st.session_state.upload_key}',
                                on_change=lambda: st.session_state.update(source='upload'))
    st.caption('Or try one of these photos from the test set (the first four are pairs never seen in training):')
    for row in (EXAMPLES[:4], EXAMPLES[4:]):
        # photos with their names, then the buttons in their own row so they line up when a name wraps
        for col, (file, name) in zip(st.columns(4), row):
            col.image(thumbnail(file), width='stretch', caption=name)
        for col, (file, name) in zip(st.columns(4), row):
            # the selected example gets a filled button
            col.button('Try it', key=file, width='stretch', on_click=pick_example, args=(file,),
                       type='primary' if st.session_state.source == file else 'secondary')

# The photo to identify (None until a photo is uploaded or an example is picked)
data = None
if st.session_state.source == 'upload' and uploaded is not None:
    data = uploaded.getvalue()
elif st.session_state.source not in (None, 'upload'):
    data = (HERE / 'examples' / st.session_state.source).read_bytes()

with right, st.container(border=True, height='stretch'):
    if data is None:
        # placeholder message, centred in the box
        with st.container(height='stretch', vertical_alignment='center', horizontal_alignment='center'):
            st.markdown('#### No photo yet')
            st.caption('Upload a leaf photo or pick one of the examples, and the prediction will appear here.')
    else:
        with st.spinner('Identifying the disease...'):   # the first run also downloads the model
            model = load_model()
            p, nn = predict(data)
        L, unseen = model.labels, set(model.labels['unseen'])
        top5 = sorted(range(len(p)), key=lambda i: -p[i])[:5]
        name = lambda i: f"{nice(L['crops'][L['pair_crop'][i]]).capitalize()}: {nice(L['diseases'][L['pair_dis'][i]])}"
        best, confidence = top5[0], p[top5[0]]

        # Green if the model is more likely right than wrong (over 50%), amber if it is only guessing
        if confidence > 0.5:
            st.success(f'This looks like **{name(best)}** ({show_percent(confidence)} confident)')
        else:
            st.warning(f'Best guess: **{name(best)}** ({show_percent(confidence)}). '
                       'Not very sure, so check the other options below too.')
        if best in unseen:
            st.badge('This crop and disease pair was never seen in training', icon=':material/new_releases:', color='orange')

        st.markdown('**Top-5 predictions**')
        for i in top5:
            st.progress(p[i], text=f"{name(i)}: {show_percent(p[i])}" + ('  (never seen in training)' if i in unseen else ''))

        st.markdown('**Most similar training photos**')
        a, b = st.columns(2)   # the photo and its closest training photos side by side
        a.image(data, caption='Your photo', width='stretch')
        with b:
            pics = load_thumbs()
            for row in (nn[:3], nn[3:]):
                for col, i in zip(st.columns(3), row):
                    col.image(ImageOps.fit(Image.open(io.BytesIO(pics[model.files[i]])), (128, 128)), width='stretch',
                              caption=f"{nice(L['crops'][int(model.kcrop[i])])}, {nice(L['diseases'][int(model.kdis[i])])}")

with st.expander('About the model'):
    st.markdown(
        '- Two models look at the photo and their scores are combined (40% and 60%):\n'
        '  - **Branch 1**: BioCLIP, a vision and language model for biology, with its top 6 of 12 blocks fine-tuned on '
        'the training photos and averaged over 3 runs. Its text features were re-fitted with synthetic (cVAE) examples '
        'of the 10 pairs that have no training photos.\n'
        '  - **Branch 2**: MaPLe prompt learning on frozen BioCLIP, averaged over 3 seeds.\n'
        '- A bias for the 10 unseen pairs and the 40/60 weight were chosen on validation photos; the test set chose nothing.\n'
        f"- Test Top-1 accuracy: **{TEST['seen']}%** on seen pairs and **{TEST['unseen']}%** on unseen pairs "
        f"(harmonic mean **{TEST['hm']}%**); top-5: **{TEST['seen5']}%** / **{TEST['unseen5']}%**\n"
        '- The similar training photos are the nearest neighbours of your photo in Branch 1\'s embedding space\n'
        '- COS30082 Applied Machine Learning, Group 4')
