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

# first, before the slow imports below, so the browser tab shows the app's name straight away
st.set_page_config(page_title='Plant Disease Identifier', page_icon='🌿', layout='wide')

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


def bars(rows):
    """One compact line per prediction: name, bar, percentage (less height than st.progress)."""
    html = ''.join(f'<span>{label}</span><div class="track"><div class="fill" style="width:{100 * min(q, 1.0):.1f}%">'
                   f'</div></div><span class="pct">{show_percent(q)}</span>' for label, q in rows)
    st.markdown(f'<div class="bars">{html}</div>', unsafe_allow_html=True)


def show_percent(p):
    """Formats a probability, capped at 99%: no model is ever certain, so the app never claims 100% or '>99%'."""
    return f'{min(p, 0.99):.0%}'


# Example photos from the test set (two rows of four): 4 pairs never seen in training, 2 seen ones, 2 tricky ones
EXAMPLES = [('CDDM_tomato_early_blight_89125.jpg', 'Tomato early blight'),
            ('MLBD_mango_anthracnose_423.jpg', 'Mango anthracnose'),
            ('CDRD_cucumber_downy_mildew_474.jpg', 'Cucumber mildew'),
            ('CDDM_bell_pepper_bacterial_spot_41490.jpg', 'Pepper bacterial spot'),
            ('MTID_maize_streak_virus_14061.jpg', 'Corn streak virus'),
            ('LSAN_apple_healthy_1349.JPG', 'Healthy apple'),
            ('FP_corn_brown_spots_1775.jpg', 'Corn brown spot'),
            ('PDN_cherry_aphis_spp_1562.jpg', 'Cherry aphids')]


@st.cache_data(max_entries=64)
def square(data: bytes, size=200):
    """Square crop so all photos in a row are the same size."""
    return ImageOps.fit(Image.open(io.BytesIO(data)).convert('RGB'), (size, size))


# CSS: a compact page that fits on one screen, and a centred upload box with a dashed border and a hover highlight.
# It uses Streamlit's internal names, so the Streamlit version is pinned in requirements.txt.
CSS = """
<style>
/* 3.75rem leaves room for Streamlit Cloud's top bar (Share, GitHub), which otherwise covers the title */
.block-container {padding-top: 3.75rem; padding-bottom: 0.5rem;}
h1 {padding: 0 0 0.2rem 0 !important; font-size: 2rem !important;}
[data-testid="stFileUploaderDropzone"] {
    min-height: 98px; flex-direction: column; justify-content: center; align-items: center; gap: 6px;
    border: 2px dashed rgba(128, 128, 128, 0.45); transition: border-color 0.2s, background-color 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2e7d32; background-color: rgba(46, 125, 50, 0.08); cursor: pointer;
}
[data-testid="stFileUploaderDropzone"]::before {content: 'Drag and drop a leaf photo here'; font-weight: 600;}
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChip"])::before {content: 'Drop another photo to replace it';}
[data-testid="stFileUploaderDropzoneInstructions"] {text-align: center;}
/* example buttons carry the photo names: smaller text so the names fit on one line */
.st-key-examples button p {font-size: 0.7rem; white-space: nowrap;}
.st-key-examples button {min-height: 1.8rem; padding: 0 0.2rem;}
/* photos are cropped to a fixed height so the whole page fits on one screen */
.st-key-examples img {height: 82px; object-fit: cover;}
.st-key-similar img {height: 70px; object-fit: cover;}
.st-key-similar [data-testid="stImageCaption"] {font-size: 0.68rem; line-height: 1.2; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;}
.bars {display: grid; grid-template-columns: max-content 1fr 3.2rem; gap: 0.35rem 0.6rem; align-items: center;
        font-size: 0.85rem; margin-bottom: 0.5rem;}
.bars .track {height: 0.5rem; border-radius: 0.25rem; background: rgba(128, 128, 128, 0.18);}
.bars .fill {height: 100%; border-radius: 0.25rem; background: #2e7d32;}
.bars .pct {text-align: right;}
.bars .new {color: #e65100; font-size: 0.75rem;}
</style>
"""

# ---------------- the page ----------------
st.markdown(CSS, unsafe_allow_html=True)
st.title('Plant Disease Identifier')
st.caption('Upload a photo of a leaf and the model will name the crop and the disease, out of 333 crop and disease '
           'pairs in DPD-Small. It can also recognise 10 pairs that it never saw in training.')

if not secret('HF_REPO'):
    st.error('Set HF_REPO (your Hugging Face model repo, e.g. "username/plant-disease-model") in the app secrets.')
    st.stop()

# Kept between reruns: which photo is shown ('upload', 'camera' or an example file), and a key used to reset the inputs
st.session_state.setdefault('source', None)
st.session_state.setdefault('upload_key', 0)


def pick_example(file):
    """Shows the clicked example and clears the upload box and the camera."""
    st.session_state.source = file
    st.session_state.upload_key += 1   # a new key gives an empty upload box and camera


# Input on the left, results on the right. height='stretch' keeps both boxes the same height.
left, right = st.columns(2, gap='medium')
with left, st.container(border=True, height='stretch', gap='small'):
    uploaded = st.file_uploader('Your leaf photo', type=['jpg', 'jpeg', 'png', 'webp'], label_visibility='collapsed',
                                key=f'upload{st.session_state.upload_key}',
                                on_change=lambda: st.session_state.update(source='upload'))
    # the camera stays off until asked for, so the browser does not ask for camera access when the page opens
    photo = None
    if st.toggle('Take a photo with my camera', key=f'cam_on{st.session_state.upload_key}'):
        photo = st.camera_input('Take a photo of the leaf', key=f'camera{st.session_state.upload_key}',
                                label_visibility='collapsed',
                                on_change=lambda: st.session_state.update(source='camera'))
    st.caption('Or try a photo from the test set. The first four are pairs never seen in training, '
               'the last two are tricky ones.')
    with st.container(key='examples', gap='small'):
        for row in (EXAMPLES[:4], EXAMPLES[4:]):
            for col, (file, name) in zip(st.columns(4, gap='small'), row):
                col.image(square((HERE / 'examples' / file).read_bytes()), width='stretch')
                # the selected example gets a filled button
                col.button(name, key=file, width='stretch', on_click=pick_example, args=(file,),
                           type='primary' if st.session_state.source == file else 'secondary')

# The photo to identify (None until a photo is uploaded, taken or an example is picked)
data = None
if st.session_state.source == 'upload' and uploaded is not None:
    data = uploaded.getvalue()   # EXIF rotation is not applied, as in training and evaluation
elif st.session_state.source == 'camera' and photo is not None:
    data = photo.getvalue()
elif st.session_state.source not in (None, 'upload', 'camera'):
    data = (HERE / 'examples' / st.session_state.source).read_bytes()

with right, st.container(border=True, height='stretch', gap='small'):
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

        # the crop and the disease on their own: the pair confidences added up per crop and per disease
        crop_p, dis_p = [0.0] * len(L['crops']), [0.0] * len(L['diseases'])
        for i, q in enumerate(p):
            crop_p[L['pair_crop'][i]] += q
            dis_p[L['pair_dis'][i]] += q
        bc, bd = L['pair_crop'][best], L['pair_dis'][best]

        pic, verdict = st.columns([1, 4], gap='small')
        pic.image(square(data), width='stretch')
        with verdict:
            # Green if the model is more likely right than wrong (over 50%), amber if it is only guessing
            if confidence > 0.5:
                st.success(f'**{name(best)}** ({show_percent(confidence)} confident)')
            else:
                st.warning(f'Best guess: **{name(best)}** ({show_percent(confidence)}). '
                           'Not very sure, so check the other options too.')
            if best in unseen:
                st.badge('This crop and disease pair was never seen in training', icon=':material/new_releases:',
                         color='orange')
            st.caption(f"On their own: crop **{nice(L['crops'][bc])}** ({show_percent(min(crop_p[bc], 1.0))}), "
                       f"disease **{nice(L['diseases'][bd])}** ({show_percent(min(dis_p[bd], 1.0))})")

        st.markdown('**Top-5 predictions**')
        bars([(name(i) + (' <span class="new">never seen in training</span>' if i in unseen else ''), p[i]) for i in top5])

        note = ('this pair has no training photos, so these are the closest of other pairs; ' if best in unseen else '')
        st.markdown('**Most similar training photos** <span style="font-size:0.8rem;opacity:0.6">'
                    f'({note}bold = same crop or disease as the answer)</span>', unsafe_allow_html=True)
        bold = lambda text, same: f'**{text}**' if same else text
        pics = load_thumbs()
        with st.container(key='similar'):
            for col, i in zip(st.columns(6, gap='small'), nn):
                c, d = int(model.kcrop[i]), int(model.kdis[i])
                col.image(square(pics[model.files[i]], 128), width='stretch',
                          caption=bold(nice(L['crops'][c]).capitalize(), c == bc) + ', ' + bold(nice(L['diseases'][d]), d == bd))

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
        "- The similar training photos are the nearest neighbours of your photo in Branch 1's embedding space\n"
        '- The crop and disease on their own add up the confidence of every pair with that crop or disease\n'
        '- COS30082 Applied Machine Learning, Group 4')
