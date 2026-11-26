# Streamlit app: upload a leaf photo to see the crop and the disease, the top-5 predictions and the most similar
# training photos. It runs the report's final model (model.py): Branch 1 (fine-tuned BioCLIP + synthetic embeddings)
# ensembled with Branch 2 (MaPLe, 3 seeds): S = 0.4 * S1 + 0.6 * S2 + 7.414 * [pair is unseen].
# The model files are too big for GitHub, so they are downloaded from the Hugging Face repo in the HF_REPO secret.
# Run locally with:  streamlit run streamlit_app.py
import base64
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


# Example photos from the test set: (file, name, lab or field, tag). Row 1: four pairs never seen in training;
# row 2: two seen pairs, then two unseen pairs the model is less sure about (62% and 57%, still right).
# Lab / field follows the project's image-source lists (src/data.py), the same split as the report's lab/field table.
NEW, HARD = 'New pair', 'New pair · harder'
EXAMPLES = [('CDDM_tomato_early_blight_89125.jpg', 'Tomato early blight', 'Lab', NEW),
            ('MLBD_mango_anthracnose_423.jpg', 'Mango anthracnose', 'Lab', NEW),
            ('CDRD_cucumber_downy_mildew_474.jpg', 'Cucumber downy mildew', 'Field', NEW),
            ('CDDM_bell_pepper_bacterial_spot_41490.jpg', 'Pepper bacterial spot', 'Lab', NEW),
            ('MTID_maize_streak_virus_14061.jpg', 'Corn streak virus', 'Lab', ''),
            ('LSAN_apple_healthy_1349.JPG', 'Healthy apple', 'Field', ''),
            ('FP_corn_brown_spots_1775.jpg', 'Corn brown spot', 'Field', HARD),
            ('PDN_cherry_aphis_spp_1562.jpg', 'Cherry aphids', 'Field', HARD)]


@st.cache_data
def cards_css():
    """Each example is one Streamlit button drawn as a photo card: the photo is the button's background, the tags are
    its ::before / ::after, and the name sits at the bottom. So the whole card (photo + name) is one click target."""
    css = []
    for i, (file, name, setting, tag) in enumerate(EXAMPLES):
        buf = io.BytesIO()
        ImageOps.fit(Image.open(HERE / 'examples' / file).convert('RGB'), (300, 200)).save(buf, 'JPEG', quality=85)
        url = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
        new = f'"{tag}"' if tag else 'none'
        css.append(f'.st-key-card{i} button {{background-image: url({url}); --tag: {new}; --env: "{setting}";}}')
        # long names get slightly smaller text so every name fits on one line
        css.append(f'.st-key-card{i} button p {{font-size: clamp(0.62rem, {min(9.6, 182 / len(name)):.1f}cqw, 1.1rem);}}')
        if not tag:
            css.append(f'.st-key-card{i} button::before {{display: none;}}')
    return '<style>' + ' '.join(css) + '</style>'


@st.cache_data(max_entries=64)
def square(data: bytes, size=200):
    """Square crop so all photos in a row are the same size."""
    return ImageOps.fit(Image.open(io.BytesIO(data)).convert('RGB'), (size, size))


# CSS: a compact page that fits on one screen, and a centred upload box with a dashed border and a hover highlight.
# It uses Streamlit's internal names, so the Streamlit version is pinned in requirements.txt.
CSS = """
<style>
/* the whole page grows with the screen: 16px on a laptop, up to 20px on a large monitor */
html {font-size: clamp(16px, min(1.05vw, 1.9vh), 20px) !important;}
/* 3.75rem leaves room for Streamlit Cloud's top bar (Share, GitHub), which otherwise covers the title */
.block-container {padding-top: 3.75rem; padding-bottom: 0.5rem;}
h1 {padding: 0 0 0.2rem 0 !important; font-size: clamp(2rem, 2vw, 2.6rem) !important;}
[data-testid="stFileUploaderDropzone"] {
    min-height: clamp(98px, calc(98px + (100vh - 768px) * 0.1), 150px); flex-direction: column; justify-content: center; align-items: center; gap: 6px;
    border: 2px dashed rgba(128, 128, 128, 0.45); transition: border-color 0.2s, background-color 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2e7d32; background-color: rgba(46, 125, 50, 0.08); cursor: pointer;
}
[data-testid="stFileUploaderDropzone"]::before {content: 'Drag and drop a leaf photo here'; font-weight: 600;}
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChip"])::before {content: 'Drop another photo to replace it';}
[data-testid="stFileUploaderDropzoneInstructions"] {text-align: center;}
/* sizes grow with the screen (clamp: smallest, preferred, largest), so big screens get bigger text and photos
   while a 1366x768 laptop still shows the whole page; example names stay on one line */
/* example cards: the photo (3:2) fills the top of the button, the name sits in a strip at the bottom */
.st-key-examples [class*="st-key-card"] {container-type: inline-size;}
.st-key-examples button {
    position: relative; width: 100%; height: calc(66.6cqw + 2.1rem); padding: 0; overflow: hidden;
    align-items: flex-end; border-radius: 0.7rem; border: 1px solid rgba(128, 128, 128, 0.3);
    background-size: 100% 66.6cqw; background-position: top; background-repeat: no-repeat;
    transition: transform 0.15s, box-shadow 0.15s, border-color 0.15s;
}
.st-key-examples button:hover {transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15); border-color: #2e7d32;}
.st-key-examples button [data-testid="stMarkdownContainer"] {width: 100%; height: 2.1rem; display: flex;
    align-items: center; justify-content: center;}
.st-key-examples button p {font-size: clamp(0.62rem, 8.4cqw, 1.1rem); white-space: nowrap; margin: 0;}
.st-key-examples button[kind="primary"], .st-key-examples button[data-testid="stBaseButton-primary"] {
    border: 2px solid #2e7d32; box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.25);}
/* tags on the photo: "New pair" top left, "Lab" / "Field" at the bottom of the photo */
.st-key-examples button::before, .st-key-examples button::after {
    position: absolute; left: 6px; padding: 0.05rem 0.4rem; border-radius: 0.3rem; color: white;
    font-size: clamp(0.65rem, 0.7vw, 0.9rem); font-weight: 600; line-height: 1.45; white-space: nowrap;}
.st-key-examples button::before {content: var(--tag); top: 6px; background: #e65100;}
.st-key-examples button::after {content: var(--env); top: calc(66.6cqw - 1.55rem); background: rgba(0, 0, 0, 0.55);}
.st-key-similar img {height: clamp(70px, calc(70px + (100vh - 768px) * 0.15), 140px); object-fit: cover;}
.st-key-similar [data-testid="stImageCaption"] {font-size: clamp(0.66rem, 0.9vw, 1.05rem); line-height: 1.2; display: -webkit-box;
    -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
    hyphens: auto; word-break: normal; overflow-wrap: normal;}
.bars {display: grid; grid-template-columns: max-content 1fr 3.2rem; gap: 0.35rem 0.6rem; align-items: center;
        font-size: clamp(0.88rem, 0.95vw, 1.15rem); margin-bottom: 0.5rem;}
.bars .track {height: 0.5rem; border-radius: 0.25rem; background: rgba(128, 128, 128, 0.18);}
.bars .fill {height: 100%; border-radius: 0.25rem; background: #2e7d32;}
.bars .pct {text-align: right;}
.bars .new {color: #e65100; font-size: 0.8em;}
/* captions (the grey lines) a bit larger than Streamlit's default */
[data-testid="stCaptionContainer"] {font-size: 0.95rem;}
</style>
"""

# ---------------- the page ----------------
st.markdown(CSS + cards_css(), unsafe_allow_html=True)
st.title('Plant Disease Identifier')
st.caption('Upload a leaf photo to find the crop and the disease, even for pairs never seen in training.')

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
    st.caption('Or try a test photo (**New pair** = never seen in training):')
    with st.container(key='examples', gap='small'):
        cards = list(enumerate(EXAMPLES))
        for row in (cards[:4], cards[4:]):
            for col, (i, (file, name, setting, tag)) in zip(st.columns(4, gap='small'), row):
                # the whole card is the button; the selected example gets the filled (green) style
                col.button(name, key=f'card{i}', width='stretch', on_click=pick_example, args=(file,),
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
                st.warning(f'Best guess: **{name(best)}** ({show_percent(confidence)}), not very sure')
            if best in unseen:
                st.badge('New pair: never seen in training', icon=':material/new_releases:',
                         color='orange')
            st.caption(f"Crop **{nice(L['crops'][bc])}** ({show_percent(min(crop_p[bc], 1.0))}), "
                       f"disease **{nice(L['diseases'][bd])}** ({show_percent(min(dis_p[bd], 1.0))})")

        st.markdown('**Top-5 predictions**')
        bars([(name(i) + (' <span class="new">new pair</span>' if i in unseen else ''), p[i]) for i in top5])

        note = (' <span style="font-size:0.9em;opacity:0.65">(of other pairs)</span>' if best in unseen else '')
        st.markdown('**Most similar training photos**' + note, unsafe_allow_html=True)
        # the crop or disease that matches the answer is shown in green
        bold = lambda text, same: f':green[**{text}**]' if same else text
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
