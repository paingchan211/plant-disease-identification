"""Plant disease identifier (DPD-Small): Streamlit app of the report's final model.

Model (model.py): Branch 1 (fine-tuned BioCLIP, text tables re-fitted with cVAE synthetic embeddings) ensembled with
Branch 2 (MaPLe, 3 seeds): S = 0.4 * S1 + 0.6 * S2 + 7.414 * [pair is unseen]. Same scoring and preprocessing as the
evaluation, so it reproduces the report (seed 42: seen Top-1 67.9 %, unseen 87.5 %).

The model files are downloaded once from a Hugging Face model repo, set in .streamlit/secrets.toml (HF_REPO, and
HF_TOKEN if the repo is private).

    streamlit run streamlit_app.py
"""
import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from huggingface_hub import snapshot_download

from model import HERE, FinalModel

st.set_page_config(page_title='Plant disease identifier', page_icon=':material/eco:', layout='centered')

FILES = ['branch1_a.pt', 'branch1_b.pt', 'heads.pt', 'thumbs.zip']
nice = lambda s: s.replace('_', ' ')


def secret(name, default=None):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except FileNotFoundError:          # no secrets.toml: fall back to environment variables
        return os.environ.get(name, default)


@st.cache_resource(show_spinner='Loading the model. The first start downloads about 750 MB, so it can take a few minutes.')
def load():
    repo = secret('HF_REPO')
    # ponytail: a local folder also works, for testing before the upload
    folder = repo if Path(repo).is_dir() else snapshot_download(repo, allow_patterns=FILES, token=secret('HF_TOKEN'))
    with zipfile.ZipFile(Path(folder) / 'thumbs.zip') as z:
        thumbs = {n: z.read(n) for n in z.namelist()}
    return FinalModel(folder), thumbs


@st.cache_data(max_entries=64, show_spinner=False)
def predict(data: bytes):
    p, _, _, sim = load()[0].predict(data)
    nn_sim, nn_idx = sim.topk(6)
    return p.tolist(), list(zip(nn_idx.tolist(), nn_sim.tolist()))


if not secret('HF_REPO'):
    st.error('Set HF_REPO (your Hugging Face model repo, e.g. "username/plant-disease-model") in .streamlit/secrets.toml '
             'or in the app settings on Streamlit Cloud.')
    st.stop()

model, thumbs = load()
L = model.labels
UNSEEN = set(L['unseen'])
EXAMPLES = sorted((HERE / 'examples').glob('*.jpg'))

st.title('Plant disease identifier', anchor=False)
st.write('Take or upload a photo of a leaf. The app tells you the **crop** and the **disease**, '
         'including some crop and disease combinations it never saw in training.')

# ---- 1. the photo -------------------------------------------------------------------------------------------------
st.subheader('1. Choose a photo', anchor=False)
up, cam, ex = st.tabs([':material/upload: Upload', ':material/photo_camera: Camera', ':material/image: Try an example'])
with up:
    f = st.file_uploader('Leaf photo', type=['jpg', 'jpeg', 'png', 'webp'], label_visibility='collapsed')
with cam:
    # behind a toggle so the browser only asks for the camera when it is wanted
    c = st.camera_input('Take a photo of the leaf') if st.toggle('Turn on the camera') else None
with ex:
    pick = st.radio('Example photos from the test set', EXAMPLES, index=None,
                    format_func=lambda p: nice(p.name.split('__')[0]).capitalize())
data = (f.getvalue() if f else None) or (c.getvalue() if c else None) or (pick.read_bytes() if pick else None)

if not data:
    st.info('Upload a photo, take one, or pick an example to see the result.', icon=':material/arrow_upward:')
    st.stop()

# ---- 2. the result ------------------------------------------------------------------------------------------------
st.subheader('2. Result', anchor=False)
with st.spinner('Looking at the leaf...'):
    p, nn = predict(data)
order = sorted(range(len(p)), key=lambda i: -p[i])[:5]
best = order[0]
crop, dis = nice(L['crops'][L['pair_crop'][best]]), nice(L['diseases'][L['pair_dis'][best]])

photo, answer = st.columns([2, 3], gap='medium')
photo.image(data, width='stretch')
with answer:
    with st.container(border=True):
        st.metric('Crop', crop.capitalize())
        st.metric('Disease', dis.capitalize())
        st.progress(p[best], text=f'Confidence: {p[best]:.0%}')
        if best in UNSEEN:
            st.badge('New combination: never seen in training', icon=':material/new_releases:', color='orange')

st.write('**Other possible answers**')
st.dataframe(
    pd.DataFrame({'Crop': [nice(L['crops'][L['pair_crop'][i]]).capitalize() for i in order],
                  'Disease': [nice(L['diseases'][L['pair_dis'][i]]) for i in order],
                  'Confidence': [p[i] for i in order],
                  'Seen in training': ['No' if i in UNSEEN else 'Yes' for i in order]}),
    hide_index=True, width='stretch',
    column_config={'Confidence': st.column_config.ProgressColumn(format='percent', min_value=0.0, max_value=1.0)})

# ---- 3. why -------------------------------------------------------------------------------------------------------
st.subheader('3. Similar training photos', anchor=False)
st.caption('The training photos that look most like yours, according to the model.')
for col, (i, s) in zip(st.columns(3) + st.columns(3), nn):
    col.image(thumbs[model.files[i]], width='stretch',
              caption=f'{nice(L["crops"][int(model.kcrop[i])])}, {nice(L["diseases"][int(model.kdis[i])])}')

with st.expander('How it works', icon=':material/info:'):
    st.markdown(
        '- Two models look at the photo and their scores are added together (40% and 60%).\n'
        '- **Model 1** is BioCLIP, a vision and language model for biology, fine-tuned on our training photos. It also '
        'learned from made-up (synthetic) examples of the combinations that have no training photos.\n'
        '- **Model 2** is BioCLIP with learned prompts (MaPLe), averaged over 3 training runs.\n'
        '- There are 333 crop and disease combinations. 10 of them have no training photos; the model works them out '
        'from what it knows about the crop and the disease separately.\n'
        '- On the test set it gets 67.9% of seen and 87.5% of unseen combinations right on the first answer.\n'
        '- The confidence is scaled on validation photos, so it is not over-confident.')
    st.caption('COS30082 Applied Machine Learning, Group 4.')
