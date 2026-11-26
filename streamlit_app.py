"""Unseen plant disease identifier (DPD-Small): Streamlit app of the project's final model.

Model (model.py): Branch 1 = BioCLIP with its top 6 blocks fine-tuned (WiSE-FT, soup of 3 seeds), text tables re-fitted
on real + synthetic (cVAE) embeddings of the unseen pairs; Branch 2 = MaPLe prompt learning on frozen BioCLIP (3 seeds).
Score = 0.4 * Branch 1 + 0.6 * Branch 2 + bias for the 10 unseen pairs, all chosen on validation data.
The app scores an image exactly as the evaluation did: on the test set it gives seen Top-1 67.9 %, unseen 87.5 %
(check_reproduces.py).

The model files are downloaded once from a Hugging Face model repo, set as the Streamlit secret HF_REPO
(plus HF_TOKEN if the repo is private).

Input: a leaf / plant photo (upload, camera or example). Output: top-5 crop-disease pairs (unseen ones flagged), the crop
and the disease separately, and the training images it matches (nearest neighbours in Branch 1's embedding space).

    streamlit run streamlit_app.py
"""
import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from huggingface_hub import snapshot_download

from model import HERE, FinalModel

st.set_page_config(page_title='Unseen plant disease identifier', page_icon=':material/eco:', layout='wide')

# test results of this exact model (cVAE seed 42), from Zwe's ensemble_synth run; the report gives the mean of 3 seeds
TEST = {'Seen Top-1': 67.9, 'Unseen Top-1': 87.4, 'Harmonic mean': 76.4, 'Seen Top-5': 91.8, 'Unseen Top-5': 99.6}
nice = lambda s: s.replace('_', ' ')


FILES = ['branch1_a.pt', 'branch1_b.pt', 'heads.pt', 'thumbs.zip']


def secret(name):
    try:
        return st.secrets.get(name, os.environ.get(name))
    except FileNotFoundError:          # no secrets.toml: fall back to environment variables
        return os.environ.get(name)


@st.cache_resource(show_spinner=False)
def model_folder():
    repo = secret('HF_REPO')
    # ponytail: a local folder also works, for testing before the upload
    return Path(repo) if Path(repo).is_dir() else Path(snapshot_download(repo, allow_patterns=FILES, token=secret('HF_TOKEN')))


@st.cache_resource(show_spinner='Loading the model. The first start downloads about 750 MB, so it can take a few minutes.')
def load_model():
    return FinalModel(model_folder())


@st.cache_resource(show_spinner=False)
def load_thumbs():
    with zipfile.ZipFile(model_folder() / 'thumbs.zip') as z:
        return {n: z.read(n) for n in z.namelist()}


@st.cache_data(max_entries=64, show_spinner=False)
def predict(data: bytes):
    p, _, _, sim = load_model().predict(data)
    nn_sim, nn_idx = sim.topk(6)
    return p.tolist(), list(zip(nn_idx.tolist(), nn_sim.tolist()))


def example_label(path):
    pair = path.name.split('__')[0]
    return nice(pair).capitalize() + (' (unseen pair)' if pair in UNSEEN_NAMES else '')


if not secret('HF_REPO'):
    st.error('Set HF_REPO (your Hugging Face model repo, e.g. "username/plant-disease-model") in the app secrets.')
    st.stop()

model = load_model()
L = model.labels
UNSEEN = set(L['unseen'])
UNSEEN_NAMES = {L['pairs'][i] for i in UNSEEN}
EXAMPLES = sorted((HERE / 'examples').glob('*.jpg'))

st.title('Unseen plant disease identifier', anchor=False)
st.caption('Upload a photo of a leaf or plant. The model names the crop and the disease, including crop-disease '
           'combinations it never saw during training, and shows the training images it matches.')

left, right = st.columns([2, 3], gap='large')
with left:
    source = st.segmented_control('Image source', ['Upload', 'Camera', 'Example'], default='Upload', required=True, key='source',
                                  format_func=lambda s: {'Upload': ':material/upload: Upload',
                                                         'Camera': ':material/photo_camera: Camera',
                                                         'Example': ':material/image: Example'}[s])
    data = None
    if source == 'Upload':
        f = st.file_uploader('Plant photo', type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
        data = f.getvalue() if f else None
    elif source == 'Camera':
        f = st.camera_input('Take a photo of the leaf')
        data = f.getvalue() if f else None
    else:
        ex = st.pills('Test images (file name = true label)', EXAMPLES, format_func=example_label, default=EXAMPLES[2],
                     key='example')
        data = ex.read_bytes() if ex else None
    if data:
        st.image(data, width=360, alt='The photo being identified')

with right:
    if not data:
        st.info('Choose a photo on the left to identify the crop and the disease.', icon=':material/eco:')
    else:
        with st.spinner('Identifying (about 1 to 3 seconds)...'):
            p, nn = predict(data)
        order = sorted(range(len(p)), key=lambda i: -p[i])[:5]
        best = order[0]
        c, d = L['pair_crop'][best], L['pair_dis'][best]
        with st.container(border=True):
            st.subheader(f'{nice(L["crops"][c]).capitalize()}: {nice(L["diseases"][d])}', anchor=False)
            if best in UNSEEN:
                st.badge('Combination never seen in training', icon=':material/new_releases:', color='orange')
                st.caption('The model composed this answer from what it learned about this crop with other diseases '
                           'and about this disease on other crops.')
            st.dataframe(
                pd.DataFrame({'Crop - disease': [nice(L['pairs'][i]) for i in order],
                              'Confidence': [p[i] for i in order],
                              'In training': ['no (unseen pair)' if i in UNSEEN else 'yes' for i in order]}),
                hide_index=True, alt='Top-5 crop-disease predictions with confidence',
                column_config={'Confidence': st.column_config.ProgressColumn(format='percent', min_value=0.0, max_value=1.0)})
            st.caption('Confidence is calibrated on validation images (temperature scaling). Crop and disease below '
                       'add up the confidence of every pair with that crop or that disease.')
        # crop and disease separately: the final model's pair probabilities summed per crop / per disease
        crop_p, dis_p = [0.0] * len(L['crops']), [0.0] * len(L['diseases'])
        for i, q in enumerate(p):
            crop_p[L['pair_crop'][i]] += q
            dis_p[L['pair_dis'][i]] += q
        cc, dc = st.columns(2)
        for col, probs, names, title in ((cc, crop_p, L['crops'], 'Crop'), (dc, dis_p, L['diseases'], 'Disease')):
            top = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
            col.dataframe(pd.DataFrame({title: [nice(names[i]) for i in top], 'Probability': [probs[i] for i in top]}),
                          hide_index=True, alt=f'Top-3 {title.lower()}',
                          column_config={'Probability': st.column_config.ProgressColumn(format='percent', min_value=0.0,
                                                                                        max_value=1.0)})

if data:
    st.subheader('Most similar training images', anchor=False)
    thumbs, mem = load_thumbs(), model
    cols = st.columns(6)
    for col, (i, s) in zip(cols, nn):
        cap = f'{nice(L["crops"][int(mem.kcrop[i])])} · {nice(L["diseases"][int(mem.kdis[i])])} (similarity {s:.2f})'
        col.image(thumbs[mem.files[i]], caption=cap, width='stretch', alt=cap)

with st.expander('How it works', icon=':material/info:'):
    st.markdown(
        '- **Branch 1**: the biology vision-language model BioCLIP with its top 6 of 12 blocks fine-tuned on the 323 seen '
        'crop-disease pairs, merged with the pre-trained weights (WiSE-FT) and averaged over 3 training runs. Its text '
        'tables are re-fitted on synthetic image embeddings (conditional VAE) of the 10 unseen pairs.\n'
        '- **Branch 2**: MaPLe prompt learning on frozen BioCLIP, 3 seeds.\n'
        '- **Score** = 0.4 × Branch 1 + 0.6 × Branch 2 + a bias for the unseen pairs. The weight and the bias were '
        'chosen on validation images; the test set chose nothing.\n'
        '- The similar training images are the nearest neighbours of the photo in Branch 1\'s embedding space.')
    st.dataframe(pd.DataFrame([TEST]), hide_index=True, alt='Test results of this model (%)',
                 column_config={k: st.column_config.NumberColumn(format='%.1f %%') for k in TEST})
    st.caption('Test results of this exact model (DPD-Small, 2,000 seen and 1,448 unseen test images). '
               'COS30082 Applied Machine Learning, group project.')
