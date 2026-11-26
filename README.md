# Plant disease identifier (Streamlit app)

Upload or take a photo of a leaf. The app shows the crop and the disease, how confident it is, the other likely
answers, and the six most similar training photos. It runs the report's final model: **Branch 1 + synthetic
embeddings, ensembled with MaPLe** (S = 0.4·S1 + 0.6·S2 + 7.414·[unseen pair]).

| File | What it is |
|---|---|
| `streamlit_app.py` | the interface |
| `model.py` | the model: preprocessing, Branch 1, MaPLe, ensemble (from Andrew's app, unchanged except where the weights are loaded from) |
| `labels.json`, `examples/` | pair, crop and disease names; eight test photos for the "Try it" buttons |
| `hf_upload/` | the model files for Hugging Face (not pushed to GitHub): `branch1_a.pt`, `branch1_b.pt`, `heads.pt`, `thumbs.zip`, about 130 MB |
| `upload_to_hf.py` | uploads `hf_upload/` to a Hugging Face model repo |

Pre-trained BioCLIP (about 600 MB) is downloaded from Hugging Face by open_clip on first start.

## 1. Put the model on Hugging Face

```
pip install huggingface_hub
hf auth login                      # paste a token with WRITE access
python upload_to_hf.py <your-hf-username>/plant-disease-model
```

## 2. Put the app on GitHub

Push this folder (without `hf_upload/` and `.streamlit/secrets.toml`; `.gitignore` already skips them) to a GitHub repo.

## 3. Deploy on Streamlit Community Cloud

1. On https://share.streamlit.io choose **Create app → from GitHub**, pick the repo, and set the main file to
   `streamlit_app.py` (or `<folder>/streamlit_app.py`). Under **Advanced settings** choose Python 3.12.
2. In **Advanced settings → Secrets** paste:
   ```
   HF_REPO = "<your-hf-username>/plant-disease-model"
   ```
   (add `HF_TOKEN = "hf_..."` only if you made the model repo private).
3. Deploy. The first start installs PyTorch and downloads about 750 MB, so it takes a few minutes. After that a
   photo takes about 1 second.
4. If the GitHub repo is private, make the app public in its **Settings → Sharing** so the markers can open the link.

## Run locally

```
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml     # then set HF_REPO (a local folder such as hf_upload also works)
streamlit run streamlit_app.py
```

Checked: the six example photos are all predicted correctly (about 0.6 s per photo on CPU). Andrew's
`check_reproduces.py` on the full test set gave seen Top-1 67.90%, unseen 87.50%, HM 76.46, matching the report.
