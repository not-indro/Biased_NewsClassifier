# Media Bias Classifier

A machine-learning system to detect and visualize linguistic bias in short text (news headlines, tweets, editorials).
Built using a fine-tuned **DistilBERT** model, with an interactive **Streamlit** dashboard and optional **NewsAPI** integration for live news analysis.

---

## Overview

The Media Bias Classifier predicts bias on a graded scale — from strongly biased to strongly unbiased — and provides an intuitive visual interpretation of results.

Key components include:

* Fine-tuned **DistilBERT** (Transformers)
* Interactive **Streamlit** interface
* Optional **live news ingestion** via NewsAPI
* Local inference support (MPS / CPU)

---

Interface Screenshot:
![App Screenshot 1](screenshots/shot1.png)

## Features

* Bias detection for short-form text
* Live headline fetch & real-time classification
* Bias score visualization (gauge-based)
* Model performance dashboard & metrics
* Works offline (local inference) if needed

---

## Dataset

**MBIC — Media Bias Annotation Dataset**
Source: Kaggle

| Metric        |               Value |
| ------------- | ------------------: |
| Total Samples |                1551 |
| Train         |                1240 |
| Test          |                 311 |
| Labels        | Biased / Non-Biased |

---

## Installation

```bash
git clone https://github.com/not-indro/Media_BiasClassifier.git
cd Media_BiasClassifier
```

(Recommended) Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### NewsAPI Key (optional)

Create `.streamlit/secrets.toml`:

```toml
NEWSAPI_KEY = "your_api_key_here"
```

Get a free API key: [https://newsapi.org](https://newsapi.org)

---

## Usage

### Train the model

```bash
python trainbias.py
```

Model checkpoint will be saved in:

```
out/distilbert-mbic-binary/best
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Open in browser:
[http://localhost:8501](http://localhost:8501)

---

## Model Performance

| Metric      | Score |
| ----------- | ----- |
| Accuracy    | 73.6% |
| F1-Macro    | 0.68  |
| F1-Weighted | 0.72  |

**Model:** DistilBERT
**Optimizer:** AdamW
**Learning Rate:** 2e-5
**Epochs:** 4

---

## Project Structure

| Section        | Description                           |
| -------------- | ------------------------------------- |
| Live News      | Fetch & classify trending headlines   |
| Text Input     | Analyze custom text                   |
| Model Insights | Performance metrics, config, examples |

---

## Roadmap

* Multilingual support (XLM-R / MiniLM)
* Bias + sentiment + stance analysis
* Larger, balanced training dataset
* Cloud deployment (HF Spaces / Streamlit Cloud)

---

## Citation

If using this dataset or research:

```
Spinde, T., Hamborg, F., et al. (2020). MBIC: A Media Bias Annotation Dataset.
Kaggle. https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset
```

---

## License

MIT License. See `LICENSE` for details.

---

## Support

If you find this project useful, please give it a ⭐ on GitHub!

---

### ✅ Let me know if you'd like:

* a **minimalist version**
* a **more visual / emoji-enhanced version**
* a **resume-ready project write-up**
* a **Hugging Face model card format**

Happy to tailor it further!
