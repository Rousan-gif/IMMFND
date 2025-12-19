

# Dataset Collection

**IMMFND: Indian Multilingual Multimodal Fake News Dataset**

This directory contains the **data collection and preprocessing pipeline** used to construct the **IMMFND** dataset. The scripts automatically collect **fact-checked fake news** and **verified real news** from reliable Indian sources, covering multiple languages and modalities.

---

##  Objective

The goal of this module is to:

* Collect **fake news** from verified Indian fact-checking organizations
* Collect **real news** from trusted Indian media outlets
* Extract **textual claims, images, and metadata**
* Build a **multilingual and multimodal** dataset suitable for misinformation research

---

##  Data Sources

###  Fake News Sources

Fake news samples are collected exclusively from **IFCN-certified Indian fact-checking organizations**, ensuring reliability and editorial rigor. These include:

* Alt News
* BoomLive
* Factly
* Vishvas News
* AFP India
* Other IFCN-certified Indian platforms

These organizations follow the **International Fact-Checking Network (IFCN) Code of Principles**, ensuring transparency, fairness, and evidence-based verification.

---

###  Real News Sources

Real news samples are collected from well-established and credible Indian news outlets, such as:

* ABP News
* News18
* Aaj Tak
* The Indian Express
* NDTV

Each real article is manually cross-verified to ensure factual authenticity.

---

##  Language Coverage

The dataset supports **13 major Indian languages**, covering nearly 90% of India’s population:

* English
* Hindi
* Bengali
* Tamil
* Telugu
* Kannada
* Malayalam
* Marathi
* Gujarati
* Punjabi
* Urdu
* Odia
* Assamese

This multilingual coverage enables robust research on **cross-lingual misinformation**.

---

##  Directory Structure

```
dataset_collection/
│
├── scrape_factcheck.py    # Collects fake news from fact-checking sites
├── scrape_realnews.py     # Collects real news from media outlets
├── parse_articles.py      # Extracts claims, text, images, metadata
├── clean_images.py        # Removes watermarks, logos, and bias cues
├── deduplicate.py         # Removes duplicate or near-duplicate samples
├── language_detect.py     # Language identification and validation
├── save_dataset.py        # Stores data in structured format (CSV/JSON)
└── README.md
```

---

##  Collection Methodology

### 1️ Web Scraping

* Implemented using **BeautifulSoup** and **Requests**
* Crawls fact-checking and news websites periodically
* Extracts article title, claim, publication date, category, and image URLs

### 2️ Image Processing

* Downloads associated images
* Removes visible *“fake”* labels, stamps, or watermarks
* Ensures images are unbiased and suitable for model training

### 3️ Metadata Extraction

Each sample includes:

* Source website
* Article URL
* Title and claim
* Full article content
* Image path
* Language label
* Veracity label (Real / Fake)

---

##  Deduplication and Noise Filtering

To ensure dataset quality:

* Exact and near-duplicate articles are removed
* Duplicate images are filtered using hash-based similarity
* Broken links and missing media are discarded
* Incomplete or noisy samples are excluded

---

##  Annotation and Labeling Protocol

* **Fake labels** are inherited directly from fact-checking platforms, each supported by evidence and explanation
* **Real labels** are assigned only after manual verification
* All samples undergo consistency checks for language, modality, and metadata alignment

This ensures **high-quality, trustworthy annotations**.

---

##  Output Format

The final dataset is stored in structured formats:

* CSV (for metadata and labels)
* JSON (for multimodal alignment)
* Image directories (JPEG/PNG)

A **demo subset** of the dataset is publicly available:

```
https://drive.google.com/file/d/1eqg36sMyrG0XBIRGWkg7YecbB1CSILLO/view
```

---

##  Ethical Considerations

* All data is collected from **publicly accessible sources**
* No private or personal user data is included
* Dataset is intended **strictly for research purposes**

---

##  Contact

For questions regarding dataset collection:

* **Rousanuzzaman** – [a24cs09008@iitbbs.ac.in](mailto:a24cs09008@iitbbs.ac.in)

---

