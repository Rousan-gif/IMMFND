 ## IMMFND: Indian Multilingual Multimodal Fake News Dataset

**with Factual Claim and Evidence Generation**

This repository provides the official implementation and resources for **IMMFND**, a large-scale **Indian Multilingual and Multimodal Fake News Dataset**, along with **FactCheck-MM**, a hybrid multimodal verification framework for factual claim verification and evidence generation.

The project targets real-world misinformation in the Indian context, covering **13 Indian languages** and combining **textual and visual modalities**.

---

## Key Contributions

* **IMMFND Dataset**:
  A large-scale dataset with **86,610** fact-checked news samples across **13 Indian languages**, each containing text, image, metadata, and veracity labels.
* **FactCheck-MM Framework**:
  A hybrid multimodal verification pipeline combining:

  * Local semantic vector cache
  * Live web multimodal retrieval
  * Cross-lingual semantic alignment
  * Structured reasoning with large language models
* **Baselines**:
  Includes training and evaluation code for **CLIP-ViT-L/14** as a strong multimodal baseline.

---

##  Supported Languages

The dataset covers the following Indian languages:

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

Together, these languages represent nearly **90% of India’s population**.

---

##  Repository Structure

```
IMMFND/
│
├── dataset_collection/
│   └── scrape_ifcn_sources.py
│
├── factcheck_mm/
│   ├── retrieval.py
│   ├── verification.py
│   └── pipeline.py
│
├── baselines/
│   └── clip_vit_l14/
│       ├── train.py
│       ├── test.py
│       └── README.md
│
├── demo_dataset/
│   └── README.md
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

##  Dataset Access

Due to size constraints, the **demo version of IMMFND** is hosted externally.

* **Demo Dataset (Google Drive)**:
  [https://drive.google.com/file/d/1eqg36sMyrG0XBIRGWkg7YecbB1CSILLO/view](https://drive.google.com/file/d/1eqg36sMyrG0XBIRGWkg7YecbB1CSILLO/view)

The **full dataset** will be made publicly available **after paper acceptance**, in accordance with conference policies.

---

##  FactCheck-MM Overview

**FactCheck-MM** is designed for **live web verification** of multilingual and multimodal claims.
It operates in two stages:

1. **Local Semantic Cache Retrieval**
   Quickly identifies previously verified claims using dense text–image embeddings.
2. **Web Multimodal Retrieval & Reasoning**
   For unseen claims, the system retrieves external textual and visual evidence and synthesizes multilingual explanations through structured reasoning.

---

##  Baseline: CLIP-ViT-L/14

We provide training and evaluation scripts for **CLIP-ViT-L/14**, used as a strong multimodal baseline for comparison with FactCheck-MM.

* Image + Text (Claim) encoding
* End-to-end supervised training
* Standard classification evaluation

---

##  Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Multilingual support requires **XeLaTeX or LuaLaTeX** for proper font rendering.

---

##  Reproducibility

* All experiments follow a **strict blind evaluation protocol**
* Ground-truth evidence is hidden during testing
* Designed to simulate real-world verification of unseen claims

---

##  Citation

If you use this dataset or code, please cite our paper:

```bibtex
@inproceedings{immfnd2026,
  title     = {IMMFND: An Indian Multilingual Multimodal Fake News Dataset with Factual Claim and Evidence Generation},
  author    = {Rousanuzzaman and Amrutha, Kankanala Siva Sai and Ghosh, Shreya},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
}
```

---

##  License

This project is released under the **MIT License**.
See the `LICENSE` file for details.

---

##  Contact

For questions or collaborations:

* **Rousanuzzaman** – [a24cs09008@iitbbs.ac.in](mailto:a24cs09008@iitbbs.ac.in)
* **Amrutha Kankanala Siva Sai** – [21cs02005@iitbbs.ac.in](mailto:21cs02005@iitbbs.ac.in)
* **Shreya Ghosh** – [shreya@iitbbs.ac.in](mailto:shreya@iitbbs.ac.in)


