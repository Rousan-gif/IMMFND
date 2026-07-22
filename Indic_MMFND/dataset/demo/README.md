

# IMMFND Demo Dataset

This directory contains a **demo subset** of the **IMMFND (Indian Multilingual Multimodal Fake News Dataset)**, released to support transparency, reproducibility, and early experimentation.
The demo dataset reflects the **structure, annotation format, and multilingual characteristics** of the full dataset, while remaining lightweight for quick access and evaluation.

---

##  Overview

The IMMFND demo dataset provides:

* Multilingual news claims covering **13 Indian languages**
* Associated **visual content (images)** for each claim
* Binary **veracity labels** (*Fake* / *Real*)
* Standard **train / validation / test** splits

This subset is intended for:

* Understanding dataset organization
* Testing data loaders and pipelines
* Reproducing baseline experiments
* Reviewing multilingual and multimodal annotations

---

##  Directory Structure

After unzipping the demo dataset archive, the following directory structure is obtained:

```text
IMMFND_Demo_Dataset/
├── train/
│   ├── Fake/
│   │   ├── data.xlsx
│   │   └── images/
│   └── Real/
│       ├── data.xlsx
│       └── images/
│
├── validation/
│   ├── Fake/
│   │   ├── data.xlsx
│   │   └── images/
│   └── Real/
│       ├── data.xlsx
│       └── images/
│
└── test/
    ├── Fake/
    │   ├── data.xlsx
    │   └── images/
    └── Real/
        ├── data.xlsx
        └── images/
```

---

##  File Description

### 1. `data.xlsx`

Each Excel file contains the **textual and metadata information** for the corresponding split and class, including:

* News claim text
* Language
* Veracity label (Fake / Real)
* Image filename or identifier
* Source information

Each row in the file corresponds to **one multimodal news instance**.

### 2. `images/`

This folder contains the **visual evidence** associated with the claims listed in `data.xlsx`.

* Image filenames match the references in the Excel file
* One image corresponds to one claim instance

---

##  Language Coverage

The demo dataset reflects the multilingual nature of IMMFND and includes samples from the following Indian languages:

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

---

##  Important Notes

* This demo dataset is a **representative subset** and does **not** include the full IMMFND data.
* The **full dataset** will be **publicly released after paper acceptance**, in accordance with research and ethical guidelines.
* Dataset splits are **fixed** and should not be mixed to ensure fair evaluation.
* Images have been **cleaned to remove explicit fact-checking watermarks** to prevent label leakage.

---

##  License & Usage

The IMMFND demo dataset is released **strictly for research and educational purposes**.
Users are requested to **cite the corresponding paper** when using this dataset in academic work.

---

##  Citation

If you use the IMMFND dataset or this demo subset, please cite our paper:

```bibtex
@inproceedings{immfnd2025,
  title     = {IMMFND: An Indian Multilingual Multimodal Fake News Dataset with Factual Claim and Evidence Generation},
  author    = {Rousanuzzaman and Kankanala Siva Sai Amrutha and Shreya Ghosh},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2025}
}
```

---

##  Contact

For questions, issues, or collaboration inquiries, please contact:

**Rousanuzzaman**
Indian Institute of Technology Bhubaneswar
 [a24cs09008@iitbbs.ac.in](mailto:a24cs09008@iitbbs.ac.in)

---

