
# CLIP-ViT-L/14 Baseline

**Multimodal Fake News Classification on IMMFND**

This directory contains the **baseline implementation of CLIP-ViT-L/14** used for multimodal fake news classification on the **IMMFND** dataset. The baseline jointly encodes **textual claims** and **associated images** and performs supervised classification into **Real** and **Fake/Misleading** classes.

This model serves as a **strong representation-level baseline** for comparison with the proposed **FactCheck-MM** verification framework.

---

##  Model Overview

* **Architecture**: CLIP-ViT-L/14
* **Input Modalities**:

  * Image
  * Text (Claim)
* **Output**:

  * Binary classification: *Real* / *Fake*
* **Training Objective**:

  * Supervised cross-entropy loss on IMMFND labels

---

##  Directory Structure

```
clip_vit_l14/
│
├── train.py        # Training script
├── test.py         # Evaluation script
├── utils.py        # Data loading and helper functions
├── config.yaml     # Training configuration
└── README.md
```

---

##  Dataset

This baseline is trained and evaluated using the **IMMFND** dataset.

* **Training split**: IMMFND (train)
* **Validation split**: IMMFND (validation)
* **Test split**: IMMFND (test)

A **demo version of the dataset** is available at:

```
https://drive.google.com/file/d/1eqg36sMyrG0XBIRGWkg7YecbB1CSILLO/view
```

---

##  Requirements

Install required dependencies before running the code:

```bash
pip install -r ../../requirements.txt
```

Key dependencies:

* PyTorch
* torchvision
* transformers
* open_clip
* PIL
* numpy
* scikit-learn

---

##  Training

To train the CLIP-ViT-L/14 baseline on IMMFND:

```bash
python train.py --config config.yaml
```

The training script:

* Loads image–claim pairs
* Encodes them using CLIP-ViT-L/14
* Trains a classification head on top of joint embeddings

---

##  Evaluation

To evaluate the trained model on the test split:

```bash
python test.py --checkpoint path/to/model.pt
```

Reported metrics:

* Accuracy
* Precision
* Recall
* F1-score (macro and class-wise)

---

##  Baseline Performance

On the IMMFND test split, the CLIP-ViT-L/14 baseline achieves:

* **Accuracy**: 89.26%
* **Balanced precision and recall** across Real and Fake classes

While CLIP-ViT-L/14 demonstrates strong multimodal classification capability, it relies purely on **representation-level pattern matching** and does not perform **explicit evidence retrieval or reasoning**, which motivates the proposed **FactCheck-MM** framework.

---

##  Limitations

* No explicit fact verification or evidence retrieval
* No cross-lingual reasoning beyond CLIP’s embedding space
* Cannot generate factual corrections or explanations

These limitations highlight the need for **hybrid retrieval-augmented verification**, as addressed by **FactCheck-MM**.

---

##  Reference

If you use this baseline, please cite:

```bibtex
@inproceedings{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and others},
  booktitle={Proceedings of ICML},
  year={2021}
}
```

---

##  Contact

For questions regarding this baseline:

* **Rousanuzzaman** – [a24cs09008@iitbbs.ac.in](mailto:a24cs09008@iitbbs.ac.in)

----
