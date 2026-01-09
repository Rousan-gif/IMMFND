
# **IMMFND**

**IMMFND (Indian Multilingual Multimodal Fake News Dataset)** is a large-scale benchmark for fake news detection in the Indian information ecosystem.
It is accompanied by **FactCheck-MM**, a hybrid multimodal verification framework designed for **factual claim validation and evidence generation** across multiple Indian languages.

---

## **Repository Contents**

This repository provides:

* A **multilingual and multimodal fake news dataset** covering **13 Indian languages**
* The complete **FactCheck-MM verification framework**
* A **demo dataset** and **inference pipeline** for evaluation and testing

---

## **Dataset**

A **demo version** of the dataset is available at:

📥 [https://drive.google.com/file/d/1eqg36sMyrG0XBIRGWkg7YecbB1CSILLO/view](https://drive.google.com/file/d/1eqg36sMyrG0XBIRGWkg7YecbB1CSILLO/view)

The **full dataset** will be released after the paper is officially accepted.

---

## **Code**

This repository contains the full implementation of **FactCheck-MM**, including:

* Multimodal evidence retrieval
* Claim verification
* Performance evaluation tools

---

## **Prerequisites**

Before running the system, you will need:

* **OpenAI API Key**
* **SerpAPI Key** (for Google Search–based evidence retrieval)

---

## **Installation**

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root directory and add your API keys:

```
OPENAI_API_KEY=your_key_here
SERPAPI_KEY=your_key_here
```

---

## **Running FactCheck-MM**

### **Option A: Using Default File Names**

If your input file is named `input_claims.xlsx` and is located in the project root:

```bash
python main.py
```

---

### **Option B: Using Custom File Names or Paths**

If your input file is stored elsewhere or has a different name:

```bash
python main.py --input data/my_experiment_data.xlsx --output data/results.xlsx
```

---

## **Evaluation**

After generating predictions, compute the performance metrics:

```bash
python evaluate.py --file data/results.xlsx
```












