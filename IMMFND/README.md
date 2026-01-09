**IMMFND**



IMMFND is an Indian Multilingual and Multimodal Fake News Dataset accompanied by

FactCheck-MM, a hybrid multimodal verification framework for factual claim and

evidence generation.



**Contents**

\- Multilingual multimodal fake news dataset (13 Indian languages)

\- FactCheck-MM verification framework

\- Demo dataset and inference scripts



**Dataset**

A demo version of the dataset is available here:

https://drive.google.com/file/d/1eqg36sMyrG0XBIRGWkg7YecbB1CSILLO/view



The full dataset will be released upon paper acceptance.



**Code**

This repository contains the implementation of FactCheck-MM.


**Prerequisites**
OpenAI API Key
SerpApi Key (for Google Search results)

**Installation**

```bash

pip install -r requirements.txt

```

Create a .env file in the root directory. Add your API keys

**Option A**: Using Default Names
If your file is named input_claims.xlsx and is in the same folder as the script, simply run:

```bash
python main.py
```

**Option B**: Using Custom Names/Paths
If your file is named differently or located in a subfolder, use the --input and --output flags:

```bash
python main.py --input data/my_experiment_data.xlsx --output data/results.xlsx
```

To calculate performance metrics, run:
```bash
python evaluate.py --file data/results.xlsx
```









