# Dataet creation for IndicXPR

## Install IndicTrans and it's dependencies 

```bash
# Clone the github repository and navigate to the project directory.
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2

# Install all the dependencies and requirements associated with the project.
source install.sh
```

Run the following to download and install dependencies

```bash
python3 -m pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
python3 -c "import nltk; nltk.download('punkt')"
python3 -m pip install bitsandbytes scipy accelerate datasets sacrebleu
python3 -m pip install sentencepiece datasets

git clone https://github.com/VarunGumma/IndicTransTokenizer
cd IndicTransTokenizer
python3 -m pip install --editable ./
cd ..
```

## Download and extract the Samanantar parallel dataset

Go to the [Samanantar website](https://ai4bharat.iitm.ac.in//samanantar/) and download the v0.3 version of the dataset and store it locally

## Add stopwords for the target language

Download the nltk stopwords

```py
import nltk
nltk.download('stopwords')
```

Locate the directory in which the stopwords are saved from the output of the previous execution. Move the stopwords files from `stopwords/` to the stopwords folder, which is usually `/root/nltk-data/corpora/stopwords/`.

## Extract Phrases
Modify main.py and change the src_lang, tgt_lang, and src_file. Run `python3 main.py` to extract phrases and save it in [XPR](https://github.com/cwszz/XPR/) format.
