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
!python3 -m pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
!python3 -c "import nltk; nltk.download('punkt')"
!python3 -m pip install bitsandbytes scipy accelerate datasets sacrebleu
!python3 -m pip install sentencepiece datasets

!git clone https://github.com/VarunGumma/IndicTransTokenizer
%cd IndicTransTokenizer
!python3 -m pip install --editable ./
%cd ..
```

## Download and extract the Samanantar parallel dataset

## Extract Phrases

Modify main.py
