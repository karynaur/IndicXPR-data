import csv
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, PhrasalConstraint
from IndicTransTokenizer import IndicTransTokenizer
from datasets import Dataset
from tqdm import tqdm
from time import time
import gc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip, bs = 256, max_length = 8):
    translations = []
    for i in tqdm(range(0, len(input_sentences), bs), leave = False):
        batch = input_sentences[i : i + bs]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)


        # Generate translations using the model
        with torch.inference_mode():
            generated_tokens = model.generate(
                **inputs,
                # constraints=constraints,
                use_cache=True,
                min_length=0,
                max_length=max_length,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs, generated_tokens, batch
        torch.cuda.empty_cache()
        gc.collect()

    return translations

def read_tsv_file(file_path, batch_size):
    with open(file_path, 'r') as file:
        for _ in tqdm(range(38335810), leave = True):
            file.readline()
        while True:
            queries = []
            positives = []
            negatives = []
            for _ in range(batch_size):
                line = file.readline()
                if not line:
                    return  # End of file
                values = line.strip().split('\t')
                queries.append(values[0])
                positives.append(values[1])
                negatives.append(values[2])
            yield queries, positives, negatives

def save_to_tsv(output_file, queries, positive, negative):
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for q, pos, neg in zip(queries, positive, negative):
            writer.writerow([q, pos, neg])