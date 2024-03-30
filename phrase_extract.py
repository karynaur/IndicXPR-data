import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from translate import batch_translate
import torch
import re
from IndicTransTokenizer import IndicProcessor
import gc
from translate import initialize_model_and_tokenizer
import os


def read_sentences(file_path, n = 3000000):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for _ in range(n):
            sentence = file.readline().strip().lower()
            sentences.append(sentence)
    return sentences

def preprocess_sentences(sentences):
    index = defaultdict(list)
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            index[word].append(i)
    return index

def find_sentence_indices_with_phrase(index, sentences, phrase, n=32):
    out = []
    count = 0

    phrase_words = phrase.split()

    sentence_indices = [set(index[word]) for word in phrase_words]
    common_indices = set.intersection(*sentence_indices)

    for i in common_indices:
        if count > n - 1:
            break
        if phrase in sentences[i]:
            count += 1
            out.append(sentences[i])
    return out

def clean_sentence(sentence, language, stop_words):
    if language == 'eng_Latn':
        regex = re.compile(r'[^\w\s]')
    elif language == 'hin_Deva':
        regex = re.compile(r'[^\u0900-\u097F]')
    elif  language == 'mal_Mlym':
        regex = re.compile(r'[^\u0D00-\u0D7F]')
    elif  language == 'tam_Taml':
        regex = re.compile(r'[^\u0B80-\u0BFF]')
    elif  language == 'tel_Telu':
        regex = re.compile(r'[^\u0C00-\u0C7F]')
    elif  language == 'kan_Knda':
        regex = re.compile(r'[^\u0C80-\u0CFF]')

    # Clean the sentence
    sentence = regex.sub(" ", sentence)
    sentence = ' '.join([w for w in sentence.split() if w not in stop_words and len(w) > 2])
    return sentence

def extract_ngrams(tokenized_sentences):
    all_ngrams = []
    for sentence in tqdm(tokenized_sentences):
        for n in range(2, 5):  # Generate n-grams of length 2 to 4
            ngrams_list = ngrams(sentence, n)
            all_ngrams.extend(ngrams_list)
    return all_ngrams

def translate_phrases(phrases, src_lang, tgt_lang, model, tokenizer, ip):
    translations = batch_translate(phrases, src_lang, tgt_lang, model, tokenizer, ip)
    return translations

def match_phrases(filtered_ngrams_src, translated_phrases, hindi_phrases):
    matches = [(filtered_ngrams_src[index], phrase) for index, phrase in enumerate(translated_phrases) if phrase in hindi_phrases]
    return matches

def save_matched_phrases(matched_phrases, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for index, phrase in matched_phrases:
            file.write(index + '\t' + phrase + '\n')

def get_top_phrases(src_file, lang):
    sentences = read_sentences(src_file)

    if lang == 'eng_Latn':
        stop_words = set(stopwords.words('english'))
    elif lang == 'hin_Deva':
        stop_words = set(stopwords.words('hindi'))
    elif lang == 'mal_Mlym':
        stop_words = set(stopwords.words('malayalam'))
    elif lang == 'tam_Taml':
        stop_words = set(stopwords.words('tamil'))
    elif lang == 'tel_Telu':
        stop_words = set(stopwords.words('telugu'))
    elif lang == 'kan_Knda':
        stop_words = set(stopwords.words('kannada'))

    clean_sentences = [clean_sentence(sentence, lang, stop_words).split(' ') for sentence in tqdm(sentences)]

    all_ngrams = extract_ngrams(clean_sentences)

    ngram_counts = Counter(all_ngrams).most_common()

    filtered_ngrams = {phrase: count for phrase, count in ngram_counts if count > 32}

    del clean_sentences, all_ngrams, ngram_counts
    gc.collect()

    return filtered_ngrams, sentences

def get_phrases(src_folder, src_lang, tgt_lang):
    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-dist-200M"
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", None)
    ip = IndicProcessor(inference=True)

    print("Loading ngrams from " + src_folder + 'train.en')
    filtered_ngrams_src, src_sentences = get_top_phrases(src_folder + 'train.en', src_lang)
    filtered_ngrams_src = [" ".join(i) for i in filtered_ngrams_src.keys()]
    print(f"Length of ngram list from {src_lang}: {len(filtered_ngrams_src)}")

    tgt_suffix = src_folder.split('/')[-2].split('-')[-1]
    print("Loading ngrams from " + src_folder + 'train.' + tgt_suffix)
    filtered_ngrams_tgt, tgt_sentences = get_top_phrases(src_folder + 'train.' + tgt_suffix, tgt_lang)
    filtered_ngrams_tgt = [" ".join(i) for i in filtered_ngrams_tgt.keys()]
    print(f"Length of ngram list from {tgt_lang}: {len(filtered_ngrams_tgt)}")

    translated_phrases = translate_phrases(filtered_ngrams_src, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)

    matched_phrases = match_phrases(filtered_ngrams_src, translated_phrases, filtered_ngrams_tgt)
    print(f"Matched phrases: {len(matched_phrases)}")
    # save_matched_phrases(matched_phrases, output_file)

    return matched_phrases, filtered_ngrams_src, src_sentences, tgt_sentences

def main(src_folder, src_lang, tgt_lang, out_folder):
    matched_phrases, filtered_ngrams_src, src_sentences, tgt_sentences = get_phrases(src_folder, src_lang, tgt_lang)
    print("Getting src index")
    src_index = preprocess_sentences(src_sentences)
    print("Getting tgt index")
    tgt_index = preprocess_sentences(tgt_sentences)

    print("Writing sentences to file")
    written_indices = set()
    for ii , (phrase_src, phrase_tgt) in tqdm(enumerate(matched_phrases), total = len(matched_phrases)):
        if phrase_tgt in written_indices:
            continue
        written_indices.add(phrase_tgt)

        src = find_sentence_indices_with_phrase(src_index, src_sentences, phrase_src)
        tgt = find_sentence_indices_with_phrase(tgt_index, tgt_sentences, phrase_tgt)

        if len(src) != 32 or len(tgt) != 32:
            continue

        with open(os.path.join(out_folder, 'sentences', 'en-' + tgt_lang.split('_')[0] + '-phrase-sentence.32.tsv'), 'a', encoding='utf-8') as file:
            file.write(f"{phrase_src}\t{ii}\t")
            for sentence in src:
                if sentence == src[-1]:
                    file.write(sentence)
                    continue
                file.write(sentence + '\t')
            file.write('\n')

        with open(os.path.join(out_folder, 'sentences', tgt_lang.split('_')[0] + '-phrase-sentence.32.tsv'), 'a', encoding='utf-8') as file:
            file.write(f"{phrase_tgt}\t{ii}\t")
            for sentence in tgt:
                if sentence == src[-1]:
                    file.write(sentence)
                    continue
                file.write(sentence + '\t')
            file.write('\n')

        if ii % 4 == 0:
            with open(os.path.join(out_folder, 'test', 'test-en-' + tgt_lang.split('_')[0]) + '-32-phrase.txt', 'a', encoding='utf-8') as file:
                file.write(phrase_src + '\t' + phrase_tgt + '\n')
        else:
            with open(os.path.join(out_folder, 'train', 'train-en-' + tgt_lang.split('_')[0]  + '-32-phrase.txt'), 'a', encoding='utf-8') as file:
                file.write(phrase_src + '\t' + phrase_tgt + '\n')
        
    print(subprocess.run(["wc", "-l", os.path.join(out_folder, 'test', 'test-en-' + tgt_lang.split('_')[0] + '-32-phrase.txt')], capture_output=True).stdout.decode())
    print(subprocess.run(["wc", "-l", os.path.join(out_folder, 'train', 'train-en-' + tgt_lang.split('_')[0] + '-32-phrase.txt')], capture_output=True).stdout.decode())
    print(subprocess.run(["wc", "-l", os.path.join(out_folder, 'sentences', tgt_lang.split('_')[0] + '-phrase-sentence.32.tsv')], capture_output=True).stdout.decode())
    print(subprocess.run(["wc", "-l", os.path.join(out_folder, 'sentences', 'en-' + tgt_lang.split('_')[0] + '-phrase-sentence.32.tsv')], capture_output=True).stdout.decode())


