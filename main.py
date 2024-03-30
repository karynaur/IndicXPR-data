from phrase_extract import main

src_lang = "eng_Latn"
languages = ["tel_Telu", "kan_Knda", "tam_Taml", "hin_Deva", "mal_Mlym"]


# Specify source and target files
src_file = 'data/'  # Change this to your source file
output_file = 'out/'  # Change this to your desired output file

for lang in languages:
    main(src_file, src_lang, lang, output_file)
