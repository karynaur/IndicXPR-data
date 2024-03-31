from phrase_extract import main

src_lang = "eng_Latn"
languages = ["tel_Telu", "kan_Knda", "tam_Taml", "hin_Deva", "mal_Mlym"]
tgt_lang = "mal_Mlym"

# Specify source and target files
src_file = 'data/en-ml/'  # Change this to your source file
output_folder = 'out/'  # Change this to your desired output file

main(src_file, src_lang, tgt_lang, output_folder)
