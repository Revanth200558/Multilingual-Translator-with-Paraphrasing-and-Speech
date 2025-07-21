from transformers import XLMRobertaTokenizer, XLMRobertaModel, T5ForConditionalGeneration, T5Tokenizer
from translatepy import Translator
from gtts import gTTS
from IPython.display import Audio, display
import torch
import os

# Load XLM-Roberta model and tokenizer
xlm_model_name = "xlm-roberta-base"
xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_model_name)
xlm_model = XLMRobertaModel.from_pretrained(xlm_model_name)

# Load T5 model and tokenizer for paraphrasing
paraphrasing_model_name = "t5-small"
paraphrasing_tokenizer = T5Tokenizer.from_pretrained(paraphrasing_model_name)
paraphrasing_model = T5ForConditionalGeneration.from_pretrained(paraphrasing_model_name)

# List of language codes (for user reference)
language_codes = {
   "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "br": "Breton", "bs": "Bosnian", "ca": "Catalan",
    "ce": "Chechen", "ch": "Chamorro", "co": "Corsican", "cr": "Cree",
    "cs": "Czech", "cu": "Church Slavic", "cv": "Chuvash", "cy": "Welsh",
    "da": "Danish", "de": "German", "dv": "Divehi", "dz": "Dzongkha",
    "ee": "Ewe", "el": "Greek", "en": "English", "eo": "Esperanto",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian",
    "ff": "Fula", "fi": "Finnish", "fj": "Fijian", "fo": "Faroese",
    "fr": "French", "fy": "Frisian", "ga": "Irish", "gd": "Scots Gaelic",
    "gl": "Galician", "gn": "Guaraní", "gu": "Gujarati", "gv": "Manx",
    "ha": "Hausa", "hi": "Hindi", "ho": "Hiri Motu", "hr": "Croatian",
    "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian", "hz": "Herero",
    "ia": "Interlingua", "id": "Indonesian", "ie": "Interlingue", "ig": "Igbo",
    "ii": "Sichuan Yi", "ik": "Inupiat", "io": "Ido", "is": "Icelandic",
    "it": "Italian", "iu": "Inuktitut", "ja": "Japanese", "jv": "Javanese",
    "ka": "Georgian", "kg": "Kongo", "ki": "Kikuyu", "kj": "Kwanyama",
    "kk": "Kazakh", "kl": "Kalaallisut", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "kr": "Kanuri", "ks": "Kashmiri", "ku": "Kurdish",
    "kv": "Komi", "kw": "Cornish", "ky": "Kyrgyz", "la": "Latin",
    "lb": "Luxembourgish", "lg": "Ganda", "li": "Limburgish", "ln": "Lingala",
    "lo": "Lao", "lt": "Lithuanian", "lu": "Luba-Katanga", "lv": "Latvian",
    "mg": "Malagasy", "mh": "Marshallese", "mi": "Māori", "mk": "Macedonian",
    "ml": "Malayalam", "mn": "Mongolian", "mo": "Moldovan", "mr": "Marathi",
    "ms": "Malay", "mt": "Maltese", "my": "Burmese", "na": "Nauru",
    "nb": "Norwegian Bokmål", "nd": "Northern Ndebele", "ne": "Nepali", "ng": "Ndonga",
    "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian", "nr": "Southern Ndebele",
    "nv": "Navajo", "ny": "Chichewa", "oc": "Occitan", "oj": "Ojibwe",
    "om": "Oromo", "or": "Oriya", "os": "Ossetian", "pa": "Punjabi",
    "pi": "Pāli", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
    "qu": "Quechua", "rm": "Romansh", "rn": "Kirundi", "ro": "Romanian",
    "ru": "Russian", "rw": "Kinyarwanda", "sa": "Sanskrit", "sc": "Sardinian",
    "sd": "Sindhi", "se": "Northern Sami", "sg": "Sango", "sh": "Serbo-Croatian",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sm": "Samoan",
    "sn": "Shona", "so": "Somali", "sq": "Albanian", "sr": "Serbian",
    "ss": "Swati", "st": "Southern Sotho", "su": "Sundanese", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "tg": "Tajik",
    "th": "Thai", "ti": "Tigrinya", "tk": "Turkmen", "tl": "Tagalog",
    "tn": "Tswana", "to": "Tonga", "tr": "Turkish", "ts": "Tsonga",
    "tt": "Tatar", "tw": "Twi", "ty": "Tahitian", "uz": "Uzbek",
    "ve": "Venda", "vi": "Vietnamese", "vo": "Volapük", "wa": "Walloon",
    "wo": "Wolof", "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba",
    "za": "Zhuang", "zu": "Zulu"
}

# Function to generate embeddings using XLM-Roberta
def get_xlm_embeddings(text):
    inputs = xlm_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = xlm_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Function to detect language (dummy implementation, replace with a real one if needed)
def detect_language(text):
    # Placeholder for real language detection
    return 'en'

# Function to translate text using translatepy
def translate_text(text, src_lang, tgt_lang):
    translator = Translator()
    try:
        translated = translator.translate(text, source_language=src_lang, destination_language=tgt_lang)
        return translated.result
    except Exception as e:
        print(f"Translation failed: {e}")
        return ""

# Function to generate paraphrases using T5
def paraphrase_text(text):
    try:
        input_ids = paraphrasing_tokenizer.encode(f"paraphrase: {text}", return_tensors="pt", max_length=512, truncation=True)
        output = paraphrasing_model.generate(input_ids, num_beams=5, num_return_sequences=3, early_stopping=True)
        paraphrases = [paraphrasing_tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output]
        return paraphrases
    except Exception as e:
        print(f"Paraphrasing failed: {e}")
        return [text]

# Function to convert text to speech using gTTS
def text_to_speech(text, lang_code, filename="output.mp3"):
    try:
        tts = gTTS(text, lang=lang_code)
        tts.save(filename)
        if os.path.isfile(filename):
            audio = Audio(filename, autoplay=True)
            display(audio)
        else:
            print(f"Audio file '{filename}' not found.")
    except Exception as e:
        print(f"Text-to-speech failed: {e}")

# Function to translate and generate paraphrases
def translate_and_paraphrase(text, src_lang, tgt_lang):
    # Translate text
    translated_text = translate_text(text, src_lang, tgt_lang)

    if not translated_text:
        print("Translation failed.")
        return

    # Generate paraphrases
    paraphrases = paraphrase_text(translated_text)

    # Display and speak
    print(f"\nTranslated text to {language_codes[tgt_lang]}: {translated_text}\n")
    text_to_speech(translated_text, tgt_lang)


# Main function to accept user input and translate
def translate_input():
    text = input("Enter the text to translate: ").strip()

    # Detect the source language (assuming English here for demonstration)
    src_lang = detect_language(text)
    print(f"Detected source language: {language_codes.get(src_lang, 'Unknown')}")

    # Prompt user for target language code
    print("\nAvailable target languages:")
    for code, lang in language_codes.items():
        print(f"{code}: {lang}")
    tgt_lang = input("Enter the target language code: ").strip()

    if tgt_lang not in language_codes:
        print("Invalid language code.")
        return

    # Translate text and generate paraphrases
    translate_and_paraphrase(text, src_lang, tgt_lang)

# Run the translation function
if __name__ == "__main__":
    translate_input()
