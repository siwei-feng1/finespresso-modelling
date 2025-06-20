# Common language mappings for use across the application
LANGUAGE_MAP = {
    # Multilanguage
    'multilanguage': 'xx',
    
    # Germanic languages
    'english': 'en',
    'german': 'de',
    'dutch': 'nl',
    'swedish': 'sv',
    'danish': 'da',
    'norwegian': 'no',
    'icelandic': 'is',
    'faroese': 'fo',
    'luxembourgish': 'lb',
    'frisian': 'fy',
    
    # Romance languages
    'french': 'fr',
    'italian': 'it',
    'spanish': 'es',
    'portuguese': 'pt',
    'romanian': 'ro',
    'catalan': 'ca',
    'galician': 'gl',
    'occitan': 'oc',
    'moldovan': 'mo',
    
    # Slavic languages
    'russian': 'ru',
    'polish': 'pl',
    'ukrainian': 'uk',
    'belarusian': 'be',
    'czech': 'cs',
    'slovak': 'sk',
    'slovenian': 'sl',
    'croatian': 'hr',
    'serbian': 'sr',
    'macedonian': 'mk',
    'bulgarian': 'bg',
    
    # Baltic languages
    'lithuanian': 'lt',
    'latvian': 'lv',
    
    # Uralic languages
    'finnish': 'fi',
    'estonian': 'et',
    'hungarian': 'hu',
    
    # Celtic languages
    'irish': 'ga',
    'scottish_gaelic': 'gd',
    'welsh': 'cy',
    'breton': 'br',
    
    # Greek
    'greek': 'el',
    
    # Albanian
    'albanian': 'sq',
    
    # Other European languages
    'maltese': 'mt',
    'basque': 'eu',
    
    # Non-European languages (from original list)
    'chinese': 'zh',
    'japanese': 'ja',
    'korean': 'ko',
    'arabic': 'ar',
}

spacy_model_mapping = {
    "en": "en_core_web_sm",     # English
    "de": "de_core_news_sm",    # German
    "fr": "fr_core_news_sm",    # French
    "es": "es_core_news_sm",    # Spanish
    "pt": "pt_core_news_sm",    # Portuguese
    "it": "it_core_news_sm",    # Italian
    "nl": "nl_core_news_sm",    # Dutch
    "el": "el_core_news_sm",    # Greek
    "no": "nb_core_news_sm",    # Norwegian
    "lt": "lt_core_news_sm",    # Lithuanian
    "hr": "hr_core_news_sm",    # Croatian
    "da": "da_core_news_sm",    # Danish
    "ja": "ja_core_news_sm",    # Japanese
    "pl": "pl_core_news_sm",    # Polish
    "ro": "ro_core_news_sm",    # Romanian
    "ru": "ru_core_news_sm",    # Russian
    "zh": "zh_core_web_sm",     # Chinese
    "xx": "xx_ent_wiki_sm"      # Multi-language
}

def normalize_language_code(language: str) -> str:
    """
    Normalize a language name or code to a standard 2-letter ISO code.
    
    Args:
        language (str): Language name or code to normalize
        
    Returns:
        str: 2-letter ISO language code, defaults to 'en' if invalid
    """
    language = language.lower().strip()
    
    # If we got a language name instead of code, try to map it
    if len(language) > 2:
        language = LANGUAGE_MAP.get(language, 'en')  # Default to 'en' if mapping not found
    
    # Ensure the language code is exactly 2 characters
    if len(language) != 2:
        language = 'en'  # Default to English if we get an invalid code
        
    return language
