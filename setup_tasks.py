import nltk

def download_nltk_resources():
    """Скачивает необходимые данные NLTK при первом запуске"""
    resources = {
        'corpora': ['stopwords', 'wordnet'],
        'taggers': ['punkt'],
        'tokenizers': ['punkt']
    }
    
    for resource_type, names in resources.items():
        for name in names:
            try:
                nltk.data.find(f'{resource_type}/{name}')
            except LookupError:
                nltk.download(name, quiet=True)

if __name__ == '__main__':
    download_nltk_resources()