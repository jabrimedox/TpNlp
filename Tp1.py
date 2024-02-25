import pandas as pd



import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Charger le dataset
df = pd.read_csv("./sample.csv")

print(df.head())

# Fonction pour nettoyer le texte
def clean_text(text):
    # Mettre en minuscules
    text = text.lower()
    # Supprimer les URLs
    text = re.sub(r'http\S+', '', text)
    # Supprimer les balises HTML
    text = re.sub(r'<.*?>', '', text)
    # Supprimer les émojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Supprimer les émoticônes
    text = re.sub(r':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    # Supprimer la ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Fonction pour supprimer les mots vides (stopwords)
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

# Fonction pour appliquer le stemming
def apply_stemming(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Fonction pour appliquer la lemmatisation
def apply_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Nettoyage des données dans le DataFrame
df['clean_text'] = df['text'].apply(clean_text)
df['clean_text'] = df['clean_text'].apply(remove_stopwords)
df['clean_text'] = df['clean_text'].apply(apply_stemming)
df['clean_text'] = df['clean_text'].apply(apply_lemmatization)


# Afficher les premières lignes après nettoyage
print(df['clean_text'].head())



