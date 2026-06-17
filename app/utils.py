import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Text preprocessing for model input.
    - Lowercase
    - Remove special characters
    - Lemmatize tokens
    No stopword removal — preserves negations like 'not', 'no', 'never'
    which are critical for sales sentiment context.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove special characters, keep letters and spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and lemmatize (no stopword removal)
    tokens = text.split()
    processed = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if len(token) > 1   # remove single characters only
    ]

    return ' '.join(processed)
