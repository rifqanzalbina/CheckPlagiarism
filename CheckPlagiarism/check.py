import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.metrics import edit_distance

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def preprocess_text(text):
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

def calculate_similarity(text1, text2):
    distance = edit_distance(text1, text2)
    max_length = max(len(text1), len(text2))
    similarity = 1 - (distance / max_length)

    # Set similarity to 100% if the texts are very similar
    if similarity >= 0.9:
        similarity = 1.0

    return similarity

text1 = "The quick brown fox jumps over the lazy dog."
text2 = "The quick brown fox jumps over the lazy dog and lazy cat."

preprocessed_text1 = preprocess_text(text1)
preprocessed_text2 = preprocess_text(text2)

similarity = calculate_similarity(preprocessed_text1, preprocessed_text2)

print(f"Similarity: {similarity * 100}%")
