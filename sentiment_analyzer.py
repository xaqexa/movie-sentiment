import pandas as pd
import numpy as np
import re
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib

# Sprawdzenie czy wymagane biblioteki są zainstalowane
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    print("Zainstaluj nltk: pip install nltk")
    exit()

try:
    import spacy
except ImportError:
    print("Zainstaluj spacy: pip install spacy")
    print("Pobierz model: python -m spacy download en_core_web_sm")
    exit()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2

class SentimentAnalyzer:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.selector = None
        self.best_model = None
        self.stop_words = None
        self.nlp = None
        
    def setup_nltk_and_spacy(self):
        """Pobieranie zasobów NLTK i ładowanie modelu spaCy"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Nie można załadować modelu spaCy. Zainstaluj: python -m spacy download en_core_web_sm")
            exit()
            
        self.stop_words = set(stopwords.words('english'))
        custom_stop_words = {
            "movie", "film", "one", "see", "make", "get", "go", "would",
            "even", "really", "think", "watch", "time", "character", "story",
            "show", "scene", "look", "say", "much", "know", "could", "also",
            "give", "take", "first", "play", "way"
        }
        self.stop_words.update(custom_stop_words)

    def clean_text(self, text):
        """Czyszczenie tekstu"""
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        return text

    def preprocess_texts(self, texts):
        """Przetwarzanie tekstów"""
        processed_texts = []
        for doc in self.nlp.pipe(texts, batch_size=500, disable=["ner", "parser"]):
            lemmatized_tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
            tokens_without_stopwords = [token for token in lemmatized_tokens if token not in self.stop_words]
            processed_texts.append(" ".join(tokens_without_stopwords))
        return processed_texts

    def load_and_preprocess_data(self, file_path):
        """Ładowanie i przetwarzanie danych"""
        # Sprawdź czy istnieje przetworzony plik
        preprocessed_path = "preprocessed.csv"
        
        if os.path.exists(preprocessed_path):
            print("Znaleziono przetworzony plik. Ładowanie...")
            self.df = pd.read_csv(preprocessed_path)
            return
            
        print("Przetwarzanie danych...")
        self.setup_nltk_and_spacy()
        
        # Wczytaj surowe dane
        df = pd.read_csv(file_path)
        print(f"Wczytano {len(df)} recenzji")
        print("Pierwszych 5 wierszy:")
        print(df.head())
        print("\nRozkład sentymentu:")
        print(df['sentiment'].value_counts())
        
        # Usunięcie duplikatów i pustych recenzji
        df.drop_duplicates(subset='review', inplace=True)
        df.dropna(subset=['review'], inplace=True)
        
        # Czyszczenie tekstu
        df['clean_review'] = df['review'].apply(self.clean_text)
        
        # Przetwarzanie tekstów
        df['processed_review'] = self.preprocess_texts(df['clean_review'])
        
        # Zapisz przetworzony plik
        df.to_csv(preprocessed_path, index=False)
        self.df = df
        print(f"Zapisano przetworzony plik: {preprocessed_path}")

    def analyze_data(self):
        """Analiza danych"""
        if self.df is None:
            print("Najpierw wczytaj dane!")
            return
            
        # Analiza częstości słów
        word_freq = Counter(" ".join(self.df["processed_review"]).split())
        common_words = word_freq.most_common(10)
        print("\nNajczęstsze słowa:")
        for word, freq in common_words:
            print(f"{word}: {freq}")
        
        # Długość recenzji
        self.df['review_length'] = self.df['processed_review'].apply(lambda x: len(x.split()))
        
        # Wykresy
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.histplot(self.df['review_length'], bins=50)
        plt.title("Rozkład długości recenzji")
        
        plt.subplot(1, 3, 2)
        sns.countplot(x=self.df["sentiment"], palette="coolwarm")
        plt.title("Rozkład klas (positive/negative)")
        
        plt.tight_layout()
        plt.show()
        
        # WordClouds
        self.create_wordclouds()

    def create_wordclouds(self):
        """Tworzenie chmur słów"""
        plt.figure(figsize=(15, 6))
        
        # Pozytywne recenzje
        plt.subplot(1, 2, 1)
        positive_text = " ".join(self.df[self.df['sentiment'] == 'positive']['processed_review'])
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(positive_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Pozytywne recenzje")
        
        # Negatywne recenzje
        plt.subplot(1, 2, 2)
        negative_text = " ".join(self.df[self.df['sentiment'] == 'negative']['processed_review'])
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(negative_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Negatywne recenzje")
        
        plt.tight_layout()
        plt.show()

    def prepare_training_data(self):
        """Przygotowanie danych treningowych"""
        if self.df is None:
            print("Najpierw wczytaj dane!")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.df['processed_review'], self.df['sentiment'], 
            test_size=0.2, stratify=self.df['sentiment'], random_state=42
        )
        
        return X_train, X_test, y_train, y_test

    def train_models(self):
        """Trenowanie modeli"""
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        # Wektoryzacja
        self.vectorizer = TfidfVectorizer(max_features=10000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Modele
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": MultinomialNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Trenowanie i ewaluacja
        for name, model in models.items():
            print(f"Trenowanie {name}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            print(f"=== {name} ===")
            print(classification_report(y_test, y_pred))
            print("="*60)
        
        # Optymalizacja Logistic Regression
        print("Optymalizacja Logistic Regression...")
        params = {
            "C": [0.01, 0.1, 1, 10],
            "solver": ['liblinear', 'lbfgs']
        }
        
        grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train_vec, y_train)
        
        print("Najlepsze parametry:", grid.best_params_)
        print("Najlepszy wynik f1:", grid.best_score_)
        
        # Selekcja cech
        self.selector = SelectKBest(chi2, k=2000)
        X_train_selected = self.selector.fit_transform(X_train_vec, y_train)
        X_test_selected = self.selector.transform(X_test_vec)
        
        # Ostateczny model
        self.best_model = LogisticRegression(max_iter=1000, **grid.best_params_)
        self.best_model.fit(X_train_selected, y_train)
        
        y_pred = self.best_model.predict(X_test_selected)
        print("\n=== KOŃCOWY MODEL ===")
        print(classification_report(y_test, y_pred))
        
        # Zapisz model
        self.save_model()

    def save_model(self):
        """Zapisywanie modelu"""
        joblib.dump({
            'model': self.best_model,
            'vectorizer': self.vectorizer,
            'selector': self.selector,
            'stop_words': self.stop_words
        }, 'sentiment_model.pkl')
        print("Model zapisany jako sentiment_model.pkl")

    def load_model(self):
        """Ładowanie modelu"""
        if os.path.exists('sentiment_model.pkl'):
            data = joblib.load('sentiment_model.pkl')
            self.best_model = data['model']
            self.vectorizer = data['vectorizer']
            self.selector = data['selector']
            self.stop_words = data['stop_words']
            self.setup_nltk_and_spacy()
            return True
        return False

    def predict_sentiment(self, text):
        """Predykcja sentymentu dla pojedynczego tekstu"""
        if self.best_model is None:
            print("Model nie został wytrenowany lub wczytany!")
            return None
            
        # Przetwarzanie tekstu
        clean_text = self.clean_text(text)
        processed_text = self.preprocess_texts([clean_text])[0]
        
        # Wektoryzacja
        text_vec = self.vectorizer.transform([processed_text])
        text_selected = self.selector.transform(text_vec)
        
        # Predykcja
        prediction = self.best_model.predict(text_selected)[0]
        probability = self.best_model.predict_proba(text_selected)[0]
        
        return {
            'sentiment': prediction,
            'confidence': max(probability),
            'probabilities': {
                'negative': probability[0],
                'positive': probability[1]
            }
        }

def main():
    analyzer = SentimentAnalyzer()
    
    # Sprawdź czy model już istnieje
    if analyzer.load_model():
        print("Wczytano istniejący model.")
        
        # Test predykcji
        test_review = "This movie was absolutely fantastic! Great acting and amazing plot."
        result = analyzer.predict_sentiment(test_review)
        print(f"\nTest recenzji: '{test_review}'")
        print(f"Przewidywany sentyment: {result['sentiment']}")
        print(f"Pewność: {result['confidence']:.2f}")
    else:
        # Wczytaj i przetwórz dane
        data_file = "IMDB.csv"  # Zmień na ścieżkę do swojego pliku
        
        if not os.path.exists(data_file):
            print(f"Nie znaleziono pliku {data_file}")
            print("Umieść plik IMDB.csv w tym samym folderze co skrypt.")
            return
            
        analyzer.load_and_preprocess_data(data_file)
        analyzer.analyze_data()
        analyzer.train_models()

if __name__ == "__main__":
    main()