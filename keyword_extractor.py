# LEVEL 3
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")  # SpaCy model for NER
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast & efficient BERT model

def preprocess_text(text):
    """ Tokenize, lemmatize, and remove stopwords """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def extract_keywords(query, corpus, top_n=5):
    """ Hybrid approach: TF-IDF + SBERT + Named Entity Recognition """
    corpus_cleaned = [preprocess_text(doc) for doc in corpus]
    query_cleaned = preprocess_text(query)

    # Step 1: Compute TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus_cleaned + [query_cleaned])
    query_vector = tfidf_matrix[-1]  # Query vector
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = {word: score for word, score in zip(feature_names, query_vector.toarray().flatten())}

    # Step 2: Compute SBERT semantic similarity scores
    corpus_embeddings = sbert_model.encode(corpus_cleaned, convert_to_tensor=True)
    query_embedding = sbert_model.encode([query_cleaned], convert_to_tensor=True)

    similarity_scores = cosine_similarity(query_embedding.cpu().numpy(), corpus_embeddings.cpu().numpy()).flatten()
    word_similarities = {feature_names[i]: similarity_scores[i] for i in range(min(len(feature_names), len(similarity_scores)))}

    # Step 3: Named Entity Recognition (NER)
    query_ner = [ent.text.lower() for ent in nlp(query).ents]
    
    # Step 4: Combine scores (TF-IDF + SBERT) and prioritize Named Entities
    combined_scores = {
        word: (tfidf_scores.get(word, 0) * 0.5 + word_similarities.get(word, 0) * 0.5)
        for word in set(tfidf_scores.keys()).union(set(word_similarities.keys()))
    }
    
    # Boost Named Entities
    for word in query_ner:
        if word in combined_scores:
            combined_scores[word] *= 1.5  # Prioritize NER terms

    # Sort and select top keywords
    sorted_keywords = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, _ in sorted_keywords[:top_n]]


def select_sentences(corpus, keywords):
    selected_sentences = []
    for doc in corpus:
        sentences = doc.split('. ')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                selected_sentences.append(sentence)
    return selected_sentences


if __name__ == "__main__":
    # Example Usage
    corpus = [
        "NLP deals with text processing, including keyword extraction.",
        "BERT improves text understanding for search engines and AI models.",
        "Deep learning is transforming natural language processing."
    ]
    query = "How does NLP extract keywords?"
    print(extract_keywords(query, corpus, 10))
