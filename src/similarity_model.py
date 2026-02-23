from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity(df):

    df["combined_features"] = (
        df["type"].astype(str) + " " +
        df["significance"].astype(str) + " " +
        df["city"].astype(str)
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


def get_similar_places(df, similarity_matrix, place_name, top_n=5):

    place_name = place_name.lower()

    indices = df[df["name"] == place_name].index

    if len(indices) == 0:
        return "Place not found"

    idx = indices[0]

    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similar_indices = [i[0] for i in similarity_scores[1:top_n+1]]

    return df.iloc[similar_indices][["name", "city", "type"]]