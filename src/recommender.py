def recommend_places(df, place_type=None, best_time=None, top_n=5):

    filtered = df.copy()

    if place_type:
        filtered = filtered[filtered["type"] == place_type.lower()]

    if best_time:
        filtered = filtered[
            filtered["best_time_to_visit"].str.contains(best_time.lower())
        ]

    recommendations = filtered.sort_values(
        by="final_score", ascending=False
    )

    return recommendations.head(top_n)