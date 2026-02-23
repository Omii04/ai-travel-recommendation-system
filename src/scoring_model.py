import pandas as pd

def normalize_column(series):
    return (series - series.min()) / (series.max() - series.min())


def calculate_score(df,
                    rating_weight=0.4,
                    review_weight=0.3,
                    significance_weight=0.2,
                    budget_weight=0.1):

    # Normalize numeric columns safely
    def normalize(series):
        if series.max() == series.min():
            return 0
        return (series - series.min()) / (series.max() - series.min())

    df["rating_score"] = normalize(df["google_review_rating"])
    df["review_score"] = normalize(df["number_of_google_review_in_lakhs"])

    # Budget score (lower entrance fee = better)
    df["budget_score"] = 1 - normalize(df["entrance_fee_in_inr"])

    # Significance boost
    df["significance_score"] = df["significance"].apply(
        lambda x: 1 if "historical" in str(x) else 0.7
    )

    df["final_score"] = (
        rating_weight * df["rating_score"] +
        review_weight * df["review_score"] +
        significance_weight * df["significance_score"] +
        budget_weight * df["budget_score"]
    )

    return df