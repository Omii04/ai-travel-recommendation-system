import pandas as pd

DATA_PATH = "data/raw/india_travel_dataset.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop useless index column
    if "unnamed:_0" in df.columns:
        df = df.drop(columns=["unnamed:_0"])

    # Convert text columns to lowercase
    df["name"] = df["name"].str.lower()
    df["type"] = df["type"].str.lower()
    df["state"] = df["state"].str.lower()
    df["city"] = df["city"].str.lower()
    df["significance"] = df["significance"].str.lower()
    df["best_time_to_visit"] = df["best_time_to_visit"].str.lower()

    # Convert numeric columns safely
    df["google_review_rating"] = pd.to_numeric(df["google_review_rating"], errors="coerce")
    df["number_of_google_review_in_lakhs"] = pd.to_numeric(
        df["number_of_google_review_in_lakhs"], errors="coerce"
    )

    # Drop duplicates
    df = df.drop_duplicates(subset=["name", "city"])

    # Fill missing values
    df = df.fillna(0)

    return df


if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    print(df.head())
    print("\nColumns:", df.columns)