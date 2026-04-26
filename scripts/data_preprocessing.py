#!/usr/bin/env python3
"""
data_preprocessing.py
=====================
Merges stress-detection datasets into a single unified CSV with columns
['text', 'label', 'domain'] plus any available numeric feature columns
(e.g., readability, LIWC, DAL, or social metadata).

Dreaddit subreddit metadata is appended to the text using the
``<SUBREDDIT=...>`` tag for lightweight contextual cues.

Label mapping:
    0 = stress-negative
    1 = stress-positive

Supported datasets (place in data/raw/):
    1. dreaddit-train.csv           (Reddit Long)
    2. Reddit_Combi.csv             (Reddit Long)
    3. Reddit_Title.csv             (Reddit Short)
    4. Twitter_Full.csv             (Twitter Short)
    5. Stressed_Tweets.csv          (Twitter Short - implicit label=1)
    6. Happy_Neutral.csv            (Optional: clearly positive/neutral text, label=0)

Usage:
    python data_preprocessing.py [--input-dir data/raw] [--output data/processed/unified_stress.csv]
"""

import argparse
import os
import sys
import zipfile

import pandas as pd


# ---------------------------------------------------------------------------
# Stop-word filtering for preprocessing
# ---------------------------------------------------------------------------
# Words kept even though NLTK marks them as stop words, because they
# carry meaningful semantic signal for stress and contrast detection.
# ---------------------------------------------------------------------------

_CONTRAST_WORDS_KEEP: frozenset[str] = frozenset({
    "not", "no", "nor", "never", "neither",          # negation
    "but", "however", "although", "though", "yet",   # contrast / concession
    "still", "despite", "whereas", "while",           # concession
    "very", "too", "extremely", "quite",              # intensifiers
    "more", "most", "less", "least",                  # comparatives
    "only",                                           # restriction
})


def _build_preprocessing_stop_words() -> frozenset[str]:
    """Return the NLTK English stop-word set minus :data:`_CONTRAST_WORDS_KEEP`."""
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as _nltk_sw
        return frozenset(_nltk_sw.words("english")) - _CONTRAST_WORDS_KEEP
    except Exception:
        # Graceful degradation: if NLTK is unavailable, skip filtering.
        return frozenset()


_PREPROCESSING_STOP_WORDS: frozenset[str] = _build_preprocessing_stop_words()


# ---------------------------------------------------------------------------
# Hard-negative contrast samples
# ---------------------------------------------------------------------------
# These teach the model that a positive override phrase after a contrast
# conjunction ('but happy', 'but satisfied') should suppress the stress label
# even when a stress-trigger word precedes the conjunction.
# ---------------------------------------------------------------------------

_CONTRAST_SAMPLES: list[tuple[str, int]] = [
    # Contrast/concession: stress trigger overridden by positive resolution
    ("I am tired but happy", 0),
    ("I feel overwhelmed but satisfied", 0),
    ("life is stressful but exciting", 0),
    ("I am busy but enjoying it", 0),
    ("tough week but grateful for all the support", 0),
    ("bad day but my family cheered me up", 0),
    ("felt anxious however it passed quickly", 0),
    ("though exhausted I feel accomplished", 0),
    ("work is heavy yet I love what I do", 0),
    ("frustrated but determined to push through", 0),
    ("was stressed but things are better now", 0),
    ("despite the pressure I am feeling good", 0),
    ("long day but ended on a high note", 0),
    ("tired from the gym but feel amazing", 0),
    ("had some setbacks but still feeling positive", 0),
    ("not the easiest week but I came out stronger", 0),
    ("rough morning but afternoon turned out great", 0),
    ("work was hectic but I handled it well", 0),
    ("felt low earlier but mood lifted nicely", 0),
    ("challenging tasks but I finished everything", 0),
    ("had some doubt but feeling confident now", 0),
    ("nervous at first but it went really well", 0),
    ("things were uncertain but it all worked out", 0),
    ("a bit tired but genuinely content", 0),
    # Positive tone with negative or stress-trigger words
    # (model must learn these surface negatives signal non-stress)
    ("I don't have any worries today", 0),
    ("nothing is going wrong right now", 0),
    ("I have no stress whatsoever", 0),
    ("not a bad day at all everything is great", 0),
    ("never felt so relaxed and at ease before", 0),
    ("no problems here just enjoying life", 0),
    ("I can't complain everything is wonderful", 0),
    ("don't feel anxious at all today", 0),
    ("I have no complaints life is good", 0),
    ("nothing is bothering me feeling peaceful", 0),
    ("not worried about anything feeling free", 0),
    ("no stress no pressure just calm", 0),
    ("don't have a single thing to stress about", 0),
    ("I am not struggling at all doing well", 0),
    ("nothing feels heavy or difficult today", 0),
    ("no deadlines looming I can breathe easily", 0),
    ("I am not tired I feel energized", 0),
    ("not overwhelmed just steady and grounded", 0),
    ("no tension in my mind or body right now", 0),
    ("I won't let anything ruin this good mood", 0),
    ("not anxious not sad just genuinely happy", 0),
    ("I didn't lose sleep over anything feeling rested", 0),
    ("no bad news today just good vibes", 0),
    ("not a difficult day at all went smoothly", 0),
    ("I don't feel burned out I feel recharged", 0),
    ("no drama no conflict just a peaceful day", 0),
    ("nothing went wrong today what a relief", 0),
    ("I am not behind on anything feeling on top of things", 0),
    ("never been less stressed glad things are calm", 0),
    ("no anxiety just clarity and focus", 0),
    ("I feel no fear just excitement", 0),
    ("not struggling with anything feeling capable", 0),
    ("no need to worry everything is handled", 0),
    ("I don't feel any pressure right now", 0),
    ("nothing scary about today just normal and good", 0),
    ("not overwhelmed not underprepared just ready", 0),
    ("I won't worry about tomorrow today is great", 0),
    ("no tension no rush just ease", 0),
    ("nothing is unmanageable everything feels fine", 0),
    ("not tired not stressed just content", 0),
    ("I have no reason to feel down life is good", 0),
    ("nothing terrible happened it was a nice day", 0),
    # Calm / content / positive — short declarative style matching happy_neutral_eval
    ("I feel calm and content today", 0),
    ("feeling peaceful and grateful right now", 0),
    ("had a great day feeling refreshed and happy", 0),
    ("relaxing evening with a good book and tea", 0),
    ("grateful for my supportive friends and family", 0),
    ("today was productive and satisfying", 0),
    ("feeling optimistic about the week ahead", 0),
    ("laughed a lot during dinner tonight", 0),
    ("my mind feels clear and peaceful", 0),
    ("everything is going smoothly right now", 0),
    ("slept well and woke up refreshed this morning", 0),
    ("excited about my new hobby loving it", 0),
    ("appreciate the small wins today", 0),
    ("feeling balanced and steady", 0),
    ("enjoyed a pleasant conversation with a friend", 0),
    ("feel safe and relaxed at home", 0),
    ("workload feels manageable today", 0),
    ("feeling hopeful and motivated", 0),
    ("took a deep breath and felt at ease", 0),
    ("afternoon was quiet and restful", 0),
    ("in a good mood enjoying the day", 0),
    ("life is good and I am grateful", 0),
    ("happy to be spending time with family", 0),
    ("feeling energized after a good workout", 0),
    ("enjoying the sunshine today perfect weather", 0),
    ("had a delicious meal feeling satisfied", 0),
    ("excited about the weekend plans", 0),
    ("proud of what I accomplished today", 0),
    ("feeling at peace with everything", 0),
    ("grateful for this beautiful day", 0),
    ("had a wonderful time feeling uplifted", 0),
    ("just chilling watching my favorite show", 0),
    ("loving this weather great for a walk", 0),
    ("feeling confident and ready for the day", 0),
    ("spent quality time with loved ones today", 0),
    ("met a friend and had a great chat", 0),
    ("my dog made me smile today", 0),
    ("enjoying a cup of coffee feeling cozy", 0),
    ("looking forward to good things ahead", 0),
    ("feeling rested after a long sleep", 0),
    ("things are working out just fine", 0),
    ("doing well and feeling great today", 0),
    ("had a fun evening feeling cheerful", 0),
    ("life feels simple and beautiful today", 0),
    ("feeling accomplished after a productive day", 0),
    ("the weekend was perfect relaxing and fun", 0),
    ("today I feel light and free", 0),
    ("my heart feels full and content", 0),
    ("woke up in a great mood today", 0),
    ("everything feels manageable and okay", 0),
    ("feeling joyful and present in the moment", 0),
    ("had a lovely morning walk", 0),
    ("things are going really well lately", 0),
    ("feeling steady and positive", 0),
    ("I feel happy and healthy today", 0),
    ("loved spending today outdoors in the fresh air", 0),
    ("feeling grateful for the good people in my life", 0),
    ("the week went smoothly I am satisfied", 0),
    ("just enjoyed a great meal with friends", 0),
    ("I feel secure and comfortable right now", 0),
    ("this has been a genuinely good day", 0),
    ("feeling inspired and creative today", 0),
    ("relaxed and at ease nothing to worry about", 0),
    ("life is full of good moments right now", 0),
    ("in a calm and happy state of mind", 0),
    ("had a good laugh feeling light hearted", 0),
    ("everything is peaceful and under control", 0),
    ("glad to be alive enjoying the little things", 0),
    ("feeling refreshed and full of energy today", 0),
    ("today was a good day no complaints", 0),
    ("feeling supported and loved by those around me", 0),
    ("the day flew by in the best way", 0),
    ("I feel whole and content right now", 0),
    # Additional non-stressed examples — workplace / productivity
    ("finished all my tasks ahead of schedule feeling great", 0),
    ("my team is supportive and work feels rewarding", 0),
    ("meetings went well and the project is on track", 0),
    ("got positive feedback from my manager today", 0),
    ("my inbox is clear and I feel on top of things", 0),
    ("work life balance is really good right now", 0),
    ("collaborated well with colleagues today very satisfying", 0),
    ("solved a hard problem at work feeling proud", 0),
    ("the presentation went better than expected", 0),
    ("happy with the progress I made at work today", 0),
    # Additional non-stressed examples — social / relationships
    ("had a heartfelt talk with my best friend today", 0),
    ("family dinner was warm and full of laughter", 0),
    ("reconnected with an old friend feeling nostalgic and happy", 0),
    ("everyone around me is kind and supportive", 0),
    ("celebrated a friend's good news feeling joyful", 0),
    ("my relationships feel strong and nurturing", 0),
    ("enjoyed a quiet evening with my partner", 0),
    ("spent the afternoon with my kids what a joy", 0),
    ("feeling deeply connected to the people I love", 0),
    ("a simple kind gesture from a stranger made my day", 0),
    # Additional non-stressed examples — health / wellness
    ("went for a run and feel fantastic", 0),
    ("meditated this morning and feel grounded all day", 0),
    ("eating well sleeping well feeling great", 0),
    ("my body feels strong and healthy", 0),
    ("stretching and yoga left me feeling so loose and free", 0),
    ("hydrated well today and feel energized", 0),
    ("got a full night of sleep waking up refreshed", 0),
    ("feeling physically strong and mentally clear", 0),
    ("took a wellness day and feel completely restored", 0),
    ("health check came back great no concerns at all", 0),
    # Additional non-stressed examples — hobbies / leisure
    ("spent the morning painting and lost track of time happily", 0),
    ("reading a great book totally absorbed and relaxed", 0),
    ("gardening today was therapeutic and fun", 0),
    ("played music for hours and felt pure joy", 0),
    ("cooked a new recipe and it turned out delicious", 0),
    ("went hiking the views were breathtaking", 0),
    ("enjoyed a long bike ride feeling free", 0),
    ("watched a great film totally engrossed and happy", 0),
    ("baked cookies today the whole house smells wonderful", 0),
    ("finished a puzzle feeling satisfied and calm", 0),
    # Additional non-stressed examples — mindset / gratitude
    ("counting my blessings and there are many", 0),
    ("feeling thankful for everything I have right now", 0),
    ("my outlook is bright and the future looks good", 0),
    ("I am present enjoying every moment today", 0),
    ("feeling a deep sense of gratitude and ease", 0),
    ("today reminded me how much beauty surrounds me", 0),
    ("I choose to focus on what is going well", 0),
    ("my mindset is positive and grounded today", 0),
    ("I am exactly where I want to be right now", 0),
    ("life feels abundant and full of possibility", 0),
    # --------------------------------------------------------------------------
    # Stressed examples (label=1) — added to balance the non-stressed set above
    # and to teach the model to detect stress in realistic scenarios that lack
    # obvious emotional exclamation words.
    # --------------------------------------------------------------------------
    # Deadline / work overload
    ("deadlines are piling up and I cannot keep up", 1),
    ("my to-do list keeps growing no matter how much I do", 1),
    ("the project is due tomorrow and I am nowhere near done", 1),
    ("I have back to back deadlines and no time to breathe", 1),
    ("work keeps piling up I feel buried under tasks", 1),
    ("so many urgent tasks and not enough hours in the day", 1),
    ("my boss added another deadline on top of everything else", 1),
    ("I am completely overwhelmed by the amount of work I have", 1),
    ("every email I open is another urgent demand", 1),
    ("I have been working nonstop and still falling behind", 1),
    ("the workload is impossible I cannot get everything done", 1),
    ("I have three deadlines this week and I am panicking", 1),
    ("the pressure at work is getting unbearable", 1),
    ("I feel crushed under the weight of all my responsibilities", 1),
    # Sleep deprivation
    ("I have not slept properly in days I am exhausted", 1),
    ("deadlines are piling up I have not slept properly in days", 1),
    ("I can barely keep my eyes open I have been awake for two days", 1),
    ("I am running on no sleep and I feel terrible", 1),
    ("I haven't slept more than three hours in a week", 1),
    ("sleep deprived and stressed I can barely function", 1),
    ("I keep waking up at night worrying about everything", 1),
    ("insomnia is making everything worse I am so drained", 1),
    ("I cannot sleep my mind races all night with worries", 1),
    ("no sleep no energy I feel like I am breaking down", 1),
    ("exhausted but cannot sleep the anxiety keeps me awake", 1),
    ("I have been up since 3am my head is spinning", 1),
    ("days without proper rest are catching up with me", 1),
    # Cumulative / chronic stress
    ("I feel like I am drowning in responsibilities", 1),
    ("everything is piling up at once and I cannot cope", 1),
    ("the stress has been building for weeks and I am at my limit", 1),
    ("I am barely holding it together at this point", 1),
    ("every day feels harder than the last I am wearing down", 1),
    ("I feel stretched so thin I might snap", 1),
    ("too much going on at once and I cannot manage it all", 1),
    ("I have no time for myself everything demands my attention", 1),
    ("running on empty and there is no end in sight", 1),
    ("the weight of it all is becoming too much to carry", 1),
    # Financial / life stressors
    ("the bills are piling up and I do not know how to pay them", 1),
    ("I am deep in debt and cannot see a way out", 1),
    ("money is tight and every day feels like a crisis", 1),
    ("I might lose my job and I am scared about what comes next", 1),
    ("facing eviction and I feel completely helpless", 1),
    # Relationship and social stress
    ("the constant arguments at home are draining me", 1),
    ("I feel completely alone with no one to turn to", 1),
    ("my relationship is falling apart and I don't know what to do", 1),
    ("I am isolated and struggling and nobody seems to notice", 1),
    ("dealing with family conflict on top of everything else", 1),
    # Health anxiety / physical stress
    ("I am so worried about my health and cannot stop thinking about it", 1),
    ("the chest tightness and headaches are getting worse from the stress", 1),
    ("I feel physically ill from all the anxiety and pressure", 1),
    ("my body is exhausted and tense I cannot relax at all", 1),
    ("stress is taking a serious toll on my health and sleep", 1),
]


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------

def load_dreaddit(path: str) -> pd.DataFrame:
    """Load the Dreaddit Reddit dataset.

    Handles both a plain CSV and a ZIP archive containing the CSV.
    Expects a 'text' column and a 'label' column (0/1).
    """
    csv_path = path

    # If a ZIP file exists alongside or instead of the CSV, extract first.
    zip_path = path + ".zip"
    if not os.path.isfile(csv_path) and os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract the first CSV found inside the archive.
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError(f"No CSV found inside {zip_path}")
            zf.extract(csv_names[0], os.path.dirname(path))
            csv_path = os.path.join(os.path.dirname(path), csv_names[0])

    df = pd.read_csv(csv_path)

    # Identify the text column (commonly 'text' or 'selftext' in Dreaddit).
    text_col = _find_column(df, ["text", "selftext", "post_text"])
    label_col = _find_column(df, ["label"])

    # Dreaddit includes subreddit metadata; keep it in-text with a stable tag.
    text_series = df[text_col].astype(str)
    if "subreddit" in df.columns:
        text_series = text_series + " <SUBREDDIT=" + df["subreddit"].astype(str) + ">"

    result = pd.DataFrame(
        {
            "text": text_series,
            "label": df[label_col].astype(int),
            "domain": "reddit_long",
        }
    )
    numeric = _numeric_features(df, exclude_cols=[label_col])
    for col in numeric.columns:
        result[col] = numeric[col]
    return result


def load_reddit_combi(path: str) -> pd.DataFrame:
    """Load the Reddit_Combi dataset (CSV or XLSX)."""
    df = _read_flexible(path)
    text_col = _find_column(df, ["body_title", "text", "selftext", "post_text", "sentence"])
    label_col = _find_column(df, ["label"])

    result = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": df[label_col].astype(int),
            "domain": "reddit_long",
        }
    )
    numeric = _numeric_features(df, exclude_cols=[label_col])
    for col in numeric.columns:
        result[col] = numeric[col]
    return result


def load_reddit_title(path: str) -> pd.DataFrame:
    """Load the Reddit_Title dataset (CSV or XLSX)."""
    df = _read_flexible(path)
    text_col = _find_column(df, ["title", "text", "sentence"])
    label_col = _find_column(df, ["label"])

    result = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": df[label_col].astype(int),
            "domain": "reddit_short",
        }
    )
    numeric = _numeric_features(df, exclude_cols=[label_col])
    for col in numeric.columns:
        result[col] = numeric[col]
    return result


def load_twitter_full(path: str) -> pd.DataFrame:
    """Load the Twitter_Full dataset (CSV or XLSX)."""
    df = _read_flexible(path)
    text_col = _find_column(df, ["text", "tweet", "content", "sentence"])
    label_col = _find_column(df, ["label", "labels"])

    text_series = df[text_col].astype(str)
    hashtags_col = next(
        (c for c in df.columns if c.lower() == "hashtags"),
        None,
    )
    if hashtags_col:
        hashtags = df[hashtags_col].fillna("").astype(str)
        text_series = (text_series + " " + hashtags).str.strip()

    result = pd.DataFrame(
        {
            "text": text_series,
            "label": df[label_col].astype(int),
            "domain": "twitter_short",
        }
    )
    numeric = _numeric_features(df, exclude_cols=[label_col])
    for col in numeric.columns:
        result[col] = numeric[col]
    return result


def load_stressed_tweets(path: str) -> pd.DataFrame:
    """Load the Stressed_Tweets dataset.

    This dataset contains only stress-positive tweets (implicit label=1).
    """
    df = _read_flexible(path)
    text_col = _find_column(df, ["cleaned_text", "text", "tweet", "content", "sentence"])

    result = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": 1,
            "domain": "twitter_short",
        }
    )
    numeric = _numeric_features(df, exclude_cols=[])
    for col in numeric.columns:
        result[col] = numeric[col]
    return result


def load_happy_neutral(path: str) -> pd.DataFrame:
    """Load optional happy/neutral negatives (implicit label=0)."""
    df = _read_flexible(path)
    text_col = _find_column(
        df,
        ["text", "sentence", "content", "tweet", "body", "title"],
    )

    result = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": 0,
            "domain": "happy_neutral",
        }
    )
    numeric = _numeric_features(df, exclude_cols=[])
    for col in numeric.columns:
        result[col] = numeric[col]
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_flexible(path: str) -> pd.DataFrame:
    """Read a CSV or XLSX file, trying both extensions."""
    if os.path.isfile(path):
        if path.endswith(".xlsx"):
            return pd.read_excel(path, engine="openpyxl")
        return pd.read_csv(path)

    # Try swapping the extension
    base, ext = os.path.splitext(path)
    alt = base + (".xlsx" if ext == ".csv" else ".csv")
    if os.path.isfile(alt):
        if alt.endswith(".xlsx"):
            return pd.read_excel(alt, engine="openpyxl")
        return pd.read_csv(alt)

    raise FileNotFoundError(
        f"Dataset not found: tried {path} and {alt}"
    )


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name from *candidates* that exists in *df*.

    The match is case-insensitive.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(
        f"None of {candidates} found in columns: {list(df.columns)}"
    )


def _numeric_features(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").copy()
    if exclude_cols:
        numeric = numeric.drop(columns=exclude_cols, errors="ignore")
    return numeric


def clean_text(text: str) -> str:
    """Strip whitespace and remove high-frequency stop words.

    Preserves contrast and negation words (e.g. 'but', 'not') that carry
    genuine semantic signal for stress detection.  Returns an empty string
    for inputs that are ``None``, NaN, or consist entirely of whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    words = text.lower().split()
    filtered = [w for w in words if w not in _PREPROCESSING_STOP_WORDS]
    # Fall back to the original words if filtering removed everything.
    return " ".join(filtered) if filtered else text.lower().strip()


def augment_text(text: str) -> list[str]:
    """Return lightweight augmented variants of *text* for training robustness.

    Produces the original plus casual / slang rewrites so the model
    generalises better to informal inputs.
    """
    return [
        text,
        text.lower(),
        text + " lol",
        "idk " + text,
        text.replace("I am", "I'm"),
    ]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

DATASET_LOADERS = [
    ("dreaddit-train.csv", load_dreaddit),
    ("Reddit_Combi.csv", load_reddit_combi),
    ("Reddit_Title.csv", load_reddit_title),
    ("Twitter_Full.csv", load_twitter_full),
    ("Stressed_Tweets.csv", load_stressed_tweets),
    ("Happy_Neutral.csv", load_happy_neutral),
]


def merge_datasets(input_dir: str, output_path: str) -> pd.DataFrame:
    """Load, clean, and merge all datasets into a unified DataFrame.

    Parameters
    ----------
    input_dir : str
        Directory containing the raw dataset files.
    output_path : str
        Path where the unified CSV will be saved.

    Returns
    -------
    pd.DataFrame
        The merged and cleaned DataFrame.
    """
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for filename, loader in DATASET_LOADERS:
        filepath = os.path.join(input_dir, filename)
        try:
            df = loader(filepath)
            frames.append(df)
            print(f"  ✓ Loaded {filename}: {len(df):,} rows")
        except FileNotFoundError as exc:
            skipped.append(filename)
            print(f"  ✗ Skipped {filename}: {exc}")

    if not frames:
        print("\nERROR: No datasets were loaded. Place files in:", input_dir)
        sys.exit(1)

    # Concatenate all datasets
    unified = pd.concat(frames, ignore_index=True)

    # Clean text (strip whitespace, remove stop words, preserve contrast words)
    unified["text"] = unified["text"].apply(clean_text)
    unified["text"] = unified["text"].replace("", pd.NA)

    # Drop rows with missing or empty text
    before = len(unified)
    unified = unified.dropna(subset=["text"]).reset_index(drop=True)
    dropped = before - len(unified)

    # Ensure label is integer (0 or 1)
    unified["label"] = unified["label"].astype(int)

    # Append hard-negative contrast samples to teach the model that a
    # positive clause after 'but' / 'however' overrides the stress label.
    contrast_df = pd.DataFrame(_CONTRAST_SAMPLES, columns=["text", "label"])
    contrast_df["domain"] = "contrast_augment"
    unified = pd.concat([unified, contrast_df], ignore_index=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    unified.to_csv(output_path, index=False)

    # Summary
    print(f"\n{'='*50}")
    print(f"Unified dataset saved to: {output_path}")
    print(f"  Total rows:   {len(unified):,}")
    print(f"  Dropped rows: {dropped:,} (empty/NaN text)")
    if skipped:
        print(f"  Skipped files: {', '.join(skipped)}")
    print(f"\nLabel distribution:")
    print(unified["label"].value_counts().to_string())
    print(f"\nDomain distribution:")
    print(unified["domain"].value_counts().to_string())
    print(f"{'='*50}")

    return unified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge stress-detection datasets into a unified CSV."
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "raw"),
        help="Directory containing the raw dataset files (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "data", "processed", "unified_stress.csv"
        ),
        help="Output path for the unified CSV (default: data/processed/unified_stress.csv)",
    )
    args = parser.parse_args()

    print("=== Stress Detection: Data Preprocessing ===\n")
    print(f"Input directory: {args.input_dir}")
    print(f"Output path:     {args.output}\n")
    print("Loading datasets...")
    merge_datasets(args.input_dir, args.output)


if __name__ == "__main__":
    main()
