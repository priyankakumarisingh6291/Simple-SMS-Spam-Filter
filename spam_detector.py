"""SMS spam detection using word-frequency counts.

Pipeline:
    1. Load the UCI SMS Spam Collection (tab-separated: label <TAB> message).
    2. Pre-process each message (lowercase, strip punctuation, tokenize,
       remove English stopwords).
    3. Count how often each word appears in spam and in ham messages.
    4. Classify a new message by comparing the summed spam vs. ham counts
       of its words.
"""

import argparse
import string
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "SMSSpamCollection.txt"
)
PUNCTUATION = set(string.punctuation)


def download_nltk_resources():
    """Download the NLTK data needed for tokenizing and stopword removal."""
    for resource in ("stopwords", "punkt", "punkt_tab"):
        nltk.download(resource, quiet=True)


def load_data(path):
    """Load the SMS Spam Collection into a DataFrame with 'label' and 'sms'."""
    return pd.read_csv(path, sep="\t", header=None, names=["label", "sms"])


def pre_process(sms, stop_words):
    """Lowercase, strip punctuation, tokenize and drop stopwords."""
    lowercase = "".join(char.lower() for char in sms if char not in PUNCTUATION)
    tokens = nltk.tokenize.word_tokenize(lowercase)
    return [word for word in tokens if word not in stop_words]


class SpamDetector:
    """Word-count based spam/ham classifier."""

    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words("english"))
        self.spam_words = Counter()
        self.ham_words = Counter()

    def fit(self, data):
        """Build spam and ham word counts from a labelled DataFrame."""
        data = data.copy()
        data["processed"] = data["sms"].apply(
            lambda sms: pre_process(sms, self.stop_words)
        )
        for label, tokens in zip(data["label"], data["processed"]):
            if label == "spam":
                self.spam_words.update(tokens)
            elif label == "ham":
                self.ham_words.update(tokens)
        return self

    def predict(self, message):
        """Return (label, confidence_percent) for a raw message string.

        label is "spam", "ham" or "uncertain" (when the counts are tied).
        """
        tokens = pre_process(message, self.stop_words)
        spam_count = sum(self.spam_words[word] for word in tokens)
        ham_count = sum(self.ham_words[word] for word in tokens)
        total = spam_count + ham_count

        if spam_count > ham_count:
            return "spam", spam_count / total * 100
        if ham_count > spam_count:
            return "ham", ham_count / total * 100
        return "uncertain", 50.0


def format_result(label, confidence):
    if label == "spam":
        return f"Message is spam, with {confidence:.2f}% confidence"
    if label == "ham":
        return f"Message is not spam, with {confidence:.2f}% confidence"
    return "Message could be spam or not spam (50% confidence)"


def main():
    parser = argparse.ArgumentParser(description="SMS spam detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="path to SMSSpamCollection.txt (default: data/SMSSpamCollection.txt)",
    )
    parser.add_argument(
        "--message",
        type=str,
        help="message to classify (omit to be prompted interactively)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        parser.error(
            f"dataset not found at {args.data}. See data/README.md for download steps."
        )

    download_nltk_resources()
    detector = SpamDetector().fit(load_data(args.data))

    message = args.message
    if message is None:
        message = input(
            "Please type a spam or ham message to check if our function predicts properly:\n"
        )

    label, confidence = detector.predict(message)
    print("Result")
    print(format_result(label, confidence))


if __name__ == "__main__":
    main()
