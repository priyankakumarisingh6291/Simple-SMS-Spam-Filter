# SMS Spam Detection

A simple SMS spam classifier built with Python and NLTK. It learns which words show up in spam versus legitimate ("ham") messages from the UCI SMS Spam Collection, then classifies new messages by comparing those word counts.

## How it works

1. **Load data**: the SMS Spam Collection (tab-separated: `label<TAB>message`).
2. **Pre-process**: lowercase, strip punctuation, tokenize with NLTK, remove English stopwords.
3. **Count words**: build separate frequency counts for words in spam and ham messages.
4. **Predict**: sum the spam and ham counts for every word in a new message. Whichever total is higher decides the label, and the share of that total is reported as a confidence score.

## Project structure

```
sms-spam-detection/
├── data/
│   └── README.md          # where to get SMSSpamCollection.txt
├── src/
│   └── spam_detector.py   # preprocessing, training, prediction, CLI
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/<your-username>/sms-spam-detection.git
cd sms-spam-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset and place `SMSSpamCollection.txt` in `data/` (see [data/README.md](data/README.md)).

## Usage

Interactive prompt:

```bash
python src/spam_detector.py
```

Classify a single message:

```bash
python src/spam_detector.py --message "WINNER!! You have won a free prize. Call now to claim"
```

Use a dataset stored elsewhere:

```bash
python src/spam_detector.py --data /path/to/SMSSpamCollection.txt
```

Example output:

```
Result
Message is spam, with 87.50% confidence
```

(The exact percentage depends on the message and dataset.)

## Use it in your own code

```python
from src.spam_detector import SpamDetector, load_data, download_nltk_resources

download_nltk_resources()
detector = SpamDetector().fit(load_data("data/SMSSpamCollection.txt"))
label, confidence = detector.predict("Free entry in a weekly competition")
```

## Limitations

- The "confidence" is the share of word counts favouring the winning class, not a calibrated probability or a measured accuracy.
- Ham messages far outnumber spam in the dataset, so raw counts lean toward ham.
- Words never seen in training are ignored.

Possible improvements: Naive Bayes or TF-IDF with logistic regression, a train/test split with precision and recall, and class-balanced counts.

## Tech stack

Python, NLTK, pandas

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) from the UCI Machine Learning Repository.

## License

MIT, see [LICENSE](LICENSE).
