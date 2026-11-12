# PredictiX: N-gram Language Model Lab

A Streamlit app that trains unigram, bigram and trigram language models on a text of your choice,
then predicts the next word, generates text, scores sentences with perplexity and demonstrates
add-k (Laplace) smoothing.

Course: Natural Language Processing (CS525PE), Unit V: Language Modeling.

## Features
1. Next-word predictor with back-off and tap-to-add suggestions
2. Text generator with a creativity control
3. Perplexity check for unigram, bigram and trigram models, with and without smoothing
4. Smoothing lab comparing maximum likelihood, Laplace and add-k
5. Corpus insights with top n-grams and statistics

Every tab can export its results as JSON.

## Project structure
```
app.py                  Streamlit user interface
ngram_engine.py         N-gram model: counts, probabilities, prediction, generation, perplexity
corpora.py              Three built-in training texts
requirements.txt        Python dependencies
.streamlit/config.toml  Pastel theme
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository (keep `app.py` at the root).
2. Go to share.streamlit.io and sign in with GitHub.
3. Choose New app, select the repository and branch, set the main file to `app.py`.
4. Click Deploy. Your public link appears in a minute or two.

