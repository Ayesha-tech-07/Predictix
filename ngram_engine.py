"""N-gram language model engine for PredictiX (CS525PE, Unit V: Language Modeling).

This file contains only plain Python. It has no Streamlit code, so every
function here can be tested and explained on its own.

Key ideas
---------
* A sentence is padded with two start markers <s> and one end marker </s>,
  so the model can learn how sentences begin and end.
* Unigram  : P(w)            = C(w) / N
* Bigram   : P(w | v)        = C(v, w) / C(v)
* Trigram  : P(w | u, v)     = C(u, v, w) / C(u, v)
* Add-k smoothing: P = (count + k) / (context count + k * V). With k = 1
  this is Laplace (add-one) smoothing.
* Perplexity = exp( -(1/N) * sum(log P(w_i | context)) ). Lower is better.
"""

import math
import random
import re
from collections import Counter, defaultdict

BOS = "<s>"      # beginning-of-sentence marker
EOS = "</s>"     # end-of-sentence marker
ORDER_NAMES = {1: "Unigram", 2: "Bigram", 3: "Trigram"}


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def tokenize(text):
    """Lower-case the text and return a list of words (letters, digits, apostrophes)."""
    text = text.lower().replace("\u2019", "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def split_sentences(text):
    """Split text at . ! ? and line breaks, then tokenize each sentence."""
    parts = re.split(r"[.!?\n]+", text)
    return [words for words in (tokenize(p) for p in parts) if words]


def context_words(text):
    """Words of the sentence currently being typed (everything after the last . ! ?)."""
    last = re.split(r"[.!?\n]+", text)[-1]
    return tokenize(last)


def show(token):
    """Human-friendly label for a token (markers become readable)."""
    if token == EOS:
        return "\u27e8end\u27e9"
    if token == BOS:
        return "\u27e8start\u27e9"
    return token


def detokenize(tokens):
    """Turn generated tokens back into a readable sentence with capital letters."""
    text, capitalize = "", True
    for tok in tokens:
        if tok == ".":
            text += "."
            capitalize = True
            continue
        word = "I" if tok == "i" else re.sub(r"^i'", "I'", tok)
        if capitalize:
            word = word[:1].upper() + word[1:]
            capitalize = False
        text += (" " if text else "") + word
    if text and not text.endswith("."):
        text += "\u2026"
    return text


def finite(value, digits=4):
    """JSON-safe number: infinity becomes the string 'infinity'."""
    if value is None:
        return None
    if isinstance(value, float) and math.isinf(value):
        return "infinity"
    return round(value, digits)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
class NGramModel:
    """Counts unigrams, bigrams and trigrams and answers questions about them."""

    def __init__(self, text):
        self.sentences = split_sentences(text)
        if not self.sentences:
            raise ValueError("The training text contains no words.")

        uni, bi, tri = Counter(), Counter(), Counter()
        next2, next3 = defaultdict(Counter), defaultdict(Counter)

        for words in self.sentences:
            padded = [BOS, BOS] + words + [EOS]
            for i in range(2, len(padded)):
                w = padded[i]
                uni[w] += 1
                bi[(padded[i - 1], w)] += 1
                tri[(padded[i - 2], padded[i - 1], w)] += 1
                next2[padded[i - 1]][w] += 1
                next3[(padded[i - 2], padded[i - 1])][w] += 1

        self.uni, self.bi, self.tri = uni, bi, tri
        self.next2, self.next3 = dict(next2), dict(next3)
        self.ctx2 = {key: sum(c.values()) for key, c in self.next2.items()}
        self.ctx3 = {key: sum(c.values()) for key, c in self.next3.items()}
        self.N = sum(uni.values())                      # total predicted tokens
        self.V = len(uni) + 1                           # vocabulary + <unk>
        self.vocab_words = {w for w in uni if w != EOS}

    # ---- counts and probabilities -----------------------------------------
    def ngram_count(self, w, context, order):
        """How many times the n-gram (context + w) occurs in the training text."""
        if order == 1:
            return self.uni.get(w, 0)
        if order == 2:
            return self.bi.get((context[-1], w), 0)
        return self.tri.get((context[-2], context[-1], w), 0)

    def prob(self, w, context, order, k=0.0):
        """P(w | context). k = 0 is plain maximum likelihood; k > 0 is add-k smoothing."""
        count = self.ngram_count(w, context, order)
        if order == 1:
            total = self.N
        elif order == 2:
            total = self.ctx2.get(context[-1], 0)
        else:
            total = self.ctx3.get((context[-2], context[-1]), 0)
        denominator = total + k * self.V
        return (count + k) / denominator if denominator > 0 else 0.0

    # ---- prediction ---------------------------------------------------------
    def candidates(self, context, order):
        """Words that followed this context in training, with simple back-off.

        Tries the trigram context first, then the bigram, then falls back to
        unigram frequencies. Returns (Counter of candidates, order actually used).
        """
        ctx = ([BOS, BOS] + list(context))[-2:]
        for n in range(order, 1, -1):
            found = self.next3.get((ctx[0], ctx[1])) if n == 3 else self.next2.get(ctx[1])
            if found:
                return found, n
        return self.uni, 1

    def predict(self, words, order, top_k=5):
        """Top-k next-word suggestions as (token, count, probability)."""
        cands, used = self.candidates(words, order)
        total = sum(cands.values())
        ranked = sorted(cands.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return {
            "items": [(w, c, c / total) for w, c in ranked],
            "used_order": used,
            "requested_order": order,
            "context_total": total,
        }

    def context_label(self, words, used_order):
        """Readable description of the context the model looked at."""
        ctx = ([BOS, BOS] + list(words))[-2:]
        need = used_order - 1
        if need <= 0:
            return "no context"
        return " ".join(show(t) for t in ctx[-need:])

    # ---- generation ---------------------------------------------------------
    def generate(self, seed_words, n_words, order, creativity=1.0, seed=0):
        """Generate text word by word by sampling from the model.

        creativity < 1 favours the most likely words; creativity > 1 flattens
        the distribution so rarer words are chosen more often.
        """
        rng = random.Random(seed)
        power = 1.0 / max(creativity, 0.05)
        context = list(seed_words)
        out = list(seed_words)
        made, guard = 0, 0
        while made < n_words and guard < n_words * 5 + 20:
            guard += 1
            cands, _ = self.candidates(context, order)
            words = sorted(cands)
            weights = [cands[w] ** power for w in words]
            pick = rng.choices(words, weights=weights, k=1)[0]
            if pick == EOS:
                if out and out[-1] != ".":
                    out.append(".")
                context = []
                continue
            out.append(pick)
            context.append(pick)
            made += 1
        return out

    # ---- evaluation ---------------------------------------------------------
    def evaluate(self, text, order, k=0.0):
        """Perplexity of `text` under the chosen model, plus a word-by-word table."""
        rows, log_sum, count, zero_prob = [], 0.0, 0, False
        unseen_words = []
        for words in split_sentences(text):
            padded = [BOS, BOS] + words + [EOS]
            for w in words:
                if w not in self.vocab_words and w not in unseen_words:
                    unseen_words.append(w)
            for i in range(2, len(padded)):
                p = self.prob(padded[i], padded[:i], order, k)
                seen = self.ngram_count(padded[i], padded[:i], order)
                ctx = padded[i - (order - 1):i] if order > 1 else []
                rows.append({
                    "word": show(padded[i]),
                    "context": " ".join(show(t) for t in ctx) or "none",
                    "probability": p,
                    "count": seen,
                })
                count += 1
                if p > 0:
                    log_sum += math.log(p)
                else:
                    zero_prob = True
        if count == 0:
            return None
        perplexity = math.inf if zero_prob else math.exp(-log_sum / count)
        return {"perplexity": perplexity, "tokens": count, "rows": rows,
                "unseen_words": unseen_words}

    # ---- smoothing lab ------------------------------------------------------
    def bigram_report(self, w1, w2, k):
        """Compare MLE, Laplace and add-k estimates of P(w2 | w1)."""
        c_ctx = self.ctx2.get(w1, 0)
        c_pair = self.bi.get((w1, w2), 0)
        seen = self.next2.get(w1, {})
        seen_mass = sum((c + k) / (c_ctx + k * self.V) for c in seen.values())
        return {
            "c_ctx": c_ctx,
            "c_pair": c_pair,
            "V": self.V,
            "mle": (c_pair / c_ctx) if c_ctx else None,
            "laplace": (c_pair + 1) / (c_ctx + self.V),
            "add_k": (c_pair + k) / (c_ctx + k * self.V),
            "unseen_mass": max(0.0, 1.0 - seen_mass),
        }

    # ---- corpus statistics --------------------------------------------------
    def stats(self):
        word_counts = {w: c for w, c in self.uni.items() if w != EOS}
        once = sum(1 for c in word_counts.values() if c == 1)
        tokens = sum(len(s) for s in self.sentences)
        return {
            "sentences": len(self.sentences),
            "tokens": tokens,
            "vocabulary": len(word_counts),
            "seen_once_pct": once / len(word_counts) if word_counts else 0.0,
            "unique_bigrams": len(self.bi),
            "unique_trigrams": len(self.tri),
            "avg_sentence_length": tokens / len(self.sentences),
        }

    def top_ngrams(self, order, n=10):
        """Most frequent n-grams as (label, count)."""
        table = {1: self.uni, 2: self.bi, 3: self.tri}[order]
        ranked = sorted(table.items(), key=lambda item: (-item[1], str(item[0])))[:n]
        out = []
        for key, count in ranked:
            parts = key if isinstance(key, tuple) else (key,)
            out.append((" ".join(show(t) for t in parts), count))
        return out
