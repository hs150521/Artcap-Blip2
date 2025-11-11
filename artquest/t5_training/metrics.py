import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.meteor_score import meteor_score

import evaluate
from sacrebleu.metrics import BLEU

gleu = evaluate.load('google_bleu')
rouge = evaluate.load('rouge')

_WORDNET_READY = False
_PUNKT_READY = False


def _ensure_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def _ensure_wordnet():
    global _WORDNET_READY
    if not _WORDNET_READY:
        _ensure_resource('corpora/wordnet', 'wordnet')
        _ensure_resource('corpora/omw-1.4', 'omw-1.4')
        _WORDNET_READY = True


def _ensure_punkt():
    global _PUNKT_READY
    if not _PUNKT_READY:
        _ensure_resource('tokenizers/punkt', 'punkt')
        _PUNKT_READY = True


def _tokenize(text):
    _ensure_punkt()
    try:
        return word_tokenize(text)
    except LookupError:
        # Fallback if Punkt is unavailable; whitespace split still satisfies METEOR input requirements.
        return text.split()


def calc_gleu(answer, pred):
    return gleu.compute(predictions=pred, references=answer)["google_bleu"]


def calc_meteor(answer, pred):
    if len(answer) == 0:
        return 0
    _ensure_wordnet()
    sum_scores = 0.0
    for a, p in zip(answer, pred):
        ref_tokens = _tokenize(a)
        hyp_tokens = _tokenize(p)
        sum_scores += meteor_score([ref_tokens], hyp_tokens)
    return sum_scores / len(answer)


def calc_rouge(answer, pred):
    return rouge.compute(predictions=pred, references=answer)["rougeL"]


def calc_bleu_sentence(answer, pred):
    sum = 0
    for a, p in zip(answer, pred):
        sum += BLEU(effective_order=True).sentence_score(
            hypothesis=a,
            references=[p]
        ).score/100
    if len(answer)==0:
        return 0
    return sum/len(answer)


def calc_em(answer, pred):
    cnt = 0
    for a, p in zip(answer, pred):
        if a == p:
            cnt += 1
    if len(answer) == 0:
        return 0
    return cnt/len(answer)


def eval_df(df):
    metrics = {}
    metrics["em"] = calc_em(df.answer.tolist(), df.pred.tolist())
    metrics["bleu"] = calc_bleu_sentence(df.answer.tolist(), df.pred.tolist())
    metrics["gleu"] = calc_gleu(df.answer.tolist(), df.pred.tolist())
    metrics["meteor"] = calc_meteor(df.answer.tolist(), df.pred.tolist())
    metrics["rougeL"] = calc_rouge(df.answer.tolist(), df.pred.tolist())
    return metrics

def eval_preds(answers, preds):
    metrics = {}
    metrics["em"] = calc_em(answers, preds)
    metrics["bleu"] = calc_bleu_sentence(answers, preds)
    metrics["gleu"] = calc_gleu(answers, preds)
    metrics["meteor"] = calc_meteor(answers, preds)
    metrics["rougeL"] = calc_rouge(answers, preds)
    return metrics

def eval_preds_rounded_strings(answers, preds):
    metrics = {}
    metrics["em"] = "{:.3f}".format(calc_em(answers, preds))
    metrics["bleu"] = "{:.3f}".format(calc_bleu_sentence(answers, preds))
    metrics["gleu"] = "{:.3f}".format(calc_gleu(answers, preds))
    metrics["meteor"] = "{:.3f}".format(calc_meteor(answers, preds))
    metrics["rougeL"] = "{:.3f}".format(calc_rouge(answers, preds))
    return metrics

def calc_metrics(df, rounded=True):
    metrics = {
        "em": [],
        "bleu": [],
        "gleu": [],
        "meteor": [],
        "rougeL": [],
    }
    answers = df.answer.apply(lambda x: x.lower()).tolist()
    preds = df.pred.apply(lambda x: x.lower()).tolist()
    metrics["em"] = calc_em(answers, preds)
    metrics["bleu"] = calc_bleu_sentence(answers, preds)
    metrics["gleu"] = calc_gleu(answers, preds)
    metrics["meteor"] = calc_meteor(answers, preds)
    metrics["rougeL"] = calc_rouge(answers, preds)
    if rounded:
        for k in metrics.keys():
            metrics[k] = round(metrics[k], 3)
    return metrics


def get_type_accuracy(df, rounded=True):
    df = df.copy()
    types = ["artist", "school", "timeframe", "type", "technique", "title"]
    df["answer"] = df["answer"].apply(lambda x: x.lower())
    df["pred"] = df["pred"].apply(lambda x: x.lower())
    type_accs = {}
    for t in types:
        tmp = df[df["question_type"] == t]
        type_accs[t] = calc_em(tmp["answer"].tolist(), tmp["pred"].tolist())
    if rounded:
        for k in type_accs.keys():
            type_accs[k] = round(type_accs[k], 3)
    return type_accs


def get_type_bleu(df, rounded=True):
    df = df.copy()
    types = ["artist", "school", "timeframe", "type", "technique", "title"]
    df["answer"] = df["answer"].apply(lambda x: x.lower())
    df["pred"] = df["pred"].apply(lambda x: x.lower())
    type_bleus = {}
    for t in types:
        tmp = df[df["question_type"] == t]
        type_bleus[t] = calc_bleu_sentence(
            tmp["answer"].tolist(), tmp["pred"].tolist())
    if rounded:
        for k in type_bleus.keys():
            type_bleus[k] = round(type_bleus[k], 3)
    return type_bleus


def remove_stopwords(preds):
    # used for zero-shot visual evaluations
    all_stopwords = stopwords.words('english')
    all_stopwords += ["One", "Two", "Three", "Four", "Five",
                      "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"]
    all_stopwords += [n.lower() for n in all_stopwords]
    out = []
    for p in preds:
        text_tokens = word_tokenize(p)
        text_tokens = [
            word for word in text_tokens if not word in all_stopwords]
        out.append((" ").join(text_tokens))
    return out
