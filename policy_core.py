"""Shared logic for the Company Policy Assistant (used by the Streamlit site and the Slack bot)."""
import json
import math
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

CSV_PATH = os.path.join(os.path.dirname(__file__), "company_policies.csv")
# Gemini model names change often. Override them with environment variables / Streamlit secrets
# (GEMINI_MODEL, GEMINI_EMBED_MODEL) after running test_gemini.py to see what your key can use.
CHAT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")

NOT_FOUND = "No matching policy found in the policy database."

SYSTEM_PROMPT = (
    "You are a company policy assistant. Answer the employee's question using ONLY the "
    "policies provided. Do not use outside knowledge and do not guess. "
    "If the provided policies do not answer the question, say so and set policy_title to null. "
    'Reply as JSON: {"answer": str, "policy_title": str | null}. '
    "policy_title must be copied exactly from the policy list."
)


# ---------- data ----------
def load_policies(path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["doc"] = df["title"] + ": " + df["policy_text"]
    return df


def get_client(api_key: str | None = None):
    return genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))


def _retry(call, tries: int = 4):
    """Run call() -> (result, seconds). Retries on rate-limit errors (free tier is strict).
    Only the seconds of the successful attempt are reported, so waiting does not distort timings."""
    delays = [8, 20, 40]
    for attempt in range(tries):
        try:
            return call()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg
            if not limited or attempt == tries - 1:
                raise
            time.sleep(delays[min(attempt, len(delays) - 1)])


def _embed(client, texts: list[str], task: str):
    def call():
        t0 = time.perf_counter()
        r = client.models.embed_content(
            model=EMBED_MODEL, contents=texts, config=types.EmbedContentConfig(task_type=task)
        )
        return r, time.perf_counter() - t0

    return _retry(call)


def _generate(client, prompt: str, system: str):
    def call():
        t0 = time.perf_counter()
        r = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0, response_mime_type="application/json"
            ),
        )
        return r, time.perf_counter() - t0

    return _retry(call)


def _usage(r):
    u = r.usage_metadata
    prompt = (u.prompt_token_count or 0) if u else 0
    total = (u.total_token_count or 0) if u else 0
    return prompt, max(total - prompt, 0)  # output includes any hidden "thinking" tokens


def _parse_json(text: str, default: dict) -> dict:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else default
    except Exception:  # noqa: BLE001
        return default


def _empty(answer=NOT_FOUND, policy=None, seconds=0.0, in_tok=0, out_tok=0, emb_tok=0, extra=None):
    return {
        "answer": answer,
        "policy": policy,
        "seconds": seconds,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "embedding_tokens": emb_tok,
        "total_tokens": in_tok + out_tok + emb_tok,
        "extra": extra or {},
    }


# ---------- 1. rules-based ----------
STOP = set(
    "a an the is are was were be to of in on at for and or do does did i my me we our you your can "
    "could should would will may might what when where which who how many much if it its this that "
    "there any have has had about from with by as not no get need must employee employees company "
    "policy rule rules".split()
)


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class RulesIndex:
    """Keyword matching with IDF weighting. No ML, no synonyms: only literal word overlap."""

    MIN_SCORE = 3.0

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.title_tok = [set(_tokens(t)) for t in df["title"]]
        self.text_tok = [set(_tokens(t)) for t in df["policy_text"]]
        n = len(df)
        dfreq = Counter()
        for a, b in zip(self.title_tok, self.text_tok):
            dfreq.update(a | b)
        self.idf = {w: math.log(n / c) + 1 for w, c in dfreq.items()}

    def search(self, question: str) -> dict:
        t0 = time.perf_counter()
        q = set(_tokens(question))
        best, best_score = None, 0.0
        for i in range(len(self.df)):
            score = sum(self.idf[w] * 2 for w in q & self.title_tok[i])
            score += sum(self.idf[w] for w in (q & self.text_tok[i]) - self.title_tok[i])
            if score > best_score:
                best, best_score = i, score
        dt = time.perf_counter() - t0
        if best is None or best_score < self.MIN_SCORE:
            return _empty(seconds=dt, extra={"score": round(best_score, 2)})
        row = self.df.iloc[best]
        return _empty(
            answer=row["policy_text"], policy=row["title"], seconds=dt, extra={"score": round(best_score, 2)}
        )


# ---------- 2. LLM, full context (no vector index) ----------
def _clean_title(t):
    if t is None:
        return None
    t = str(t).strip()
    return None if t.lower() in ("", "none", "null", "n/a") else t


def _ask_llm(client, context: str, question: str):
    r, dt = _generate(client, f"POLICIES:\n{context}\n\nQUESTION: {question}", SYSTEM_PROMPT)
    data = _parse_json(r.text or "", {"answer": r.text or "", "policy_title": None})
    in_tok, out_tok = _usage(r)
    return data, dt, in_tok, out_tok


def llm_full_context(client, df: pd.DataFrame, question: str) -> dict:
    policies = "\n".join(f"- {r.title}: {r.policy_text}" for r in df.itertuples())
    data, dt, in_tok, out_tok = _ask_llm(client, policies, question)
    return _empty(
        answer=data.get("answer", ""),
        policy=_clean_title(data.get("policy_title")),
        seconds=dt,
        in_tok=in_tok,
        out_tok=out_tok,
    )


# ---------- 3. LLM + vector index ----------
class VectorIndex:
    def __init__(self, client, df: pd.DataFrame, cache_path: str | None = None):
        self.client, self.df = client, df
        if cache_path and os.path.exists(cache_path):
            self.emb = np.load(cache_path)
        else:
            vecs = []
            docs = df["doc"].tolist()
            for i in range(0, len(docs), 50):  # small batches are gentler on free-tier limits
                r, _ = _embed(client, docs[i : i + 50], "RETRIEVAL_DOCUMENT")
                vecs += [e.values for e in r.embeddings]
            self.emb = np.array(vecs, dtype="float32")
            if cache_path:
                try:
                    np.save(cache_path, self.emb)
                except OSError:
                    pass
        self.emb = self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)

    def ask(self, question: str, k: int = 3) -> dict:
        r, emb_dt = _embed(self.client, [question], "RETRIEVAL_QUERY")
        q = np.array(r.embeddings[0].values, dtype="float32")
        q /= np.linalg.norm(q)
        sims = self.emb @ q
        top = np.argsort(-sims)[:k]
        context = "\n".join(f"- {self.df.iloc[i]['title']}: {self.df.iloc[i]['policy_text']}" for i in top)
        data, chat_dt, in_tok, out_tok = _ask_llm(self.client, context, question)
        return _empty(
            answer=data.get("answer", ""),
            policy=_clean_title(data.get("policy_title")),
            seconds=emb_dt + chat_dt,
            in_tok=in_tok,
            out_tok=out_tok,
            emb_tok=max(len(question) // 4, 1),  # estimate: the Gemini embedding API does not report tokens
            extra={"retrieved": [(self.df.iloc[i]["title"], round(float(sims[i]), 3)) for i in top]},
        )


# ---------- grounding check ----------
def check_unsupported(df: pd.DataFrame, expected_covered: bool | None, result: dict) -> bool:
    """Cheap check, no extra API call. An answer is unsupported when it cites a policy that is not in
    the database, or when it gives an answer although the database has no relevant policy
    (only knowable for benchmark questions, where expected_covered is False)."""
    if result["policy"] is None:
        return False
    if df[df["title"].str.lower() == result["policy"].lower()].empty:
        return True
    return expected_covered is False


def check_grounding(client, df: pd.DataFrame, question: str, result: dict) -> dict:
    """Stronger, optional check that costs one extra API call per answer.

    - title_in_db: the cited policy must exist in the CSV (catches invented policies)
    - judge_supported: a separate LLM call checks the answer is fully backed by the cited policy text
    A declined answer (policy None) is never counted as unsupported.
    """
    if result["policy"] is None:
        return {"declined": True, "title_in_db": True, "judge_supported": True, "unsupported": False}
    match = df[df["title"].str.lower() == result["policy"].lower()]
    if match.empty:
        return {"declined": False, "title_in_db": False, "judge_supported": False, "unsupported": True}
    policy_text = match.iloc[0]["policy_text"]
    system = (
        "You are a strict fact checker. Decide whether every factual claim in the ANSWER "
        'is stated in the POLICY TEXT. Reply as JSON: {"supported": true or false}.'
    )
    r, _ = _generate(
        client, f"QUESTION: {question}\nPOLICY TEXT: {policy_text}\nANSWER: {result['answer']}", system
    )
    ok = bool(_parse_json(r.text or "", {}).get("supported"))
    return {"declined": False, "title_in_db": True, "judge_supported": ok, "unsupported": not ok}


# ---------- benchmark questions ----------
# (question, expected policy title or None when the database has no relevant policy)
TEST_QUESTIONS = [
    ("How many vacation days do I get per year?", "Vacation Policy"),
    ("Can I carry unused vacation days into next year?", "Vacation Policy"),
    ("Can I work from another country for a month?", "Work From Abroad Policy"),
    ("How long is the probation period for new hires?", "Probation Period Policy"),
    ("Do I need approval to buy something for $800?", "Expense Approval Policy"),
    ("How often do I have to change my password?", "Password Policy"),
    ("Can I give an interview to a journalist?", "Media Contact Policy"),
    ("How much paid parental leave can I take?", "Parental Leave Policy"),
    ("By when must I submit receipts for a business trip?", "Travel Reimbursement Policy"),
    ("Can I install my own software on my work laptop?", "Software Installation Policy"),
    ("How many days off do I get if a family member dies?", "Bereavement Leave Policy"),
    ("What do I do if I notice a security breach?", "Security Incident Policy"),
    ("Can I report misconduct without giving my name?", "Whistleblower Policy"),
    ("Do I need a VPN when connecting from home?", "Remote Access Policy"),
    ("How many days a week can I work from home?", "Remote Work Policy"),
    ("Can I bring my dog to the office?", None),
    ("What is the company pension contribution?", None),
    ("How big is the annual bonus for managers?", None),
    ("Does the company pay for a gym membership?", None),
]
