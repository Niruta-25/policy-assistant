"""Company Policy Assistant: compares rules-based, LLM (full context) and LLM + vector index."""
import os
import re
import time

import pandas as pd
import streamlit as st

from policy_core import (
    CHAT_MODEL,
    EMBED_MODEL,
    TEST_QUESTIONS,
    RulesIndex,
    VectorIndex,
    check_grounding,
    check_unsupported,
    get_client,
    llm_full_context,
    load_policies,
)

st.set_page_config(page_title="Company Policy Assistant", page_icon="📋", layout="wide")


# ---------- setup ----------
def _api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


@st.cache_data
def _policies():
    return load_policies()


@st.cache_resource
def _rules(_df):
    return RulesIndex(_df)


@st.cache_resource
def _vector(_df, key):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", EMBED_MODEL)
    return VectorIndex(get_client(key), _df, cache_path=f"embeddings_cache_{safe}.npy")


df = _policies()
key = _api_key()

st.title("📋 Company Policy Assistant")
st.caption(
    f"{len(df)} policies · three ways to answer the same question · models: {CHAT_MODEL} + {EMBED_MODEL}"
)

if not key:
    st.error("GEMINI_API_KEY is not set. Add it under Settings → Secrets (Streamlit Cloud) or as an environment variable.")
    st.stop()

client = get_client(key)
rules = _rules(df)
try:
    vindex = _vector(df, key)
except Exception as e:  # keep the page alive so the error is visible instead of a blank screen
    st.error(f"Could not build the vector index: {e}")
    st.stop()

tab_ask, tab_bench, tab_summary = st.tabs(["Ask a question", "Benchmark", "Comparison"])

APPROACHES = ["Rules-based", "LLM, full context", "LLM + vector index"]


def run_all(question: str) -> dict:
    return {
        APPROACHES[0]: rules.search(question),
        APPROACHES[1]: llm_full_context(client, df, question),
        APPROACHES[2]: vindex.ask(question),
    }


# ---------- tab 1: live comparison ----------
with tab_ask:
    examples = ["How many vacation days do I get?", "Can I talk to a journalist?", "Can I bring my dog to work?"]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        if c.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    question = st.text_input("Your policy question", key="q", placeholder="e.g. How often must I change my password?")
    verify = st.checkbox(
        "Fact-check answers with an extra LLM call (uses more of your free quota; not counted in tokens)", value=False
    )

    if question:
        with st.spinner("Running all three approaches…"):
            results = run_all(question)
            checks = {n: check_grounding(client, df, question, r) if verify else None for n, r in results.items()}
            invented = {n: check_unsupported(df, None, r) for n, r in results.items()}

        cols = st.columns(3)
        for col, name in zip(cols, APPROACHES):
            r, ck = results[name], checks[name]
            with col:
                st.subheader(name)
                st.markdown(f"**Answer**\n\n{r['answer']}")
                st.markdown(f"**Relevant policy:** {r['policy'] or '— none —'}")
                m1, m2 = st.columns(2)
                m1.metric("Response time", f"{r['seconds']:.2f} s")
                m2.metric("Tokens", f"{r['total_tokens']:,}")
                if r["embedding_tokens"]:
                    st.caption(f"{r['input_tokens']} in + {r['output_tokens']} out + ≈{r['embedding_tokens']} embedding (estimated)")
                elif r["total_tokens"]:
                    st.caption(f"{r['input_tokens']} in + {r['output_tokens']} out")
                else:
                    st.caption("No LLM used")
                if invented[name]:
                    st.error("⚠️ Cites a policy that is not in the database")
                elif ck is not None:
                    if ck["declined"]:
                        st.info("Declined: no matching policy")
                    elif ck["unsupported"]:
                        st.error("⚠️ Not supported by the policy database")
                    else:
                        st.success("✓ Supported by the cited policy")
                elif r["policy"] is None:
                    st.info("Declined: no matching policy")
                if "retrieved" in r["extra"]:
                    st.caption("Retrieved: " + " · ".join(f"{t} ({s})" for t, s in r["extra"]["retrieved"]))
                if "score" in r["extra"]:
                    st.caption(f"Keyword score: {r['extra']['score']}")

# ---------- tab 2: benchmark ----------
with tab_bench:
    st.write(
        f"Runs fixed questions ({sum(e is not None for _, e in TEST_QUESTIONS)} answerable from the "
        f"database, {sum(e is None for _, e in TEST_QUESTIONS)} deliberately not covered) through all three approaches."
    )
    n_q = st.slider("Number of questions", 4, len(TEST_QUESTIONS), len(TEST_QUESTIONS))
    pace = st.slider("Pause between questions (seconds), to respect free-tier rate limits", 0, 20, 5)
    judge = st.checkbox("Also fact-check each answer with an LLM (doubles the number of API calls)", value=False)
    # keep a mix: always include the not-covered questions when shortening the list
    covered = [t for t in TEST_QUESTIONS if t[1] is not None]
    uncovered = [t for t in TEST_QUESTIONS if t[1] is None]
    n_unc = min(len(uncovered), max(1, round(n_q * len(uncovered) / len(TEST_QUESTIONS))))
    chosen = covered[: n_q - n_unc] + uncovered[:n_unc]
    if st.button("Run benchmark", type="primary"):
        rows = []
        bar = st.progress(0.0)
        for i, (q, expected) in enumerate(chosen):
            res = run_all(q)
            for name, r in res.items():
                if judge:
                    unsupported = check_grounding(client, df, q, r)["unsupported"]
                else:
                    unsupported = check_unsupported(df, expected is not None, r)
                correct = (r["policy"] == expected) if expected else r["policy"] is None
                rows.append(
                    {
                        "approach": name,
                        "question": q,
                        "expected": expected or "(not covered)",
                        "returned": r["policy"] or "(declined)",
                        "correct": correct,
                        "answered_when_not_covered": expected is None and r["policy"] is not None,
                        "unsupported": unsupported,
                        "seconds": r["seconds"],
                        "tokens": r["total_tokens"],
                    }
                )
            bar.progress((i + 1) / len(chosen))
            time.sleep(pace)
        st.session_state["bench"] = pd.DataFrame(rows)
        st.session_state["bench_questions"] = chosen

    if "bench" in st.session_state:
        b = st.session_state["bench"]
        chosen_q = st.session_state["bench_questions"]
        n_out = sum(e is None for _, e in chosen_q)
        summary = (
            b.groupby("approach", sort=False)
            .agg(
                correct=("correct", "mean"),
                seconds=("seconds", "mean"),
                tokens=("tokens", "mean"),
                unsupported=("unsupported", "mean"),
                bad_answers=("answered_when_not_covered", "sum"),
            )
            .reset_index()
        )
        summary = pd.DataFrame(
            {
                "Approach": summary["approach"],
                "Correct policy": (summary["correct"] * 100).round(0).astype(int).astype(str) + "%",
                "Avg response time (s)": summary["seconds"].round(2),
                "Avg tokens / question": summary["tokens"].round(0).astype(int),
                "Unsupported answers": (summary["unsupported"] * 100).round(0).astype(int).astype(str) + "%",
                f"Answered the {n_out} uncovered questions": summary["bad_answers"].astype(int).astype(str) + f" / {n_out}",
            }
        )
        st.markdown("**Table 1. Summary per approach**")
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.markdown("**Table 2. Results per question**")
        detail = b.assign(result=b["correct"].map({True: "✓", False: "✗"}))
        pivot = detail.pivot(index="question", columns="approach", values="returned")[APPROACHES]
        pivot.insert(0, "expected", detail.drop_duplicates("question").set_index("question")["expected"])
        pivot = pivot.loc[[q for q, _ in chosen_q]].reset_index()
        st.dataframe(pivot, hide_index=True, use_container_width=True)
        st.download_button("Download raw results (CSV)", b.to_csv(index=False), "benchmark_results.csv")

# ---------- tab 3: written comparison ----------
with tab_summary:
    st.write(
        "PLACEHOLDER PARAGRAPH 1: replace with your first paragraph (compare the three approaches using the "
        "benchmark numbers)."
    )
    st.write("PLACEHOLDER PARAGRAPH 2: replace with your second paragraph (preferred approach and why).")
