"""Slack bot for the Company Policy Assistant (LLM + vector index, same approach as the website).

Run it on your computer:
    pip install -r requirements.txt
    set GEMINI_API_KEY, SLACK_BOT_TOKEN (xoxb-...) and SLACK_APP_TOKEN (xapp-...) as environment variables
    python slack_bot.py
Then, in your Slack test channel, type:  @YourBotName how many vacation days do I get?
"""
import os
import re
import sys

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from policy_core import EMBED_MODEL, VectorIndex, get_client, load_policies

missing = [k for k in ("GEMINI_API_KEY", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN") if not os.environ.get(k)]
if missing:
    sys.exit(f"Missing environment variable(s): {', '.join(missing)}")

df = load_policies()
client = get_client(os.environ["GEMINI_API_KEY"])
safe = re.sub(r"[^A-Za-z0-9_.-]", "_", EMBED_MODEL)
index = VectorIndex(client, df, cache_path=f"embeddings_cache_{safe}.npy")
app = App(token=os.environ["SLACK_BOT_TOKEN"])


@app.event("app_mention")
def answer_policy_question(event, say):
    question = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
    thread = event.get("thread_ts") or event["ts"]  # reply inside a thread to keep the channel tidy
    if not question:
        say(text="Ask me a policy question, for example: how many vacation days do I get?", thread_ts=thread)
        return
    try:
        r = index.ask(question)
    except Exception as e:  # noqa: BLE001
        say(text=f"Sorry, something went wrong while answering: {e}", thread_ts=thread)
        return
    policy = r["policy"] or "none found in the policy database"
    say(
        text=(
            f"*Answer:* {r['answer']}\n"
            f"*Relevant policy:* {policy}\n"
            f"_{r['seconds']:.1f} s · {r['total_tokens']} tokens_"
        ),
        thread_ts=thread,
    )


@app.event("message")
def ignore_other_messages():
    pass  # only @mentions are answered; this just stops Slack warnings about unhandled events


if __name__ == "__main__":
    print("Policy bot is running. Press Ctrl+C to stop.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
