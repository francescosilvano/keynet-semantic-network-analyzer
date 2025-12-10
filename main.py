from atproto import Client, models
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import time
import json

# --- CONFIGURATION ---
HANDLE = "provauni.bsky.social"   # ⬅️ Insert your Bluesky handle here
PASSWORD = "s.sabatino"               # ⬅️ Insert your password here
KEYWORDS = [
    "complexity", "unpredictability", "dynamic", "sensitivity",
    "fear", "nervousness", "aggressiveness", "apprehension",
    "stress", "isolation", "boredom"]
LOCATION_KEYWORDS = ["florence", "firenze", "tuscany", "toscana", "italy", "italia"]
DATE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
DATE_END = datetime(2025, 11, 25, tzinfo=timezone.utc)
OUTPUT_FILE = "bluesky_posts_complex.csv"

# --- LOGIN ---
print("🔐 Connecting to Bluesky...")
client = Client()
client.login(HANDLE, PASSWORD)
print("✅ Login successful!")

# --- DATA COLLECTION ---
records = []


for keyword in KEYWORDS:
    cursor = '0'
    while True:
        if cursor is None or int(cursor) > 1000:
            break
        print(f"\n🔍 Searching posts with keyword: {keyword}")
        try:
            params = models.AppBskyFeedSearchPosts.Params(q=keyword, limit=100, cursor=cursor)
            feed = client.app.bsky.feed.search_posts(params)
            cursor = json.loads(feed.json())["cursor"]
            posts = feed.posts or []
            print(f"   → found {len(posts)} results | reached {cursor}")

            for post in posts:
                # --- Text and author ---
                text = getattr(post.record, "text", "")
                created_at = getattr(post.record, "created_at", "")
                author = post.author
                handle = getattr(author, "handle", "")
                display_name = getattr(author, "display_name", "") or ""
                description = getattr(author, "description", "") or ""

                # --- Location filter (bio + display_name) with None handling ---
                author_text = (description or "") + (display_name or "")
                location_match = True
                # any(
                #     loc in author_text.lower()
                #     for loc in LOCATION_KEYWORDS
                # )

                # --- Date filter ---
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(timezone.utc)
                    except:
                        continue
                    if not (DATE_START <= dt <= DATE_END):
                        continue
                else:
                    continue

                # --- Sentiment analysis ---
                sentiment_score = TextBlob(text).sentiment.polarity
                sentiment_label = "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"

                # --- Determine if repost or original ---
                post_type = "original"
                final_text = text

                if hasattr(post, "post") and hasattr(post.post, "record"):
                    record_obj = post.post.record
                    if hasattr(record_obj, "embed") and hasattr(record_obj.embed, "record") and hasattr(record_obj.embed.record, "value"):
                        embed_value = record_obj.embed.record.value
                        if hasattr(embed_value, "text"):
                            post_type = "repost"
                            final_text = embed_value.text
                elif hasattr(post, "repost") and hasattr(post.repost, "record") and hasattr(post.repost.record, "text"):
                    post_type = "repost"
                    final_text = post.repost.record.text

                # --- Save only if location matches ---
                if location_match:
                    records.append({
                        "keyword": keyword,
                        "type": post_type,
                        "author": display_name,
                        "handle": handle,
                        "bio": description,
                        "text": final_text,
                        "date": dt.strftime("%Y-%m-%d") if dt else "",
                        "sentiment": sentiment_label,
                        "score": sentiment_score
                    })

            time.sleep(2)  # to avoid saturating requests

        except Exception as e:
            print(f"⚠️ Error searching for '{keyword}': {e}")
            continue

# --- CSV SAVE ---
if records:
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved {len(df)} posts in '{OUTPUT_FILE}'")

    # --- Sentiment chart ---
    sentiment_counts = df["sentiment"].value_counts()
    sentiment_counts.plot(kind="bar", title="Sentiment distribution (filtered posts)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

else:
    print("\n⚠️ No posts found with the specified criteria.")

print(KEYWORDS)
from itertools import combinations
# print(df["text"][2000])

new_df = {"w1":[],"w2":[],"n":[]}
for kws in list(combinations(KEYWORDS, 2)):
    all_ks = None
    for kw in kws:
        if all_ks is None:
            all_ks = df.text.str.contains(kw)
        else:
            all_ks = all_ks & df.text.str.contains(kw)
    new_df["w1"].append(kws[0])
    new_df["w2"].append(kws[1])
    new_df["n"].append(all_ks.sum())
    print(kws, all_ks.sum())
pd.DataFrame(new_df).to_excel("grafo.xlsx")