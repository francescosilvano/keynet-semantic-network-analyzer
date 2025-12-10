"""
Main module for collecting and analyzing Bluesky posts
"""

import json
import os
import time
from datetime import datetime, timezone
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from atproto import Client, models
from textblob import TextBlob

# --- IMPORT SHARED CONFIGURATION ---
from config import (
    KEYWORDS,
    HANDLE,
    PASSWORD,
    DATE_START,
    DATE_END,
    OUTPUT_DIR
)

OUTPUT_FILE = f"{OUTPUT_DIR}/bluesky_posts_complex.csv"

# --- LOGIN ---
print("Connecting to Bluesky...")
client = Client()
client.login(HANDLE, PASSWORD)
print("Login successful!")

# --- DATA COLLECTION ---
records = []


for keyword in KEYWORDS:
    cursor = '0'
    while True:
        if cursor is None or int(cursor) > 1000:
            break
        print(f"\nSearching posts with keyword: {keyword}")
        try:
            params = models.AppBskyFeedSearchPosts.Params(
                q=keyword, limit=100, cursor=cursor
            )
            feed = client.app.bsky.feed.search_posts(params)
            cursor = json.loads(feed.json())["cursor"]
            posts = feed.posts or []
            print(f"   Found {len(posts)} results | reached cursor {cursor}")

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
                        dt = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        ).astimezone(timezone.utc)
                    except (ValueError, AttributeError):
                        continue
                    if not DATE_START <= dt <= DATE_END:
                        continue
                else:
                    continue

                # --- Sentiment analysis ---
                sentiment_score = TextBlob(text).sentiment.polarity
                if sentiment_score > 0.1:
                    sentiment_label = "positive"
                elif sentiment_score < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"

                # --- Determine if repost or original ---
                post_type = "original"
                final_text = text

                if hasattr(post, "post") and hasattr(post.post, "record"):
                    record_obj = post.post.record
                    if (hasattr(record_obj, "embed") and
                        hasattr(record_obj.embed, "record") and
                            hasattr(record_obj.embed.record, "value")):
                        embed_value = record_obj.embed.record.value
                        if hasattr(embed_value, "text"):
                            post_type = "repost"
                            final_text = embed_value.text
                elif (hasattr(post, "repost") and
                      hasattr(post.repost, "record") and
                      hasattr(post.repost.record, "text")):
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
            print(f"WARNING: Error searching for '{keyword}': {e}")
            continue

# --- CSV SAVE ---
if records:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} posts in '{OUTPUT_FILE}'")

    # --- Sentiment chart ---
    sentiment_counts = df["sentiment"].value_counts()
    sentiment_counts.plot(
        kind="bar",
        title="Sentiment distribution (filtered posts)"
    )
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

else:
    print("\nWARNING: No posts found with the specified criteria.")

print(KEYWORDS)

new_df = {"w1": [], "w2": [], "n": []}
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
pd.DataFrame(new_df).to_excel(f"{OUTPUT_DIR}/grafo.xlsx")
