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
    ANALYSIS_CONFIGS,
    KEYWORDS,
    HANDLE,
    PASSWORD,
    DATE_START,
    DATE_END,
    OUTPUT_DIR
)

# --- LOGIN ---
print("Connecting to Bluesky...")
client = Client()
client.login(HANDLE, PASSWORD)
print("Login successful!")

print("\n" + "="*80)
print("COLLECTING DATA FOR ALL KEYWORDS (ONCE)")
print(f"Total keywords: {len(KEYWORDS)}")
print("="*80)

# --- DATA COLLECTION (ONCE FOR ALL KEYWORDS) ---
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

# --- PROCESS AND SAVE DATA FOR EACH ANALYSIS ---
print("\n" + "="*80)
print(f"DATA COLLECTION COMPLETE: {len(records)} total posts collected")
print("Now filtering and saving for each analysis...")
print("="*80)

if records:
    # Convert to DataFrame for efficient filtering
    df_all = pd.DataFrame(records)
    
    for config_idx, analysis_config in enumerate(ANALYSIS_CONFIGS, 1):
        config_name = analysis_config["name"]
        config_keywords = analysis_config["keywords"]
        description = analysis_config["description"]
        
        # Create subfolder for this analysis
        analysis_output_dir = f"{OUTPUT_DIR}/{config_name}"
        os.makedirs(analysis_output_dir, exist_ok=True)
        OUTPUT_FILE = f"{analysis_output_dir}/bluesky_posts_complex.csv"
        
        print("\n" + "="*80)
        print(f"ANALYSIS {config_idx}/3: {description}")
        print(f"Keywords: {len(config_keywords)}")
        print(f"Output directory: {analysis_output_dir}")
        print("="*80)
        
        # Filter posts that contain ANY of the keywords for this analysis
        mask = df_all['keyword'].isin(config_keywords)
        df = df_all[mask].copy()
        
        print(f"Filtered {len(df)} posts (from {len(df_all)} total) for this analysis")
        
        if len(df) > 0:
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved {len(df)} posts in '{OUTPUT_FILE}'")

            # --- Sentiment chart ---
            sentiment_counts = df["sentiment"].value_counts()
            fig = plt.figure()
            sentiment_counts.plot(
                kind="bar",
                title=f"Sentiment distribution - {description}"
            )
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"{analysis_output_dir}/sentiment_distribution.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved sentiment chart in '{analysis_output_dir}/sentiment_distribution.png'")

            # --- Co-occurrence analysis ---
            print(f"\nCalculating co-occurrences for {len(config_keywords)} keywords...")
            new_df = {"w1": [], "w2": [], "n": []}
            for kws in list(combinations(config_keywords, 2)):
                all_ks = None
                for kw in kws:
                    if all_ks is None:
                        all_ks = df.text.str.contains(kw, case=False, na=False)
                    else:
                        all_ks = all_ks & df.text.str.contains(kw, case=False, na=False)
                new_df["w1"].append(kws[0])
                new_df["w2"].append(kws[1])
                new_df["n"].append(all_ks.sum())
            
            pd.DataFrame(new_df).to_excel(f"{analysis_output_dir}/grafo.xlsx")
            print(f"Saved co-occurrence matrix in '{analysis_output_dir}/grafo.xlsx'")
        else:
            print(f"WARNING: No posts found for {description}.")

else:
    print("\nWARNING: No posts found with the specified criteria.")

print("\n" + "="*80)
print("ALL ANALYSES COMPLETE!")
print("="*80)
