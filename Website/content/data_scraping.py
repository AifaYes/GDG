import pandas as pd
import time
from playwright.sync_api import sync_playwright

# ✅ CSV Configuration
csv_file = "tiktok_trends.csv"
columns = ["Trend", "Engagement", "Caption", "URL"]

# ✅ Initialize CSV file
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def scrape_tiktok_trending():
    with sync_playwright() as p:
        print("✅ Opening TikTok Explore Page...")

        browser = p.chromium.launch(headless=False)  
        context = browser.new_context()
        page = context.new_page()

        page.goto("https://www.tiktok.com/explore", timeout=60000)
        print("✅ TikTok Explore Pacge Opened")

        for _ in range(25):  
            page.mouse.wheel(0, 1500)  
            time.sleep(2)  

        selectors = [
            "div[data-e2e='recommend-list-item-container']", 
            "div.tiktok-1soki6-DivContainerForCard", 
            "div.css-x6y88p-DivItemContainerV2"
        ]
        
        trending_videos = []
        video_urls = []  

        for selector in selectors:
            video_elements = page.query_selector_all(selector)
            if video_elements:
                print(f"✅ Found {len(video_elements)} videos using: {selector}")
                break
        else:

            print("❌ No trending videos found. TikTok may have changed its layout.")
            browser.close()
            return [], []

        for video in video_elements[:1000]:  
            try:
                title = video.inner_text()[:100]  
                print(f"✅ Extracted Title: {title}")  # Print the extracted title
                
                link = video.query_selector("a")  
                link = link.get_attribute("href") if link else "No link found"

                # Extract engagement (likes count)
                engagement = "Unknown"
                words = title.split()
                for word in words:
                    #print("\n Word :",word ," \n ")
                    #print("\n word [-2] :",word[-2] ," \n ")
                    if (word.endswith("K") and word[-2].isdigit()) or word.isdigit():
                        print("\n Likes :",word ," \n ")
                        engagement = word  
                        break

                video_urls.append(link)
                trending_videos.append({"title": title, "link": link, "engagement": engagement})
            except Exception as e:
                print(f"⚠️ Error extracting video info: {e}")
                continue

        browser.close()
        return trending_videos, video_urls
    
def extract_hashtags_from_video(video_url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  
        context = browser.new_context()
        page = context.new_page()

        page.goto(video_url, timeout=60000)
        print(f"✅ Visiting video page: {video_url}")

        time.sleep(5)

        hashtags = []
        hashtag_elements = page.query_selector_all("a[data-e2e='search-common-link']")
        for tag in hashtag_elements:
            hashtag = tag.inner_text().strip()
            if hashtag:
                hashtags.append(hashtag)
        
        # Extract the caption from the H1 element using the updated selector
        caption = ""
        caption_element = page.query_selector("h1.css-1fbzdvh-H1Container.ejg0rhn1")
        if caption_element:
            caption = caption_element.inner_text().strip()

        browser.close()
        return hashtags, caption

# ✅ Main Execution
if __name__ == "__main__":
    trending_videos, video_urls = scrape_tiktok_trending()

    print("\n🔍 Extracting hashtags for each video...\n")
    video_data = []
    
    for video in trending_videos:
        url = video["link"]
        if url and url.startswith("http"):
            hashtags, caption = extract_hashtags_from_video(url)
            trend = hashtags[0].lstrip("#") if hashtags else "Unknown_Trend"
            
            video_data.append({
                "Trend": trend,
                "Engagement": video["engagement"],
                "Caption": caption,
                "URL": url
            })
        else:
            print(f"⚠️ Skipping invalid URL: {url}")

    # ✅ Save data to CSV
    df_new = pd.DataFrame(video_data, columns=columns)
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(csv_file, index=False)

    print(f"\n📁 CSV file '{csv_file}' updated successfully! ✅")
