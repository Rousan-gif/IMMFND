from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import time
import pandas as pd

# Set up Chrome options (optional: headless mode)
options = Options()
# options.add_argument("--headless")  # Uncomment to run in headless mode

# Path to your ChromeDriver
chrome_driver_path = "C:/Users/rousa/Desktop/Dataset Collection/chromedriver-win64/chromedriver.exe"

# Set up Chrome service
service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

#------------------------- For www.altnews.in -----------------------------------#

# Configuration
BASE_URL = "https://www.altnews.in/"

# Go to the website
driver.get(BASE_URL)

# Scroll to the bottom until no new content loads
scroll_pause_time = 3
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause_time)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# After all content is loaded, parse the page
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Close the browser
driver.quit()

# Step 1: Extract article links
articles = soup.find_all('h4', class_="entry-title")
article_links = [urljoin(BASE_URL, h4.a['href']) for h4 in articles if h4.a]

# Step 2: Visit each article and extract data
all_articles_data = []

for url in article_links:
    try:
        web_data = requests.get(url)
        data = BeautifulSoup(web_data.text, 'html.parser')

        # Extract title
        title_tag = data.find("h1", class_="headline")
        title = title_tag.text.strip() if title_tag else "No Title"

        # Extract first paragraph
        content_tag = data.find("p")
        content = content_tag.text.strip() if content_tag else ""

        # Extract image source (adjust class as needed)
        image_tag = data.find("img", class_=["alignnone", "size-full", "wp-image"])
        image_src = image_tag['src'] if image_tag else None

        # Save the data
        all_articles_data.append({
            "url": url,
            "title": title,
            "content": content,
            "image_src": image_src
        })

    except Exception as e:
        print(f"Error processing {url}: {e}")

# Step 3: Save to DataFrame
df = pd.DataFrame(all_articles_data)
print(df)

#------------------------- For www.boomlive.in -----------------------------------#

# Base URL
base_url = "https://www.boomlive.in/fact-check/"

# Headers for polite scraping
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# List to hold all article links
article_links = []

# Loop through pages 1 to 333
for i in range(1, 333):  # 334 because range is exclusive at the end
    url = f"{base_url}{i}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('h4', class_="font-alt normal")

        for h4 in articles:
            if h4.a and h4.a.get('href'):
                full_link = urljoin(base_url, h4.a['href'])
                article_links.append(full_link)

    except requests.RequestException as e:
        print(f"Failed to fetch page {url}: {e}")

# Optional: print or process the collected links
print(f"Collected {len(article_links)} article links.")

#------------------------- For digiteye.in -----------------------------------#
# Base URL
base_url = "https://digiteye.in/category/health/page/"
# Headers for polite scraping
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# List to hold all article links
article_links = []

# Loop through pages 1 to 333
for i in range(1, 47):  # 334 because range is exclusive at the end
    url = f"{base_url}{i}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('h2', class_="post-title")

        for h4 in articles:
            if h4.a and h4.a.get('href'):
                full_link = urljoin(base_url, h4.a['href'])
                article_links.append(full_link)

    except requests.RequestException as e:
        print(f"Failed to fetch page {url}: {e}")

# Optional: print or process the collected links
print(f"Collected {len(article_links)} article links.")

# Step 2: Scrape each article's content
all_articles_data = []

for url in article_links:
    try:
        web_data = requests.get(url, headers=headers, timeout=10)
        web_data.raise_for_status()

        data = BeautifulSoup(web_data.text, 'html.parser')

        title_tag = data.find("span",{"itemprop":"name"})
        title = title_tag.get_text(strip=True) if title_tag else None

        content_tag = data.find("div", class_="entry")
        content = content_tag.get_text(strip=True) if content_tag else None

        image_tag = data.find("img",{"decoding":"async"})
        image_src = image_tag['src'] if image_tag and image_tag.has_attr('src') else None

        all_articles_data.append({
            "url": url,
            "title": title,
            "content": content,
            "image_src": image_src
        })

    except requests.RequestException as e:
        print(f"Failed to fetch article {url}: {e}")
    except Exception as e:
        print(f"Error parsing article {url}: {e}")


#------------------------- For newsmeter.in -----------------------------------#

# Step 1: Collect all article links
base_url = "https://newsmeter.in/fact-check/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

article_links = []

for i in range(1, 2):
    url = f"{base_url}{i}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('h5', class_="mt-0")

        for h4 in articles:
            if h4.a and h4.a.get('href'):
                full_link = urljoin(base_url, h4.a['href'])
                article_links.append(full_link)

    except requests.RequestException as e:
        print(f"Failed to fetch page {url}: {e}")

# Step 2: Scrape each article's content
all_articles_data = []

for url in article_links:
    try:
        web_data = requests.get(url, headers=headers)
        web_data.raise_for_status()

        data = BeautifulSoup(web_data.text, 'html.parser')

        title_tag = data.find("h1", class_="text-left")
        title = title_tag.get_text(strip=True) if title_tag else None

        content_tag = data.find("div", class_="pasted-from-word-wrapper")
        content = content_tag.get_text(strip=True) if content_tag else None

        image_tag = data.find("img", {"data-class"=="h-custom-image"})
        image_src = image_tag['src'] if image_tag and image_tag.has_attr('src') else None

        all_articles_data.append({
            "url": url,
            "title": title,
            "content": content,
            "image_src": image_src
        })

    except requests.RequestException as e:
        print(f"Failed to fetch article {url}: {e}")
    except Exception as e:
        print(f"Error parsing article {url}: {e}")

# Step 3: Save to Excel
df = pd.DataFrame(all_articles_data)
df.to_excel("newsmeter_articles.xlsx", index=False)

print("Saved all articles to newsmeter_articles.xlsx")




#------------------------- For thelogicalindian.com -----------------------------------#

# Base URL
base_url = "https://thelogicalindian.com/category/conscious-consumer/page/"
# Headers for polite scraping
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# List to hold all article links
article_links = []

# Loop through pages 1 to 333
for i in range(1, 15):  # 334 because range is exclusive at the end
    url = f"{base_url}{i}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('h3', class_="elementor-post__title")

        for h3 in articles:
            if h3.a and h3.a.get('href'):
                full_link = urljoin(base_url, h3.a['href'])
                article_links.append(full_link)

    except requests.RequestException as e:
        print(f"Failed to fetch page {url}: {e}")

# Optional: print or process the collected links
print(f"Collected {len(article_links)} article links.")
web_data = requests.get(url, headers=headers, timeout=10)
web_data.raise_for_status()
data = BeautifulSoup(web_data.text, 'html.parser')
content_tag = data.find("div", {"data-widget_type":"theme-post-content.default"})
content = content_tag.get_text(strip=True) if content_tag else None
                        
# Step 2: Scrape each article's content
all_articles_data = []

for url in article_links:
    try:
        web_data = requests.get(url, headers=headers, timeout=10)
        web_data.raise_for_status()

        data = BeautifulSoup(web_data.text, 'html.parser')

        title_tag = data.find("h1", class_ ="elementor-heading-title elementor-size-default")
        title = title_tag.get_text(strip=True) if title_tag else None

        content_tag = data.find("div", {"data-widget_type":"theme-post-content.default"})
        content = content_tag.get_text(strip=True) if content_tag else None

        image_tag = data.find("img", class_ = "attachment-full")
        image_src = image_tag['src'] if image_tag and image_tag.has_attr('src') else None

        all_articles_data.append({
            "url": url,
            "title": title,
            "content": content,
            "image_src": image_src
        })

    except requests.RequestException as e:
        print(f"Failed to fetch article {url}: {e}")
    except Exception as e:
        print(f"Error parsing article {url}: {e}")

