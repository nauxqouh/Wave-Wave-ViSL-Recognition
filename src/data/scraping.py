import os
import json
import time
import requests
from tqdm import tqdm
import re
from html import unescape
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://qipedc.moet.gov.vn"
URL = "https://qipedc.moet.gov.vn/dictionary"
VIDEO_DIR = 'dataset/videos'
METADATA_PATH = 'dataset/metadata.jsonl'
CHUNK_SIZE = 1024*32
NUM_PAGES = 219
HEADLESS = True
WAIT_TIME = 5
MAX_WORKERS = 2

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("logs/scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("qipedc_scraper")

def init_driver(headless: bool = HEADLESS) -> webdriver.Chrome:
    """Initialize Chrome driver."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument("--ignore-ssl-errors=yes")
    options.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def download_video(video_data, video_dir='dataset/videos', chunk_size=1024*32):
    """Download one video."""
    
    filename = f"{video_data['id']}.mp4"
    os.makedirs(video_dir, exist_ok=True)
    output_path = os.path.join(video_dir, filename)
    
    # Skip if existed
    if os.path.exists(output_path):
        log.debug(f"Skipped existing video {filename}")
        return
    
    # download video
    try:
        response = requests.get(video_data['url'], stream=True, verify=False, timeout=10)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=f"Progess {filename}"
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
            log.info(f"Downloaded video successfully: {filename}")
    except Exception as e:
        log.error(f"Failed to download {filename}: {e}")
        return f"Video {filename} downloaded failed: {e}"

def scrape_one_page(driver, video_dir='dataset/videos', chunk_size=1024*32):
    """Scrape data included getting metadata and downloading video in one page."""
    
    data = []
    try:
        videos = driver.find_elements(By.CSS_SELECTOR, "#product a")
        start_time = time.time()
        
        tasks = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for i, vid in enumerate(videos):
                try:
                    video_data = {}
                    label = vid.find_element(By.TAG_NAME, "p").text.strip()
                    thumbs_url = vid.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
                    video_id = thumbs_url.replace("https://qipedc.moet.gov.vn/thumbs/", "").replace(".png", "")
                    video_url = f"{BASE_URL}/videos/{video_id}.mp4"
                    
                    # driver.execute_script("modalData(arguments[0], arguments[1], arguments[2], arguments[3])",
                    #                       id, label, l_definition, flag)
                    # WebDriverWait(driver, WAIT_TIME).until(
                    #     expected_conditions.presence_of_element_located((By.CSS_SELECTOR, "#s_expert"))
                    # )
                    # iframe = driver.find_element(By.CSS_SELECTOR, "#s_expert")
                    # video_data['url'] = iframe.get_attribute("src").replace('?autoplay=true','')
                    
                    local_url = os.path.join(video_dir, video_id)
                    # add metadata
                    data.append({
                        'id': video_id,
                        'word': label,
                        'video_url': local_url
                    })
                    
                    # create task video download
                    video_data['id'] = video_id
                    video_data['url'] = video_url
                    tasks.append(executor.submit(download_video, video_data, video_dir, chunk_size))
                    
                    ## quit modal
                    # driver.execute_script("$('#exampleModal').modal('hide');")
                    # time.sleep(0.5)
                    
                except Exception as e:
                    log.warning(f"Error at video {video_data['id']}: {e}")
            
            for future in as_completed(tasks):
                result = future.result()
                if result:
                    log.debug(result)

        log.info(f"Page scrapped in {time.time() - start_time:.2f}s")
            
    except Exception as e:
        log.error(f"Page scrape failed: {e}")
    
    return data
        
def turn_page(driver, page_num):
    """Go to the page_num page."""
    try:
        WebDriverWait(driver, WAIT_TIME).until(
            expected_conditions.presence_of_element_located((By.CSS_SELECTOR, "#pagination-wrapper"))
        )
        found = False
        attempt = 0

        while not found and attempt < NUM_PAGES:
            attempt += 1
            buttons = driver.find_elements(By.CSS_SELECTOR, "#pagination-wrapper button.page")
            for btn in buttons:
                if btn.get_attribute("value") == str(page_num):
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(1.5)

                    # Page is showing
                    current_page = driver.find_element(By.CSS_SELECTOR, "#pagination-wrapper .btn-info")
                    log.info(f"Page {current_page.get_attribute('value')} is being opened.")
                    found = True
                    break
                
            if found:
                break
            
            next_btn = driver.find_element(By.CSS_SELECTOR, "#pagination-wrapper button.next")
            if next_btn.is_enabled():
                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(1)
            else:
                log.error(f"Cannot found page {page_num}, no next button available.")
                break  
                
        if not found:
            log.warning(f"Failed to reach page {page_num} after {attempt} attempts.")

    except Exception as e:
        log.error(f"Cannot go to page {page_num}: {e}")

def safe_turn_page(driver, page_num):
    """Recover driver when session lost."""
    try:
        turn_page(driver, page_num)
    except Exception as e:
        if "invalid session id" in str(e).lower():
            log.warning("Driver session lost. Re-initialize.")
            driver.quit()
            driver = init_driver()
            driver.get(URL)
            turn_page(driver, page_num)
        else:
            raise
    return driver

def save_jsonl(data, output_path):
    """Append JSON lines to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    log.info(f"Saved {len(data)} records to {output_path}.")

def scrape_data(metadata_path='dataset/metadata.jsonl', video_dir='dataset/videos', num_pages=219):
    """Scrape all pages."""
    
    driver = init_driver()
    start_time = time.time()
    page_num = 1
    
    try:
        driver.get(URL)
        driver.implicitly_wait(5)
        log.info(f"Connected to {driver.title} ({driver.current_url})")
    except Exception as e:
        log.error(f"Failed to connect: {e}")
        driver.quit()
        return
    
    while page_num <= num_pages:
        try:
            log.info(f"Scraping page {page_num}/{num_pages} ...")
            
            # Check driver session
            try:
                driver.title
            except Exception as e:
                driver = safe_turn_page(driver, num_pages)
                    
            page_data = scrape_one_page(driver, video_dir, chunk_size=1024*32)
            save_jsonl(page_data, metadata_path)
            
            if page_num < num_pages:
                driver = safe_turn_page(driver, page_num + 1)
            page_num += 1 
        except Exception as e:
            log.error(f"Error at page {page_num}: {e}. Retrying after 5s...")
            time.sleep(5)
            # try:
            #     driver.title
            # except:
            #     log.warning("Driver lost. Re-initialize.")
            #     driver.quit()
            #     driver = init_driver()
            #     driver.get(URL)
            #     turn_page(driver, page_num)
            driver = safe_turn_page(driver, page_num)
        
    driver.quit()   
    log.info(f"Done scraping {num_pages} pages in {time.time() - start_time:.4f}s")
    
if __name__ == "__main__":
    scrape_data(metadata_path=METADATA_PATH, video_dir=VIDEO_DIR, num_pages=NUM_PAGES)