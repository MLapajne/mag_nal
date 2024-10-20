from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import sys
import signal
from urllib.parse import urlparse, urlunsplit, urlsplit
import random
import time
from pymongo import MongoClient
import json
import requests
import os
from PIL import Image
import re

client = MongoClient('mongodb://localhost:27017/')  # Update with your connection string if different
db = client['ClimbingReutes']  # Database name
collection = db['Images']
images_dir = os.path.dirname(os.path.abspath(__file__)) + "/downloaded_images/"

def download_image(url, download_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(download_path, 'wb') as file:
            file.write(response.content)
        return download_path
    else:
        print(f"Failed to download image from {url}")
        return None
    
def check_and_convert_image_type(image_path):
    try:
        with Image.open(image_path) as img:
            # Construct the new path with .png extension
            #new_image_path = os.path.join(image_path,".png") 
            new_image_path = f"{os.path.splitext(image_path)[0]}.png"
            # Save the image as PNG
            img.save(new_image_path, 'PNG')
            new_image_path = f"{os.path.splitext(image_path)[0]}.png"
            if not os.path.exists(new_image_path):
                print(f"Failed to convert image to PNG: {new_image_path}")
                return None
            return new_image_path
            
    except Exception as e:
        print(f"Failed to identify or convert image type for {image_path}: {e}")
        return None, None
    
def replace_image_url(url):
    return re.sub(r'https://image\.thecrag\.com/[^/]+/', 'https://image.thecrag.com/original-image/', url)

def traverse_and_download_images(document, base_path):
    if isinstance(document, dict):
        for key, value in document.items():
            if key == "Image" and isinstance(value, list):
                for url in value:
                    #url = replace_image_url(url)
                    image_name_with_extension = os.path.basename(url)
                    image_name, _ = os.path.splitext(image_name_with_extension)
                    sanitized_base_path = base_path.replace('/', '_')
                    download_path = os.path.join(images_dir, f"{sanitized_base_path}_{image_name}")               
                    downloaded_image_path = download_image(url, download_path)
                    if downloaded_image_path:
                        new_image_path = check_and_convert_image_type(downloaded_image_path)
                        if not new_image_path:
                            print(f"Failed to identify or convert image type for {downloaded_image_path}")
                    else:
                        print(f"Failed to download image from {url}")
            else:
                new_base_path = os.path.join(base_path, key)
                traverse_and_download_images(value, new_base_path)
    elif isinstance(document, list):
        print("List found. Skipping...")

def update_document_with_downloaded_images():
    document = collection.find_one({"_id": "hierarchy_root"})
    if document:
        images_dir = os.path.dirname(os.path.abspath(__file__)) + "/downloaded_images/"
        os.makedirs(images_dir, exist_ok=True)
        traverse_and_download_images(document, "")
    else:
        print("Hierarchy document not found.")


def add_path(nested_dict, path, images_src):

    current_level = nested_dict
    for part in path:
        if part not in current_level:
            current_level[part] = {}
        current_level = current_level[part]
    
    # Ensure the "Image" field is a list
    if "Image" not in current_level:
        current_level["Image"] = []
    
    # Append the new image URL to the "Image" array
    for src in images_src:
        if src not in current_level["Image"]:
            current_level["Image"].append(src)

    # Update the document in MongoDB
    collection.update_one(
        {"_id": "hierarchy_root"},
        {"$set": {"World": nested_dict["World"]}}
    )
    
    print(f"Path {path} added successfully.")
 

def insert_initial_document():
    nested_document = {
        "World": {},
        "PageNum" : 1,
        "imageNumPage" : {}
    }

    # Assign a unique _id if desired
    nested_document['_id'] = 'hierarchy_root'

    # Insert the document
    try:
        collection.insert_one(nested_document)
        print("Initial nested document inserted successfully.")
    except Exception as e:
        print(f"Error inserting initial document: {e}")

def get_page_num():
    document = collection.find_one({"_id": "hierarchy_root"})
    if document:
        return document["PageNum"]
    else:
        return 1

def update_page_num(page_num, image_num_per_page):
    collection.update_one(
        {"_id": "hierarchy_root"},
        {"$set": {"PageNum": page_num, "ImageNumPerPage": image_num_per_page}}
    )

def get_image_num_per_page():
    document = collection.find_one({"_id": "hierarchy_root"}, {"ImageNumPerPage": 1})
    if document and "ImageNumPerPage" in document:
        return document["ImageNumPerPage"]
    else:
        print("ImageNumPerPage not found in the document")
        return 0

def add_image_num_page(page_num, image_num):
    document = collection.find_one({"_id": "hierarchy_root"})
    if document:
        # Ensure page_num is a string
        page_num_str = str(page_num)
        
        # Initialize imageNumPage if it doesn't exist
        if "imageNumPage" not in document:
            document["imageNumPage"] = {}

        document["imageNumPage"][page_num_str] = image_num
        collection.update_one(
            {"_id": "hierarchy_root"},
            {"$set": {"imageNumPage": document["imageNumPage"]}}
        )
    else:
        print("Hierarchy document not found.")



def update_document_with_path(path, image_srcs):
    """
    Retrieve the existing document, add the new path with image sources, and update it in MongoDB.
    
    :param path: The new path to be added.
    :param image_srcs: The image sources to be added.
    """
    # Retrieve the existing document
    existing_document = collection.find_one({"_id": "hierarchy_root"})
    
    if existing_document:
        # Remove the _id to prevent duplication
        del existing_document['_id']
    else:
        print("Hierarchy document not found. Inserting initial document first.")
        insert_initial_document()
        existing_document = collection.find_one({"_id": "hierarchy_root"})
        if existing_document:
            del existing_document['_id']
        else:
            print("Failed to insert and retrieve the initial document.")
            return

    # Add the new path
    add_path(existing_document, path, image_srcs)
    
    

# Function to display the nested document
def display_document():
    document = collection.find_one({"_id": "hierarchy_root"})
    if document:
        # Remove the _id for display purposes
        del document['_id']
        print(json.dumps(document, indent=4, ensure_ascii=False))
    else:
        print("Hierarchy document not found.")

def signal_handler(sig, frame):         
    print('You pressed Ctrl+C! Exiting gracefully...')
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

class Crawler:
    def __init__(self, url, proxy):
        self.url = url
        self.page_content = None
        self.browser = None
        self.proxy = proxy

    def check_service_unavailable(self, page, browser):
        if page.locator('text="Service Unavailable"').count() > 0:
            print("Service Unavailable page detected. Closing browser.")
            browser.close()
            return True
        elif page.locator('text="Potrdite, da ste človek, tako da opravite spodnje dejanje."').count() > 0:
            print("Human verification page detected. Closing browser.")
            browser.close()
            return True
        elif page.locator('text="Preverjanje, ali ste človek. To lahko traja nekaj sekund."').count() > 0:
            print("Human verification page detected. Closing browser.")
            browser.close()
            return True
        elif page.locator('text="You have exceeded your allowed anonymous requests. Please login to continue or come back later."').count() > 0:
            print("Human verification page detected. Closing browser.")
            browser.close()
            return True
        elif page.locator('text="Internal Server Error"').count() > 0:
            print("Internal Server Error")
            browser.close()
            return True
        return False

    def handle_sock_interaction(self, page, pageNumber):
        #selector = 'a.heading__t'
        try:
            photo_list_handle = page.locator('.photo-list')
            photo_list_handle.wait_for(state='visible', timeout=30000)
            print("Photo-list is visible")
        except PlaywrightTimeoutError:
            print("Timeout waiting for photo-list to be visible")
            return False

        
        link_photos = page.locator('.link-photo')

        if link_photos.count() == 0:
            print("No link-photo elements found")
            return None

        
      
        data_list = []
        # Iterate over each 'link-photo' div
        for i in range(get_image_num_per_page(), link_photos.count()):


            # Locate the 'a' element inside the current 'link-photo' div
            image = link_photos.nth(i).locator('a.photo')
            title_elements = link_photos.nth(i).locator('p.title a')

            if title_elements.count() > 0:
            
                title_element = title_elements.nth(1) if title_elements.count() > 1 else title_elements
    
                title = title_element.get_attribute('title')

                if title is not None:
                    title_parts = [part.replace('\xa0', ' ').replace(' ', '').strip() for part in title.split('›')]
                  
            else:
                print(f"No <a> element found for image {i}")
                # Optionally, you can add a default title or handle it differently
                title_parts = ["Unknown"]
       

            image.scroll_into_view_if_needed()
            element_href = image.get_attribute('href')
            #image.click(position={'x': 10, 'y': 1})

            full_url = f"https://www.thecrag.com{element_href}"

            data_dict = {
                'title_parts': title_parts,
                'full_url': full_url
            }
            data_list.append(data_dict)

        for i, data in enumerate(data_list, start=get_image_num_per_page()):
            print(f"Index: {i}, Data: {data}")
            title_parts = data['title_parts']
            full_url = data['full_url']



            page.goto(full_url)
            page.wait_for_load_state('load')



            if self.check_service_unavailable(page, self.browser):
                return None
            
            view_full_size_element = page.locator('text="View full size"')

            if view_full_size_element is not None:
                src = view_full_size_element.get_attribute('href')
                if src:
                    update_document_with_path(title_parts, [src])
                else:
                    print("View full size element does not have href attribute")
            else:
                print("View full size element not found")
                return None        

            print(i)
            update_page_num(pageNumber, i + 1)

        

        # Print the list of image src attributes
    
       

        return True

        
       
   
    def crawl(self):
        with sync_playwright() as p:
            while True:
                #browser = p.chromium.launch(headless=False)
                self.browser = p.chromium.launch(
                #proxy={'server': f'{self.proxy}'},
                #proxy={'server': "34.77.56.122:8080"},
                #proxy={'server': "5.75.150.14:3128"},
                proxy={"server": "socks5://127.0.0.1:9050"},                   
                #headless=False
                ) 
                context = self.browser.new_context(user_agent="niki")
                context.set_default_timeout(10000000)
                page = context.new_page()
                
                #get the page number from db
                
                while True:
                    pageNumber = get_page_num()
                    print(f"Processing page {pageNumber}...")
                    response = page.goto(self.url + "?page=" + str(pageNumber), wait_until='networkidle')
                    if response.ok:
                        success = self.handle_sock_interaction(page, pageNumber)
                        if success:
                            update_page_num(pageNumber + 1, 0)
                        else:
                            print(f"Retrying page {pageNumber}...")
                            break  # Exit the inner loop to restart the browser
                    else:               
                        print(f"Failed to load page {pageNumber}. Retrying...")
                        break
                    #time.sleep(20) 
                self.browser.close()
                
    

def canonicalize_url(url):
    if url.endswith('/'):
        url = url[:-1]
    # Use urlsplit to extract the netloc part
    netloc = urlsplit(url).netloc

    return netloc


#proxy = FreeProxy(rand=True, https=True).get()
proxy = 0

#print(canonicalize_url(proxy))


if __name__ == "__main__":
    crawler = Crawler("https://www.thecrag.com/en/climbing/world/photos", proxy)
    crawler.crawl()
    #update_document_with_downloaded_images()
