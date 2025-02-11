import mimetypes
from playwright.sync_api import (
    sync_playwright,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)
import sys
import signal
from urllib.parse import urlparse, urlunsplit, urlsplit
from pymongo import MongoClient
import json
import logging
import os
import re
from colorlog import ColoredFormatter

client = MongoClient(
    "mongodb://localhost:27017/"
)  # Update with your connection string if different
db = client["ClimbingReutes"]  # Database name
collection = db["Images1"]
images_dir = os.path.dirname(os.path.abspath(__file__)) + "/downloaded_images/"


formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)s] %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "magenta",
    },
)


# Define the cookie
cookies = {"ApacheSessionID": "a79fc81857af5def72edd1c20e45db78651afe34"}


def replace_image_url(url):
    return re.sub(
        r"https://image\.thecrag\.com/[^/]+/",
        "https://image.thecrag.com/original-image/",
        url,
    )


def traverse_and_download_images(document, base_path):
    if isinstance(document, dict):
        for key, value in document.items():
            if key == "Image" and isinstance(value, list):
                for url in value:
                    # url = replace_image_url(url)
                    image_name_with_extension = os.path.basename(url)
                    image_name, _ = os.path.splitext(image_name_with_extension)
                    sanitized_base_path = base_path.replace("/", "_")
                    download_path = os.path.join(
                        images_dir, f"{sanitized_base_path}_{image_name}"
                    )
                    downloaded_image_path = download_image(url, download_path, "path")

            else:
                new_base_path = os.path.join(base_path, key)
                traverse_and_download_images(value, new_base_path, "path")
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
        {"_id": "hierarchy_root"}, {"$set": {"World": nested_dict["World"]}}
    )

    print(f"Path {path} added successfully.")


def insert_initial_document():
    nested_document = {"World": {}, "PageNum": 1, "imageNumPage": {}}

    # Assign a unique _id if desired
    nested_document["_id"] = "hierarchy_root"

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
        {"$set": {"PageNum": page_num, "ImageNumPerPage": image_num_per_page}},
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
            {"$set": {"imageNumPage": document["imageNumPage"]}},
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
        del existing_document["_id"]
    else:
        print("Hierarchy document not found. Inserting initial document first.")
        insert_initial_document()
        existing_document = collection.find_one({"_id": "hierarchy_root"})
        if existing_document:
            del existing_document["_id"]
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
        del document["_id"]
        print(json.dumps(document, indent=4, ensure_ascii=False))
    else:
        print("Hierarchy document not found.")


def signal_handler(sig, frame):
    print("You pressed Ctrl+C! Exiting gracefully...")
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


class Crawler:
    def __init__(
        self,
        playwright: Playwright,
        user_agent: str,
        proxy_server: str = None,
        images_dir: str = "images",
    ):
        self.playwright = playwright
        self.user_agent = user_agent
        self.proxy_server = proxy_server
        self.images_dir = images_dir
        self.browser = None
        self.context = None
        self.image_exist = 0

        # Create a logger
        self.logger = logging.getLogger("colored_logger")
        self.logger.setLevel(logging.DEBUG)  # Adjust the logging level as needed

        # Create console handler and set the formatter
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # Add handler to the logger
        self.logger.addHandler(ch)

    def download_image(self, image_url, download_path, path_parts):
        """
        Downloads an image by navigating to its URL and saving the page content.

        Args:
            image_url (str): The URL of the image.
            download_path (str): The base directory where images will be saved.
            path_parts (list): A list to form subdirectory structure.

        Returns:
            str or None: The file path where the image was saved, or None if failed.
        """
        try:
            page = self.context.new_page()
            response = page.goto(image_url)

            if response and response.ok:
                content_type = response.headers.get("content-type", "")
                extension = (
                    mimetypes.guess_extension(content_type.split(";")[0]) or ".jpg"
                )

                file_name = os.path.basename(image_url)
                if not file_name.lower().endswith(extension):
                    file_name += extension

                combined_path = "_".join(path_parts).replace("/", "").replace("\\", "")
                combined_file_name = combined_path + "_" + file_name
                download_image_path = os.path.join(download_path, combined_file_name)
                os.makedirs(os.path.dirname(download_image_path), exist_ok=True)

                # Save the image content to disk
                with open(download_image_path, "wb") as f:
                    f.write(response.body())
                self.logger.info(f"Image downloaded and saved to {download_image_path}")

                return download_image_path
            else:
                self.logger.error(f"Failed to navigate to image URL: {image_url}")
                return None
        except Exception as e:
            self.logger.error(f"An error occurred while downloading the image: {e}")
            return None

    def check_service_unavailable(self, page, browser):
        if page.locator('text="Service Unavailable"').count() > 0:
            print("Service Unavailable page detected. Closing browser.")
            return True
        elif (
            page.locator(
                'text="Potrdite, da ste človek, tako da opravite spodnje dejanje."'
            ).count()
            > 0
        ):
            print("Human verification page detected. Closing browser.")
            return True
        elif (
            page.locator(
                'text="Preverjanje, ali ste človek. To lahko traja nekaj sekund."'
            ).count()
            > 0
        ):
            print("Human verification page detected. Closing browser.")
            return True
        elif (
            page.locator(
                'text="You have exceeded your allowed anonymous requests. Please login to continue or come back later."'
            ).count()
            > 0
        ):
            print("Human verification page detected. Closing browser.")
            return True
        elif page.locator('text="Internal Server Error"').count() > 0:
            print("Internal Server Error")
            return True
        return False

    def handle_sock_interaction(self, page, pageNumber):

        try:
            photo_list_handle = page.locator(".photo-list")
            photo_list_handle.wait_for(state="visible", timeout=70000)
            self.logger.info("Photo-list is visible.")
        except PlaywrightTimeoutError:
            self.logger.error("Timeout waiting for photo-list to be visible.")
            return False

        link_photos = page.locator(".link-photo")
        try:
            link_photos.first.wait_for(state="visible", timeout=10000)
            self.logger.info("At least one link-photo element is visible.")
        except PlaywrightTimeoutError:
            self.logger.error("Timeout waiting for link-photo elements to be visible.")
            return False

        if link_photos.count() == 0:
            self.logger.warning("No link-photo elements found.")
            return False

        data_list = []
        # Iterate over each 'link-photo' div
        for i in range(
            get_image_num_per_page(), link_photos.count()
        ):  # Ensure get_image_num_per_page() is defined
            image = link_photos.nth(i).locator("a.photo")
            title_elements = link_photos.nth(i).locator("p.title a")

            if title_elements.count() > 0:
                title_element = (
                    title_elements.nth(1)
                    if title_elements.count() > 1
                    else title_elements.first
                )
                title = title_element.get_attribute("title")

                if title:
                    title_parts = [
                        part.replace("\xa0", " ").replace(" ", "").strip()
                        for part in title.split("›")
                    ]
                else:
                    self.logger.warning(f"No title attribute found for image {i}.")
                    title_parts = ["Unknown"]
            else:
                self.logger.warning(f"No <a> element found for image {i}.")
                title_parts = ["Unknown"]

            element_href = image.get_attribute("href")
            if not element_href:
                self.logger.warning(f"No href attribute found for image {i}. Skipping.")
                return False

            full_url = f"https://www.thecrag.com{element_href}"

            data_dict = {"title_parts": title_parts, "full_url": full_url}
            data_list.append(data_dict)

        for i, data in enumerate(data_list, start=get_image_num_per_page()):
            self.logger.info(f"Index: {i}, Data: {data}")
            title_parts = data["title_parts"]
            full_url = data["full_url"]

            try:
                # Restart the browser before navigating to the new URL
                self.logger.info("Restarting browser before navigating to new URL.")
                self.restart_browser()
                new_page = self.context.new_page()
                if not new_page:
                    self.logger.error("Failed to restart browser. Skipping this URL.")
                    return False
                print(full_url)
                new_page.goto(full_url, timeout=50000)
                self.logger.info(f"Navigated to {full_url}")
            except PlaywrightTimeoutError:
                self.logger.error(f"Timeout while trying to navigate to {full_url}")
                return False
            except PlaywrightError as e:
                self.logger.error(
                    f"An error occurred while navigating to {full_url}: {e}"
                )
                return False

            if self.check_service_unavailable(
                new_page, self.browser
            ):  # Ensure this method is defined
                self.logger.error(
                    "Service is unavailable. Exiting interaction handler."
                )
                return False

            try:
                # view_full_size_element = new_page.locator('div:not(#photoMessaging) .link-photo img')
                view_full_size_element = new_page.locator(
                    '//div[contains(@class,"link-photo")]//img[not(ancestor::div[@id="photoMessaging"]) and not(ancestor::form)]'
                )
                view_full_size_element.wait_for(state="visible", timeout=10000)
                self.logger.info("View full size element found.")
            except PlaywrightTimeoutError:
                self.logger.error("Timeout waiting for view full size element.")
                return False

            if view_full_size_element:
                src = view_full_size_element.get_attribute("src")
                data_src = view_full_size_element.get_attribute("data-src")

                if (
                    src
                    and not src.startswith("data:")
                    and not src.lower().endswith("placeholder")
                ):
                    image_url = src
                elif data_src:
                    image_url = data_src
                else:
                    self.logger.warning(f"No valid image URL found for index {i}.")
                    return False
            else:
                self.logger.warning("View full size element not found.")
                return False

            if not image_url:
                self.logger.warning(f"No valid image URL found for index {i}.")
                return False

            self.restart_browser()

            # Ensure this function is defined

            # Append the new image URL to the "Image" array

            downloaded_image = self.download_image(image_url, images_dir, title_parts)
            if downloaded_image:
                update_page_num(pageNumber, i + 1)
                self.logger.info(f"Processed image index: {i}")

                update_document_with_path(title_parts, [image_url])
            else:
                self.logger.error(f"Failed to download image at index {i}.")
                self.image_exist = self.image_exist + 1
                print(self.image_exist)
                if self.image_exist > 2:
                    update_page_num(pageNumber, i + 1)
                    self.image_exist = 0
                return False

        return True

    def close_browser(self):
        """
        Closes the current Playwright browser instance.
        """
        try:
            if self.browser:
                self.browser.close()
                self.logger.info("Browser closed successfully.")
        except PlaywrightError as e:
            self.logger.error(f"Error closing browser: {e}")

    def restart_browser(self):
        """
        Restarts the Playwright browser by closing the current instance and launching a new one.

        Returns:
            page (Playwright Page object): A new browser page after restart.
        """
        self.close_browser()
        # time.sleep(1)  # Brief pause to ensure the browser has closed properly
        self.launch_browser()

    def launch_browser(self):
        """
        Launches a Playwright browser with predefined settings and creates a browser context.

        Args:
            p (Playwright): The Playwright instance.

        Returns:
            bool: True if the browser and context are launched successfully, False otherwise.
        """
        try:
            self.browser = self.playwright.chromium.launch(
                proxy={"server": self.proxy_server} if self.proxy_server else None,
                slow_mo=500,  # Moderate slow-down to mimic human interaction
                # headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--disable-infobars",
                    "--enable-automation",
                    "--no-first-run",
                    "--enable-webgl",
                    "--window-size=1920,1080",
                ],
            )
            self.context = self.browser.new_context(
                user_agent=self.user_agent,
                viewport={"width": 1920, "height": 1080},
                screen={"width": 1920, "height": 1080},
                device_scale_factor=1,
            )
            self.context.add_cookies(
                [
                    {
                        "name": "ApacheSessionID",
                        "value": "a79fc81857af5def72edd1c20e45db78651afe34",
                        "domain": "www.thecrag.com",
                        "path": "/",
                        "expires": -1,  # Session cookie
                        "httpOnly": False,
                        "secure": False,
                        "sameSite": "None",
                    }
                ]
            )

            self.context.set_default_timeout(30000)  # 30 seconds default timeout
            self.logger.info("Browser and context launched successfully.")
            return True

        except PlaywrightError as e:
            self.logger.error(f"Failed to launch browser: {e}")
            return False

        except PlaywrightError as e:
            self.logger.error(f"Failed to launch browser: {e}")
            return False

    def crawl(self, url: str):
        """
        Main crawling loop that processes pages.
        """
        self.launch_browser()
        if not self.browser or not self.context:
            self.logger.error("Browser initialization failed. Exiting crawl loop.")
            return

        while True:
            pageNumber = get_page_num()  # Define this function based on your logic
            self.logger.info(f"Processing page {pageNumber}...")
            try:
                page = self.context.new_page()
                response = page.goto(
                    f"{url}?page={pageNumber}", wait_until="networkidle", timeout=30000
                )
                if response and response.ok:
                    success = self.handle_sock_interaction(page, pageNumber)
                    if success:
                        update_page_num(
                            pageNumber + 1, 0
                        )  # Define this function based on your logic
                    else:
                        self.logger.info(
                            f"Handling interaction failed for page {pageNumber}. Restarting browser."
                        )
                        continue
                else:
                    self.logger.warning(
                        f"Failed to load page {pageNumber}. Retrying..."
                    )

            except PlaywrightTimeoutError:
                self.logger.error(
                    f"Timeout while navigating to {url}?page={pageNumber}"
                )
            except PlaywrightError as e:
                self.logger.error(f"Error during navigation: {e}")

            # Close the page after processing
            self.restart_browser()
            # time.sleep(3)


def canonicalize_url(url):
    if url.endswith("/"):
        url = url[:-1]
    # Use urlsplit to extract the netloc part
    netloc = urlsplit(url).netloc

    return netloc


if __name__ == "__main__":
    base_url = "https://www.thecrag.com/en/climbing/world/photos"
    proxy_server = "socks5://127.0.0.1:9050"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"
    with sync_playwright() as p:
        crawler = Crawler(
            playwright=p,
            user_agent=user_agent,
            proxy_server=proxy_server,
            images_dir="downloaded_images",
        )
        crawler.crawl(url=base_url)
    # update_document_with_downloaded_images()
