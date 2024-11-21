from langchain_community.document_loaders import PyPDFLoader
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the PDF document


def clean_index(index):
    """
    Cleans the index by removing invalid or inappropriate entries.

    Args:
        index (list): A list of dictionaries containing "title" and "page".

    Returns:
        list: A cleaned list of dictionaries with valid titles and pages.
    """
    cleaned_index = []
    max_pages = 5000  # Set a maximum limit for the number of pages

    for item in index:
        title = item["title"]
        page = item["page"]

        # Ignore entries with invalid or unrealistic page numbers
        if page <= 0 or page > max_pages:
            continue

        # Remove titles that are just numbers or symbols
        if re.match(r"^[\d\s\W]+$", title):
            continue

        # Clean fragmented or inappropriate titles
        title = re.sub(r"\s+", " ", title).strip()  # Remove extra spaces
        title = re.sub(r"(\w)([A-Z])", r"\1 \2", title)  # Add spaces between concatenated words
        if len(title) < 5:  # Ignore very short titles
            continue

        # Add the corrected item to the cleaned index
        cleaned_index.append({"title": title, "page": page})

    return cleaned_index


def extract_title_page(data):
    """
    Extracts titles and page numbers from the "Contents" section of a PDF.

    Returns:
        list: A list of dictionaries containing "title" and "page" for each index entry.
        If no entries are found, returns None.
    """
    full_text = " ".join(data)

    # Search for the "Contents" section in the text
    contents_match = re.search(r"CONTENTS(\s\w.+)", full_text, re.DOTALL | re.IGNORECASE)
    if contents_match:
        contents_text = contents_match.group(1)
        # logging.info(f"Extracted text from the Contents section: {contents_text[:500]}")

        # Handle line breaks within titles
        contents_text = re.sub(r"(?<=[a-zA-Z])\n(?=[a-zA-Z])", " ", contents_text)

        # Extract titles and page numbers
        matches = re.findall(r"(\d+(\.\d+)*\s+.+?)(?:\.{3,}|\s+)+(\d+)", contents_text)

        if matches:
            index = [{"title": match[0].strip(), "page": int(match[2])} for match in matches]

            # Clean the index
            index = clean_index(index)
            # logging.info(f"Cleaned index titles: {index}")
            return index
        else:
            logging.warning("No titles found in the index.")
            return None
    else:
        logging.warning("Contents section not found in the PDF.")
        return None
