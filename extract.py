# This file only handles reading text out of a PDF.

import fitz


def extract_text(pdf_path):
    # This function opens a PDF file and joins text from every page into one string.
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


if __name__ == "__main__":
    # This block is for simple terminal testing.
    text = extract_text("data/sample.pdf")
    print("---")
    print(f"Total characters extracted: {len(text)}")
    print(text[:500])
