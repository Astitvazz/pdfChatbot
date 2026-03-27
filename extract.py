import fitz

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    
    text = extract_text("data/sample.pdf")
    print("---")
    print(f"Total characters extracted: {len(text)}")
    print(text[:500])