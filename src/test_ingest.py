import os
from ingest import extract_text_with_ocr  # We import your function directly

# Path to your test file
TEST_FILE = "./data/test.pdf" 

def test_ocr_quality():
    print(f"🔬 Testing OCR on: {TEST_FILE}")
    
    if not os.path.exists(TEST_FILE):
        print("❌ File not found! Please add a PDF to /data/ folder first.")
        return

    # Force the OCR function to run (skipping the normal text check for this test)
    print("⏳ Running RapidOCR... (This may take a few seconds)")
    text = extract_text_with_ocr(TEST_FILE)
    
    print("\n" + "="*40)
    print("       OCR RESULT (What the bot sees)       ")
    print("="*40)
    
    if text:
        # Print first 1000 characters to verify
        print(text[:100000]) 
        print("\n" + "="*40)
        print(f"✅ Extracted {len(text)} characters.")
        print("👀 Check the text above. Are there typos? Is it gibberish?")
    else:
        print("❌ No text extracted. The image might be too blurry or blank.")

if __name__ == "__main__":
    test_ocr_quality()