import os
import shutil
import re # Add this at the top
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import DB_PATH, DATA_PATH

# Initialize OCR engine once (Global)
ocr_engine = RapidOCR()

def clean_text(text):
    """
    Fixes common PDF formatting issues.
    """
    # 1. Merge hyphenated words at line breaks (e.g., "com-\nputer" -> "computer")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 2. Fix weird newlines in the middle of sentences
    # (Replaces a newline with a space if it's not followed by a capital letter)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 3. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_text_with_ocr(file_path):
    """
    Advanced extraction: Converts PDF pages to images, then reads text using RapidOCR.
    Used for scanned documents.
    """
    print(f"     [!] No text found. Activating OCR (Computer Vision)...")
    full_text = ""
    
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # 1. Convert page to image (Pixmap)
            pix = page.get_pixmap(dpi=300) # High DPI for better reading
            img_bytes = pix.tobytes("png")
            
            # 2. Run OCR
            result, _ = ocr_engine(img_bytes)
            
            if result:
                # result format: [[box, text, score], ...]
                page_text = "\n".join([line[1] for line in result])
                full_text += page_text + "\n"
                
        pdf_document.close()
        return full_text
        
    except Exception as e:
        print(f"     ❌ OCR Failed: {e}")
        return ""

def load_and_tag_books():
    documents = []
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Folder '{DATA_PATH}' not found.")
        return []

    files = [f for f in os.listdir(DATA_PATH) if f.endswith((".txt", ".pdf"))]
    
    if not files:
        print("❌ No valid files found in /data folder!")
        return []

    print(f"[*] Found {len(files)} documents. Starting ingestion...")

    for filename in files:
        file_path = os.path.join(DATA_PATH, filename)
        
        # Determine Book Index
        try:
            book_index = int(''.join(filter(str.isdigit, filename)))
        except ValueError:
            book_index = 99
        
        print(f"   Processing: {filename} (Index: {book_index})")

        try:
            extracted_text = ""
            
            if filename.endswith(".pdf"):
                # Try 1: Fast Extraction (Standard)
                try:
                    loader = PyPDFLoader(file_path)
                    raw_docs = loader.load()
                    # Combine all pages to check content
                    extracted_text = "\n".join([d.page_content for d in raw_docs])
                except:
                    extracted_text = ""

                # Try 2: OCR Fallback (If text is empty or too short)
                if len(extracted_text.strip()) < 50:
                    extracted_text = extract_text_with_ocr(file_path)

            elif filename.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        extracted_text = f.read()
                except:
                    # Windows encoding fallback
                    with open(file_path, "r", encoding="cp1252") as f:
                        extracted_text = f.read()

            # Create a LangChain Document
            if extracted_text:
                cleaned_text = clean_text(extracted_text)
                
                doc = Document(
                    page_content=cleaned_text, # Use cleaned version
                    metadata={"source": filename, "book_index": book_index}
                )
                documents.append(doc)
                print(f"     [+] Successfully extracted {len(cleaned_text)} characters.")
            else:
                print("     ⚠️ Warning: Could not extract any text.")

        except Exception as e:
            print(f"   ❌ Critical Error loading {filename}: {e}")

    return documents


def chunk_data_semantically(documents):
    """
    Splits text based on MEANING (Semantic Similarity) rather than fixed size.
    """
    print("🧠 Initializing Semantic Chunker...")
    print("   (Note: This is slower than standard splitting, please wait...)")
    
    # We use this model to measure how similar two sentences are
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Breakpoint: Split if the topic shifts significantly (95th percentile difference)
    text_splitter = SemanticChunker(
        embedding_model, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95 
    )
    
    # Since SemanticChunker works on raw text mostly, we map it over docs carefully
    # or just let it handle the list of documents directly if supported.
    # Note: SemanticChunker .split_documents is standard in recent versions.
    chunks = text_splitter.split_documents(documents)
    
    print(f"[+] Created {len(chunks)} semantic chunks.")
    return chunks

def save_to_vector_db(chunks):
    """
    Saves the smart chunks to ChromaDB.
    """
    if os.path.exists(DB_PATH):
        # We delete the old DB to ensure a fresh start with the new chunking method
        shutil.rmtree(DB_PATH)
        print("[-] Cleared old database.")

    print("💾 Saving to Vector Database...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    print(f"[+] Database built at {DB_PATH}")
    return db

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load
    raw_docs = load_and_tag_books()
    
    if raw_docs:
        # 2. Chunk (Smart Way)
        smart_chunks = chunk_data_semantically(raw_docs)
        
        # 3. Save
        save_to_vector_db(smart_chunks)
        
        print("\n[*] Ingestion Complete!")