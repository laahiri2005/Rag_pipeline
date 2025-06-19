

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import numpy as np
import pickle
from PIL import Image
import io
from huggingface_hub import login

# Login using your HF token (get it from https://huggingface.co/settings/tokens)
login()

# === STEP 1: Load PDF Text with Bounding Box Metadata ===
def load_pdf_text_with_metadata(file_path):
    page_data = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            full_text = page.extract_text()

            # Save raw image of the page
            image_stream = io.BytesIO()
            page_image = page.to_image(resolution=150).original
            page_image.save(image_stream, format='PNG')

            page_data.append({
                "page_num": page_num,
                "text": full_text.strip() if full_text else "",
                "words": words,
                "image_bytes": image_stream.getvalue()
            })
    return page_data

# === STEP 2: Chunk Text and Track Bounding Boxes ===
def chunk_text_with_metadata(page_data, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["

", "
", ".", " "]
    )

    all_chunks = []
    for page in page_data:
        page_text = page["text"]
        page_num = page["page_num"]
        words = page["words"]
        image_bytes = page["image_bytes"]

        if not page_text:
            continue

        chunks = splitter.create_documents([page_text])
        offset = 0

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            start_idx = page_text.find(chunk_text, offset)
            end_idx = start_idx + len(chunk_text)
            offset = end_idx  # To avoid overlapping match with earlier chunk

            # Estimate coordinates based on matching word positions
            chunk_words = [w for w in words if start_idx <= w["doctop"] <= end_idx]
            if chunk_words:
                x0 = min(float(w['x0']) for w in chunk_words)
                x1 = max(float(w['x1']) for w in chunk_words)
                top = min(float(w['top']) for w in chunk_words)
                bottom = max(float(w['bottom']) for w in chunk_words)
                bbox = (x0, top, x1, bottom)
            else:
                bbox = None

            all_chunks.append({
                "content": chunk_text,
                "metadata": {
                    "page_num": page_num,
                    "chunk_index": i,
                    "char_range": (start_idx, end_idx),
                    "bbox": bbox,
                    "image_bytes": image_bytes  # Optional – can skip if not needed
                }
            })

    return all_chunks

# === STEP 3: Embed and Index Chunks ===
def embed_and_index_chunks(chunks_with_meta, model_name="paraphrase-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    texts = [chunk["content"] for chunk in chunks_with_meta]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings, axis=1)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    with open("SOP.pkl", "wb") as f:
        pickle.dump({
            "chunks": chunks_with_meta,
            "embeddings": embeddings
        }, f)

    return model, index, chunks_with_meta

# === STEP 4: Search Function with BBox Info ===
def search(query, model, index, chunks_with_meta, top_k=3, min_score=0.3, show_image=False):
    query_vec = model.encode([query])
    query_vec = normalize(query_vec, axis=1).astype("float32")
    scores, indices = index.search(query_vec, top_k)

    print(f"
🔍 Top {top_k} results for: '{query}'
")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if score < min_score:
            print(f"🔸 Rank {rank+1}: Skipped (Low Score: {score:.2f})")
            continue

        chunk = chunks_with_meta[idx]
        metadata = chunk["metadata"]
        print(f"🔹 Rank {rank+1} (Score: {score:.2f}) | Page {metadata['page_num']} | Chunk {metadata['chunk_index']}")
        print(f"📍 Char Range: {metadata['char_range']} | 📦 BBox: {metadata['bbox']}")
        print(f"{chunk['content'][:700]}...
")

        # Optional: display or save image with bbox overlay
        if show_image and metadata["bbox"]:
            img = Image.open(io.BytesIO(metadata["image_bytes"]))
            draw = ImageDraw.Draw(img)
            draw.rectangle(metadata["bbox"], outline="red", width=2)
            img.show()

# === RUN ALL ===
if __name__ == "_main_":
    file_path = "SOP1.pdf"
    page_data = load_pdf_text_with_metadata(file_path)
    chunks_with_meta = chunk_text_with_metadata(page_data)
    model, index, chunks_with_meta = embed_and_index_chunks(chunks_with_meta)

    # === Interactive Loop ===
    while True:
        user_query = input("❓ Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        search(user_query, model, index, chunks_with_meta, show_image=False)
        
