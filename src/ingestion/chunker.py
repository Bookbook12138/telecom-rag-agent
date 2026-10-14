def split_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """简单文本分块"""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
