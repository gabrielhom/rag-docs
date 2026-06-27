"""Pure-ish RAG pipeline: parse -> chunk -> embed. Run directly for a self-check."""
import hashlib
import os

# ---- parse -------------------------------------------------------------
def parse_pdf(path):
    """Return [(page_number, text)] for each page with text."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


# ---- chunk -------------------------------------------------------------
def chunk_text(text, size=800, overlap=100):
    """Split text into overlapping windows by character count. Pure, testable."""
    text = text.strip()
    if not text:
        return []
    step = size - overlap
    assert step > 0, "size must be > overlap"
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_pages(pages, size=800, overlap=100):
    """[(page, text)] -> [(page, chunk)]."""
    out = []
    for page, text in pages:
        for c in chunk_text(text, size, overlap):
            out.append((page, c))
    return out


# ---- embed -------------------------------------------------------------
def chunk_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def embed(texts, client=None):
    """Embed a list of strings. client injected for tests; real one is OpenAI."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


# ---- self-check --------------------------------------------------------
if __name__ == "__main__":
    # chunking math: 250 chars, size 100, overlap 20 -> step 80 -> ceil(250/80)=4 windows
    chunks = chunk_text("x" * 250, size=100, overlap=20)
    assert len(chunks) == 4, len(chunks)
    assert chunks[0] == "x" * 100
    assert chunk_text("") == []
    assert chunk_text("   ") == []

    # overlap actually overlaps: window 2 starts at char 80, so shares 20 chars with window 1
    body = "".join(chr(65 + i % 26) for i in range(250))
    c = chunk_text(body, size=100, overlap=20)
    assert c[0][80:100] == c[1][0:20], "overlap broken"

    # page chunking preserves page numbers
    pc = chunk_pages([(1, "a" * 150), (2, "b" * 50)], size=100, overlap=20)
    assert pc[-1][0] == 2 and pc[-1][1] == "b" * 50

    # embed wiring with a fake client (no network)
    class Fake:
        class embeddings:
            @staticmethod
            def create(model, input):
                return type("R", (), {"data": [type("D", (), {"embedding": [0.0] * 1536}) for _ in input]})
    vecs = embed(["hi", "there"], client=Fake())
    assert len(vecs) == 2 and len(vecs[0]) == 1536

    print("pipeline self-check OK")
