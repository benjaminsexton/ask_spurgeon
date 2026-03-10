# Ask Spurgeon — Ingestion Script

This script takes all 3,500+ Spurgeon sermons from the cloned GitHub repo,
breaks them into passages, and loads them into Pinecone so they can be searched later.
You only ever run this once.

---

## What it does, step by step

1. **Finds every sermon file** — scans all 63 volume folders for `.md` files
2. **Parses each sermon** — pulls out the title, scripture reference, and body text
3. **Chunks the text** — splits each sermon into ~500 word passages with a little overlap so no idea gets cut off mid-thought
4. **Embeds each chunk** — sends each passage to OpenAI, which converts it into a list of numbers (a "vector") that represents its meaning
5. **Uploads to Pinecone** — stores each vector alongside the original text and metadata (title, scripture, volume) so you can retrieve it later

---

## Setup

### 1. Install Python dependencies

Open a terminal in this folder and run:

```bash
pip install openai pinecone
```

### 2. Fill in your API keys

Open `ingest.py` and replace the placeholder values at the top:

```python
OPENAI_API_KEY   = "sk-your-openai-key-here"      # from platform.openai.com
PINECONE_API_KEY = "your-pinecone-key-here"        # from app.pinecone.io
```

Make sure `SERMONS_PATH` points to where you cloned the repo:

```python
SERMONS_PATH = r"C:\Users\yourg\Downloads\Ask Spurgeon"
```

### 3. Run it

```bash
python ingest.py
```

It will print progress as it goes. Expect it to take **30–60 minutes** to process
all 3,500 sermons. You can leave it running in the background.

---

## What happens in Pinecone

Each chunk of sermon text gets stored as a "vector" with this metadata attached:

| Field         | Example                          |
|---------------|----------------------------------|
| `title`       | "Sovereign Grace and Man's Responsibility" |
| `scripture`   | "Ezekiel 18:31"                  |
| `volume`      | 10                               |
| `chunk_index` | 2                                |
| `text`        | the raw passage text             |

This metadata is what gets returned when someone searches — it's how we show
the sermon title and scripture reference alongside each quote.

---

## After ingestion

Once this finishes, the next step is building the search API — the part that takes
a user's question, rewrites it for better retrieval, searches Pinecone, and returns
a summary + quotes. That's a separate script we'll build next.

---

## Cost estimate

- ~3,500 sermons × ~10 chunks each = ~35,000 chunks
- OpenAI embedding cost: roughly **$0.07 total** (very cheap)
- Pinecone free tier: handles this easily
