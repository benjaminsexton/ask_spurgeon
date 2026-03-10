from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone
import os

# ─────────────────────────────────────────────
# CONFIGURATION — fill these in
# ─────────────────────────────────────────────
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX   = "spurgeon-sermons"
EMBEDDING_MODEL  = "text-embedding-3-small"
TOP_K            = 8   # how many chunks to retrieve before summarizing

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def rewrite_query(user_query):
    """Expand the user's query into a richer theological search query."""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a theological search assistant helping retrieve relevant passages "
                    "from Charles Spurgeon's sermons. Rewrite the user's query into a rich, "
                    "expanded theological search query that will surface the most relevant sermon "
                    "passages. Include related theological concepts, synonyms, and scripture themes. "
                    "Return only the rewritten query — no explanation, no preamble."
                )
            },
            {"role": "user", "content": user_query}
        ],
        max_tokens=150,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def search_sermons(query, top_k=TOP_K):
    """Embed the query and search Pinecone."""
    response = openai_client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    embedding = response.data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches


def generate_response(user_query, chunks):
    """Generate a summary and select the best quotes from retrieved chunks."""
    context = ""
    for i, match in enumerate(chunks):
        meta = match.metadata
        context += f"\n---\nSermon: \"{meta.get('title', 'Unknown')}\"\n"
        context += f"Scripture: {meta.get('scripture', '')}\n"
        context += f"Volume: {meta.get('volume', '')}\n"
        context += f"Text: {meta.get('text', '')}\n"

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant for Charles Spurgeon's sermons. "
                    "Based ONLY on the sermon passages provided, answer the user's question. "
                    "Return your response as JSON with this exact structure:\n"
                    "{\n"
                    '  "summary": "2-3 sentence summary of what Spurgeon said on this topic",\n'
                    '  "quotes": [\n'
                    '    {\n'
                    '      "text": "the most relevant quote from the passage (2-4 sentences)",\n'
                    '      "sermon": "sermon title",\n'
                    '      "scripture": "scripture reference",\n'
                    '      "volume": "volume number"\n'
                    '    }\n'
                    '  ]\n'
                    "}\n"
                    "Include 2-3 quotes. Return ONLY the JSON, no markdown, no explanation."
                )
            },
            {
                "role": "user",
                "content": f"Question: {user_query}\n\nSermon passages:\n{context}"
            }
        ],
        max_tokens=1000,
        temperature=0.3
    )

    import json
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Step 1: Rewrite query for better retrieval
        rewritten = rewrite_query(user_query)

        # Step 2: Search Pinecone
        chunks = search_sermons(rewritten)

        if not chunks:
            return jsonify({"error": "No relevant sermons found"}), 404

        # Step 3: Generate summary + quotes
        result = generate_response(user_query, chunks)
        result["rewritten_query"] = rewritten  # useful for debugging

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Ask Spurgeon is running at http://localhost:5000")
    app.run(debug=False, port=5000, host="0.0.0.0")
