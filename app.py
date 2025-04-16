from flask import Flask, request, jsonify
from rag_engine import qa_chain

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "Missing 'question' field"}), 400

    result = qa_chain.invoke(query)
    cleaned_answer = result["result"].replace("\\n", "\n").replace("\\t", "\t")
    return jsonify({
        "Answer": cleaned_answer
#      ,  "sources": [doc.metadata["source"] for doc in result["source_documents"]]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
