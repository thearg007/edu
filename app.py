import os
import time
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import core

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# globals bc passing args is for tryhards 🥸
global_retriever = core.Retriever()
query_cache = {}

RATE_PER_TOKEN = 0.0001 # rent is due so we tracking every cent 💸

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # blending the pdf into little chunks like a smoothie 🍓
            text = core.extract_text_from_pdf(filepath)
            chunks = core.chunk_text(text)
            global_retriever.build_index(chunks)
            
            # wiping the memories clean 🧼
            query_cache.clear()
            
            return jsonify({"message": f"Successfully processed {len(chunks)} chunks.", "chunks": len(chunks)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query')
    mode = data.get('mode', 'baseline') 
    answer_type = data.get('answer_type', 'detailed')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
        
    start_time = time.time()
    
    # chat is this cached? 💀
    cache_key = f"{query}_{mode}_{answer_type}"
    if cache_key in query_cache:
        cached_result = query_cache[cache_key]
        cached_result['metrics']['latency'] = round(time.time() - start_time, 2)
        return jsonify(cached_result)

    # fetching the receipts 🧾
    retrieved_chunks = global_retriever.retrieve(query, top_k=5)
    raw_context = "\n\n".join(retrieved_chunks)
    
    if not raw_context:
        return jsonify({
            "answer": "Bruh upload a textbook first or try a different query.",
            "metrics": {
                "original_tokens": 0, "compressed_tokens": 0, "reduction": 0,
                "cost_before": 0, "cost_after": 0, "savings": 0, "latency": round(time.time() - start_time, 2), "mode": mode
            }
        })
    
    original_tokens = core.count_tokens(raw_context)
    
    if mode == 'optimized':
        # gatekeeping the long text 💅 -> short king context
        pruned_context = core.compress_context(raw_context)
        final_context = pruned_context
    else:
        final_context = raw_context
        
    final_tokens = core.count_tokens(final_context)
    
    # making gemini do our homework 🧠
    answer = core.generate_response(query, final_context, answer_type)
    
    latency = round(time.time() - start_time, 2)
    
    reduction = 0
    if original_tokens > 0:
        reduction = round(((original_tokens - final_tokens) / original_tokens) * 100, 2)
        
    # girl math savings 🎀
    cost_before = round(original_tokens * RATE_PER_TOKEN, 4)
    cost_after = round(final_tokens * RATE_PER_TOKEN, 4)
    savings = round(cost_before - cost_after, 4)
    
    result = {
        "answer": answer,
        "metrics": {
            "original_tokens": original_tokens,
            "compressed_tokens": final_tokens,
            "reduction": reduction,
            "cost_before": cost_before,
            "cost_after": cost_after,
            "savings": savings,
            "latency": latency,
            "mode": mode
        }
    }
    
    query_cache[cache_key] = result
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
