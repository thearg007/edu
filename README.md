# PRUNEnLEARN 

Welcome to PRUNEnLEARN! This system provides a low-cost, curriculum-aligned educational assistant that ingests textbook PDFs, pulls highly relevant context, and significantly reduces API token costs using **ScaleDown** before generating answers with Gemini.

## 🚀 Setup Steps

1. **Clone or Download** this repository.
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Mac/Linux
   .\venv\Scripts\activate        # On Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Your API Keys**:
   Open `core.py` and replace `YOUR_SCALEDOWN_API_KEY` and `YOUR_GEMINI_API_KEY` with your actual keys.
   *(Alternatively, set `GEMINI_API_KEY` and `SCALEDOWN_API_KEY` as environment variables.)*
5. **Run the Application**:
   ```bash
   python app.py
   ```
6. **Access the Demo**: Open `http://localhost:5000` in your web browser.

## 🔌 API Explanation

The backend interacts with two key APIs in sequence for the Optimized Mode:
1. **ScaleDown API (`api.scaledown.xyz/compress/raw/`)**: Once the semantic search component (TF-IDF scikit-learn) retrieves the top paragraphs from the PDF, this context is sent to ScaleDown. ScaleDown acts as a context pruner that semantically extracts and condenses the critical facts while removing fluff and repetition.
2. **Gemini API (`google.generativeai`)**: The heavily compressed token string is then passed as system context to the Gemini LLM to answer the user's specific query. 

### How ScaleDown Reduces Tokens
When a 1,000-token chunk of context is pulled from a textbook, much of it might consist of transitional sentences, examples not strictly relevant to the query, or redundant phrasing. ScaleDown rewrites the context precisely, often reducing the size by 50-80% without losing the facts required by Gemini to answer the question accurately.

## ⚡ Demo Instructions

1. **Start the server**.
2. **Upload Text**: Use the document setup section to upload a PDF (e.g., `sample.pdf`).
3. **Run Baseline**: Type a question like *"Explain photosynthesis"* and click **Run Baseline Mode**. Observe the token size and estimated cost limits.
4. **Run Optimized**: Click **Run Optimized Mode**. 
5. **Compare**: View the Metrics Panel table at the bottom. You will see the percentage reduction in tokens, and the real-time calculated cost savings compared to the Baseline mode side-by-side!

### Screenshots
*(Include screenshots of the Metrics table comparing Baseline vs. Optimized modes here.)*
