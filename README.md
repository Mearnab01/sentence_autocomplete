# Smart Autocomplete System

A real-time, production-ready autocomplete system that compares Deep Learning (PyTorch LSTM) and LLM-based (Hugging Face) approaches.

## 🧪 Phase 1: Dataset & Validation Notebook

Before running the full system, you must validate the ML logic using the provided Jupyter Notebook. This ensures the tokenizer, sequence generator, and LSTM behave properly.

### 1. Dataset Instructions
We use an accessible public text dataset (Tiny Shakespeare) as a fallback so that no Kaggle API key is required. The notebook downloads this automatically using `urllib.request`. 
If you prefer a Kaggle dataset, install `kaggle` (`pip install kaggle`), authenticate, and run `kaggle datasets download -d <dataset-name>`.

### 2. Running the Notebook
1. Install Jupyter and packages in your virtual environment:
   ```bash
   pip install notebook torch
   ```
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `demo.ipynb` in your browser. Run all cells to see the data loaded, preprocessed, vectorized, modeled with an LSTM, trained, and tested for next-word generation.

### 3. Transition to Modular Code (IMPORTANT)
After validating `demo.ipynb`, the notebook logic is mapped into modular files seamlessly:
- **`data_loader.py`**: Instead of monolithic cells, the logic is in `backend/models/dataset.py` via an extensible `TextSequenceDataset` class.
- **`model.py`**: The neural network is encapsulated inside `backend/models/dl_model.py` under the class `LSTMModel`.
- **`train.py`**: The training pipeline lives in `backend/services/training.py` enabling backend-triggered jobs without blocking endpoints.
- **`predict.py`**: The inference executes in `backend/services/inference.py` (`predict_dl`) utilizing advanced caching.

---

## 🧰 Phase 2: MODULAR ENVIRONMENT SETUP (MANDATORY)

You must include full setup instructions so the project runs locally without confusion.

### 1. Virtual Environment

Create a Python virtual environment inside the `backend` folder:

**Command (Windows):**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
```

**Command (Mac/Linux):**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Dependencies

All required packages are inside the `requirements.txt` file. 

The required libraries include:
- `fastapi`
- `uvicorn`
- `torch`
- `transformers`
- `python-dotenv`
- `huggingface_hub`
- `pydantic`
- `aiohttp`

Provide command to install them:
```bash
pip install -r requirements.txt
```

*(Optional: Set your Hugging Face API key in your environment to use the LLM mode)*
```bash
export HUGGINGFACE_API_KEY="your_api_key_here"
```

---

### 3. Running Backend Server

To start the FastAPI server, run this command from the `backend` folder:
```bash
uvicorn main:app --reload
```

Ensure the backend runs on:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### 4. Frontend Setup

Open a **new terminal**, navigate to the `frontend` folder, and install node modules:
```bash
cd frontend
npm install
npm run dev
```

Ensure the frontend runs on:
[http://localhost:3000](http://localhost:3000)

---

### 5. Browser Access

To use the system, follow these steps:
1. Open your browser.
2. Go to the frontend URL: [http://localhost:3000](http://localhost:3000)
3. Type text into the input box and see autocomplete suggestions.

---

### 6. API Testing

You can test the API backend endpoints without the browser UI:

- **Browser (Interactive Docs)**: Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to access the integrated Swagger UI for testing `/predict` and `/train`.
- **cURL or Postman**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/predict" \
       -H "Content-Type: application/json" \
       -d '{"text": "The quick brown fox", "mode": "llm_mode", "top_k": 3}'
  ```

---

### 7. Common Errors Handling

- **`venv` not activating?**
  - *Cause*: Windows execution policies block scripts.
  - *Fix*: Open PowerShell as Administrator and run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`, then try activating `venv` again.

- **Module not found?**
  - *Cause*: Dependencies were not installed or installed outside the virtual environment.
  - *Fix*: Ensure your virtual environment is activated (you should see `(venv)` on the left of your terminal prompt) and re-run `pip install -r requirements.txt`.

- **Port already in use?**
  - *Cause*: Another application is occupying port 8000 (Backend) or port 3000 (Frontend).
  - *Fix*: Kill the process occupying the port, OR start the server on a different port:
    - Backend: `uvicorn main:app --port 8001 --reload`
    - Frontend: `npm run dev -- --port 3001`

---

## 🚀 Additional Details

- **Deep Learning Mode**: A local PyTorch LSTM model trained from scratch with word-level embedding. Once you manually start training via the API (`/train`), it sequentially learns.
- **LLM-Based Mode**: A context-aware integration with the Hugging Face API to fetch intelligent sentence completions using zero-shot inference.
- **Caching**: Contains backend caching to avoid re-rendering duplicate completions.

<!---LeetCode Topics Start-->
# LeetCode Topics
## Array
|  |
| ------- |
| [0049-group-anagrams](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0049-group-anagrams) |
| [0057-insert-interval](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0057-insert-interval) |
| [0153-find-minimum-in-rotated-sorted-array](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0153-find-minimum-in-rotated-sorted-array) |
| [0217-contains-duplicate](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0217-contains-duplicate) |
## Hash Table
|  |
| ------- |
| [0049-group-anagrams](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0049-group-anagrams) |
| [0217-contains-duplicate](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0217-contains-duplicate) |
| [0242-valid-anagram](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0242-valid-anagram) |
## Sorting
|  |
| ------- |
| [0049-group-anagrams](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0049-group-anagrams) |
| [0217-contains-duplicate](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0217-contains-duplicate) |
| [0242-valid-anagram](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0242-valid-anagram) |
## Binary Search
|  |
| ------- |
| [0153-find-minimum-in-rotated-sorted-array](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0153-find-minimum-in-rotated-sorted-array) |
## Two Pointers
|  |
| ------- |
| [0125-valid-palindrome](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0125-valid-palindrome) |
## String
|  |
| ------- |
| [0020-valid-parentheses](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0020-valid-parentheses) |
| [0049-group-anagrams](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0049-group-anagrams) |
| [0125-valid-palindrome](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0125-valid-palindrome) |
| [0242-valid-anagram](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0242-valid-anagram) |
## Stack
|  |
| ------- |
| [0020-valid-parentheses](https://github.com/Mearnab01/sentence_autocomplete/tree/master/0020-valid-parentheses) |
<!---LeetCode Topics End-->