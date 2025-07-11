# ContextoESL

ContextoESL is an interactive, educational word-guessing game designed to help English-as-a-Second-Language (ESL) learners improve their vocabulary, grammar, and language skills through contextual feedback, grammar hints, translations, and definitions.

# Key Features

- Word Guessing Game: Inspired by Contexto, users guess a secret word based on contextual similarity.
- Sentence-Based Hints: Enter a sentence using your best guess and receive grammar suggestions with explanation. If your sentence is correct, you receive a hint word which is halfway in rank between your best guess and target word.
- Linguistic Acceptability Check: Validates your sentence using the [CoLA (Corpus of Linguistic Acceptability)] dataset.
- "Learn More" Popup: Shows the latest guessed word's:
  - Meaning (via Dictionary API)
  - Pronunciation + audio
  - Translation to your chosen native language
- Multilingual Support: User can pick their preferred language (e.g. Hindi, Spanish, French) for translation.
- Semantic Ranking: Uses Sentence-BERT to provide similarity scores between your guess and the target word.
- Win Screen: Displays congratulations, total guesses taken, and a restart button.

---

# Tech Stack

| Layer        | Tools Used                                                                  |
|--------------|-----------------------------------------------------------------------------|
| Backend      | Python, Flask                                                               |
| NLP/NLU      | Sentence-BERT, LanguageTool, CoLA, MyMemory API, Dictionary API             |
| Frontend     | HTML, CSS, JavaScript                                                       |                                        |

---

# Run the Project Locally
- Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate     # For Windows
- Run the file generate_vectors.py to generate word embeddings

# Prerequisites

- Python 3.8+
- pip
- Virtual Environment (venv)
- Internet Connection(for first run only)

# Installation Steps

```bash
git clone https://github.com/your-username/ContextoESL.git
cd ContextoESL
pip install -r requirements.txt
python generate_vectors.py
python app.py
