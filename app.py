from flask import Flask, redirect, render_template, request, jsonify, url_for
import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import language_tool_python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

app = Flask(__name__)

#Load CEFR words
with open("words/cefr_words.json") as f:
    cefr = json.load(f)

#Load full English word list
with open("words/10KEnglishWords.txt") as f:
    all_words = [line.strip() for line in f if line.strip()]

#Load word vectors
with open("embeddings/word_vectors_real.pkl", "rb") as f:
    word_vectors = pickle.load(f)

#Home page with difficulty selection
@app.route('/')
def index():
    return render_template('index.html')

# === Start game with difficulty ===
@app.route('/start')
def start_game():
    difficulty = request.args.get('difficulty', 'easy')
    user_lang = request.args.get('lang', 'hi')
    app.config['user_lang'] = user_lang

    level_map = {
        "easy": ["C1", "C2"],
        "medium": ["B1", "B2"],
        "hard": ["A1", "A2"]
    }
    allowed_levels = level_map.get(difficulty.lower(), ["A1", "A2"])
    eligible_words = [word for level in allowed_levels for word in cefr[level] if word in word_vectors]

    secret_word = random.choice(eligible_words)
    print(secret_word)
    secret_vector = np.array(word_vectors[secret_word])

    # Store data globally for now (not ideal, but fine for MVP)
    app.config['secret_word'] = secret_word
    app.config['secret_vector'] = secret_vector

    # Compute similarities
    similarities = {
        word: cosine_similarity([secret_vector], [np.array(vector)])[0][0]
        for word, vector in word_vectors.items()
    }
    app.config['similarities'] = similarities

    ranked_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    rank_map = {word: i + 1 for i, (word, _) in enumerate(ranked_words)}
    app.config['rank_map'] = rank_map

    return render_template('game.html', difficulty=difficulty.capitalize())

# === Guess route ===
@app.route('/guess', methods=['POST'])
def guess():
    word = request.form['word'].lower()
    word_vectors_dict = word_vectors
    similarities = app.config.get('similarities')
    rank_map = app.config.get('rank_map')

    if word not in word_vectors_dict:
        return jsonify({'error': f'"{word}" not found in dictionary. Try another guess.'})

    rank = rank_map.get(word, -1)
    sim = round(similarities[word], 4)

    return jsonify({'word': word, 'rank': rank, 'similarity': sim})


cola_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
cola_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")
cola_model.eval()
tool = language_tool_python.LanguageTool('en-US')

@app.route('/hint', methods=['POST'])
def hint():
    data = request.get_json()
    sentence = data.get("sentence", "").strip()
    best_guess = data.get("best_guess", "")

    #Check if sentence contains the guessed word
    if best_guess not in sentence.lower():
        return jsonify({"error": "Your sentence must include your best guess word."})

    #Run grammar check
    matches = tool.check(sentence)
    if matches:
        grammar_issues = [{
            "message": match.message,
            "suggestions": match.replacements,
            "offset": match.offset,
            "length": match.errorLength,
            "context": match.context,
        } for match in matches]

        return jsonify({"grammar_issues": grammar_issues})

    #CoLA Acceptability Check
    inputs = cola_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = cola_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0][prediction].item()

    if prediction==0:
        return jsonify({
            "error": "Your sentence is grammatically okay, but not acceptable in natural English.",
            "confidence": round(confidence, 2)
        })

    #Give the hint
    rank_map = app.config.get("rank_map")
    similarities = app.config.get("similarities")

    best_rank = rank_map.get(best_guess)
    if not best_rank:
        return jsonify({"error": "Could not find rank of your best guess."})

    target_rank = max(1, best_rank // 2)
    reverse_map = {v: k for k, v in rank_map.items()}
    hint_word = reverse_map.get(target_rank)

    if not hint_word:
        return jsonify({"error": "No hint available."})

    sim = round(similarities[hint_word], 4)
    return jsonify({"hint": hint_word, "rank": target_rank, "similarity": sim})

@app.route('/define/<word>')
def define(word):
    print("word received",word)
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        res=requests.get(url)
        data = res.json()

        if isinstance(data, dict) and data.get("title")=="No Definitions Found":
            return jsonify({"error": "Word not found in dictionary."})

        entry=data[0] 
        meaning = entry['meanings'][0]['definitions'][0]['definition']
        phonetics = [p for p in entry.get("phonetics", []) if p.get("text")]
        audio = ''
        for p in phonetics:
            if p.get('audio'):
                audio = p['audio']
                break
        
        user_lang = app.config.get("user_lang", "hi")
        translation=translate_word(word, app.config["user_lang"])

        return jsonify({
            'word': word,
            'definition': meaning,
            'phonetic':phonetics[0]['text'] if phonetics else '',
            'audio': audio,
            'translation': translation
        })

    except Exception as e:
        print("Error fetching or parsing definition:", e)
        return jsonify({'error': 'Failed to fetch definition'}), 500
    

def translate_word(word, target_lang):
    try:
        url = "https://api.mymemory.translated.net/get"
        params = {
            "q": word,
            "langpair": f"en|{target_lang}"
        }

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()
        translated = data["responseData"]["translatedText"]
        return translated

    except Exception as e:
        print("MyMemory translation failed:", e)
        return "Translation Unvailable"

@app.route('/win')
def win():
    num_guesses = request.args.get('guesses', type=int, default=0)
    return render_template('win.html', guesses=num_guesses)


if __name__ == '__main__':
    app.run(debug=True)