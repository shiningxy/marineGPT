from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


app = FastAPI()

class QuestionInput(BaseModel):
    keywords: List[str]


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def generate_question(keywords: List[str]) -> str:
    # Step 1: Tokenize and remove stop words
    tokens = [word_tokenize(keyword.lower()) for keyword in keywords]
    filtered_tokens = [[token for token in token_list if token not in stop_words] for token_list in tokens]

    # Step 2: Lemmatize tokens
    lemmatized_tokens = [[lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in token_list] for token_list in filtered_tokens]

    # Step 3: Generate question
    question = "What is the " + " ".join([" ".join(token_list) for token_list in lemmatized_tokens]) + " in shipping?"
    return question


def get_wordnet_pos(word: str) -> str:
    """Map POS tag to first character used by WordNetLemmatizer."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

@app.get("/test")
async def hello_test():
    return {"text":"hello world!!!"}

@app.post("/generate_question/")
async def generate_question_api(question_input: QuestionInput):
    question = generate_question(question_input.keywords)
    return {"question": question}

