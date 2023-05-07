import pandas as pd
import os
import openai
from dotenv import load_dotenv

load_dotenv()

TARGET_TOPIC = os.environ.get('TARGET_TOPIC')
openai.api_key = os.environ.get('OPENAI_API_KEY')

def generate_review(review_title, review_text):
    return """Change the following customer review so that it is about a {}.

Review title: {}
Review description: {}""".format(TARGET_TOPIC, review_title, review_text)


def answer_question(
    model="text-davinci-003",
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer my question
    """
    try:
        # Create a completions using the question
        response = openai.Completion.create(
            prompt=question,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


df = pd.read_csv("dataset.csv", usecols = ['dateUpdated', 'reviews.rating', 'reviews.title', 'reviews.text'])
df['what-to-run'] = df.apply(lambda row: generate_review(row['reviews.title'], row['reviews.text']), axis=1)
df['result'] = df.apply(lambda row: answer_question(question=row['what-to-run']), axis=1)
df.to_csv('processed.csv')
