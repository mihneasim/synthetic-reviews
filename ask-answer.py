# imports
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
import os
from dotenv import load_dotenv

load_dotenv()
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# calculate embeddings
# OpenAI's best embeddings as of Apr 2023
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request
openai.api_key = os.environ.get('OPENAI_API_KEY')

# generate embeddings
embeddings = []
reviews_strings = ["The payments menu, including payments between my accounts, has recently stopped working properly. When you enter it, it freezes. To recover, I must go to apps and delete app data  clearing only the cache is ineffective. But it will only work for a short time before the bug takes effect again... Once this is resolved, I will revise my one-star rating.",
                   "The app needs improvement on some functionalities (like preselection of periods of time in reports- I always have to filter and select custom period to see more months), but recently I experienced a lot of app crashes or blocked functions such as transfers between personal accounts and payments. It just freezes. And this happened on a separate phone as well for a different account!",
                   "I had the app since many years now. It had its share of bad days as well. It still crushes now, but to be honest quite rarely. All in all I am happy with this app as it allows me to easily do online payments and it \"feels\" secure enough. In the most recent updates, the new interface - I find it less user friendly and less intuitive than former versions. But I was still able to navigate it and find what I need. It does not excel, but, in my opinion, it's a pretty decent banking up.",
                   "I saw good improvements in the app lately and it's more stable. Improvement points: 1. Bring back the Favorite button, one level up 2. Improve for less use of data, so that when the internet connection is not 4G or more, to still work a bit. 3. Top-ups requests from Revolut don't always work as intended. After confirming in HomeBank, sometimes the information is not sent to Revolut. Thank you and keep up the good work!",
                   "Feature-wise it's a really great app, intuitive and easy to figure out. I like the new Round up functionality. The fingerprint authentication doesn't work on my device, but that doesn't bother me too much. Performance-wise however it's really bad. It has always been slow, but with the last updates it takes 30 seconds just to load any screen after navigation (portfolio, payments etc). So if for example you login, go to payments, then to my payments and open an entry, that's 2 minutes of loading",
                   "please add dark mode, it's to bright according with today standards and it consume more energy.",
                   "Perfect app. It is verry easy to use. Transfers are a piece of cake. It is safe. Customer service has been always excelent. People in the bank go above and beyond to offer u support. Even though some people might complain about some features i do believe is because rhey are not familiar with the app. I learned a lot about the app by calling the customer support or going to the bank directly. For me this is a reliable bank abd app.",
                   "Ing bazar has stopped working, and I used to use it a lot. Now, no matter the update, it doesn't work.",
                   "The app doesn't work anymore.It is closing after loading process. I tried already to delete the cache and reinstall the app but it still useless.",
                   "Almost perfect - when it will have dark mode, it will be perfect!",
                   "I may consider moving to another bank just because of the app appearance. I asked everywhere for a dark mode in this app. There is no option even if my first request was 2 years ago. For 2023 it's very disappointing to be honest. I'm looking forward a reply from the devs."]
for batch_start in range(0, len(reviews_strings), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = reviews_strings[batch_start:batch_end]
    # print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        # double check embeddings are in same order as input
        assert i == be["index"]
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": reviews_strings, "embedding": embeddings})

# search function


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


# examples
strings, relatednesses = strings_ranked_by_relatedness(
    "curling gold medal", df, top_n=5)
# for string, relatedness in zip(strings, relatednesses):
#     print(f"{relatedness=:.3f}")
#     print(string)


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below reviews of a banking app to answer the subsequent. If the answer cannot be found, write "I am not sure."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_review = f'\n\nReview:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_review + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_review
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about a banking app features."},
        {"role": "user", "content": message},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )

    response_message = response["choices"][0]["message"]["content"]

    firstPersonMessage = [
        {"role": "system", "content": "You rephrase text to first person."},
        {"role": "user", "content": f'\n\nReview:\n"""\n{response_message}\n"""'},
    ]

    responseAsFirstPerson = openai.ChatCompletion.create(
        model=model,
        messages=firstPersonMessage,
        temperature=0
    )

    final_response = responseAsFirstPerson["choices"][0]["message"]["content"]

    print(final_response)
    return final_response


ask('What do you think about the design of the app?')
