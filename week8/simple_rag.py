from dotenv import load_dotenv

load_dotenv()

import os

from itertools import islice

from datasets import load_dataset
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel


class SimpleRAGNews():

    def __init__(self):

        # load the dataset "permutans/fineweb-bbc-news" in streaming mode,
        # using the subset: "CC-MAIN-2013-20"
        # and the split: "train"

        self.dataset = load_dataset(
            "permutans/fineweb-bbc-news",
            "CC-MAIN-2013-20",
            split="train",
            streaming=True,
        )

        # load the model "ibm-granite/granite-embedding-30m-english"
        # and corresponding tokenizer using AutoModel and AutoTokenizer
        # as per the instructions here (scroll down a bit): 
        # https://huggingface.co/ibm-granite/granite-embedding-30m-english

        self.tokenizer = AutoTokenizer.from_pretrained(
            "ibm-granite/granite-embedding-30m-english"
        )
        self.model = AutoModel.from_pretrained(
            "ibm-granite/granite-embedding-30m-english"
        )
        self.model.eval()

        self.setup_db()

        # finally, create a client to use huggingface inference
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HFTOKEN"), # paste your key in here or load from a variable
        )


    def setup_db(self):
        # TODO

        # take the first 100 samples from the bbc-news dataset
        # pare down the columns and only keep the "text" column
        # then convert to a list: list(ds) for easy retrieval later

        # for each entry in the dataset, call self.embed() on the text
        # save these vectors for later use

        initial_samples = list(islice(self.dataset, 100))
        self.articles = [entry["text"] for entry in initial_samples]
        article_embeddings = [self.embed(text) for text in self.articles]
        self.article_embeddings = torch.stack(article_embeddings)


    def embed(self, text):
        # TODO 

        # given a passed string, tokenize the text using the ibm-granite
        # tokenizer

        # hint: make sure you pass the following to the tokenizer:
        # padding=True, truncation=True, return_tensors='pt'

        # then pass through the embedding model as shown on the ibm-granite page:
        # embedding = model(**tokens)[0][:, 0]
        # embedding = torch.nn.functional.normalize(embedding, dim=1)

        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.model.device) for key, value in tokens.items()}
        with torch.no_grad():
            embedding = self.model(**tokens)[0][:, 0]
            embedding = torch.nn.functional.normalize(embedding, dim=1)
        return embedding.squeeze(0)


    def get_most_relevant_news_article_text(self, user_query):
        # given a user query (string), this method should:
        # call self.embed(user_query)
        # compare the embedding against all stored embeddings using
        # torch.nn.functional.cosine_similarity
        # use the top match to return the text of most relevant article

        query_embedding = self.embed(user_query)
        query_embedding = query_embedding.unsqueeze(0).expand_as(self.article_embeddings)
        similarities = torch.nn.functional.cosine_similarity(self.article_embeddings, query_embedding, dim=1)
        best_match_index = torch.argmax(similarities).item()
        return self.articles[best_match_index]

    def summarize_article(self, article):
        # Use the HF Inference API to ask "openai/gpt-oss-20b" to
        # summarize an article. 

        # This should then return the model's final response text.
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that summarizes news articles for busy readers.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following article:\n\n{article}",
                },
            ],
        )
        return completion.choices[0].message.content

    def summary_for_query(self, query):
        # get the most relevant article
        # get a summary of the article
        # return to user

        # In practice this could be expanded to use the article text in a more
        # complex way.

        article = self.get_most_relevant_news_article_text(query)
        return self.summarize_article(article)



if __name__ == "__main__":

    rag = SimpleRAGNews()
    query = "california wildfires"
    news_blurb_for_user = rag.summary_for_query(query)

    print("An AI-generated summary of the most relevant article:")
    print(news_blurb_for_user)
