from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

class BM25RetrieverWithScores(BM25Retriever):
    def get_relevant_documents_with_scores(self, query: str):
        # Preprocess the query
        processed_query = self.preprocess_func(query)
        # Retrieve top documents
        top_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        # Calculate scores for the top documents
        scores = self.vectorizer.get_scores(processed_query)
        # Pair documents with their scores
        doc_score_pairs = [(doc, scores[idx]) for idx, doc in enumerate(self.docs) if doc in top_docs]
        # Sort documents by score in descending order
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return doc_score_pairs

# Sample documents
docs = [
    Document(page_content="5NA.827.851.E Quantum computing is a type of computation that harnesses the collective properties of quantum states."),
    Document(page_content="Machine learning involves the use of algorithms that improve over time with data."),
    Document(page_content="Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales."),
    Document(page_content="Artificial intelligence encompasses machine learning, natural language processing, and robotics."),
]

# Custom BM25 parameters
bm25_params = {
    'k1': 1.5,
    'b': 0.75
}

# Initialize the BM25 retriever with custom parameters
retriever = BM25RetrieverWithScores.from_documents(docs, bm25_params=bm25_params, k=2) # BM25Retriever

# Retrieve relevant documents
query = "What is 5NA.827.851.E?"
results = retriever.get_relevant_documents_with_scores(query) # get_relevant_documents

# Display results
# for i, doc in enumerate(results, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")
    
for i, (doc, score) in enumerate(results, 1):
    print(f"Document {i} (Score: {score}):\n{doc.page_content}\n")





# -----------------------------------

# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings

# # Initialize OpenAI embeddings
# embeddings = OpenAIEmbeddings()

# # Create FAISS vector store from documents
# faiss_vectorstore = FAISS.from_documents(docs, embeddings)

# # Create retriever from FAISS vector store
# faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})



# from langchain.retrievers import EnsembleRetriever

# # Initialize the ensemble retriever with specified weights
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, faiss_retriever],
#     weights=[0.5, 0.5]  # Adjust weights as needed
# )    
    
    





from rank_bm25 import BM25Plus
from nltk.tokenize import word_tokenize

# Sample Corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "A fox is quick and jumps over dogs"
]

# Preprocess: Tokenize the documents
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

# Initialize BM25Plus
bm25_plus = BM25Plus(tokenized_corpus)

# Query
query = "quick fox"
tokenized_query = word_tokenize(query.lower())

# Get scores
scores = bm25_plus.get_scores(tokenized_query)

# Rank documents
ranked_docs = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)

# Print Results
for doc, score in ranked_docs:
    print(f"Document: {doc}\nScore: {score}\n")