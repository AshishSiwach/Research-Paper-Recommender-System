# Research-Paper-Recommender-System
Repository of codes and dataset of an AI powered research paper recommender system
<img width="2752" height="1536" alt="unnamed (1)" src="https://github.com/user-attachments/assets/ae0cab56-bb85-4ec9-b0a2-845873a6e278" />


# 🎯 The Research Challenge
The project tackles the significant question: “How can an automated recommendation system, enhanced with tag extraction, improve the discovery and comprehension of relevant research papers?”.
We aim to make recommendations more efficient and provide users with easier access to descriptive tags (keywords) to quickly decide which paper to read.

--------------------------------------------------------------------------------
## ✨ Key Features
• Semantic Retrieval: Recommends research papers based on deep semantic similarity using dense vector representations.

• Context-Aware Keywords: Extracts contextually relevant descriptive tags (keywords) for each recommended paper using KeyBERT, moving beyond simple high-frequency terms.

• GPU Acceleration: Utilizes GPU acceleration with Sentence Transformers, enabling the efficient processing of large datasets (136,238 examples) and eliminating CPU bottlenecks associated with older methods like TF-IDF.

• Robust Data Foundation: Built upon the multidisciplinary ArXiv Scientific Research Papers Dataset.

--------------------------------------------------------------------------------
# 🛠️ Methodology: Two-Part System
The project is logically divided into two primary parts: creating the research paper recommender system and extracting keywords for the resulting recommendations.
## Part 1: Semantic Recommender via Sentence Transformers
This system follows an unsupervised modeling approach.
1. Text Corpus Generation: The text corpus for each paper is created by combining the text fields: Title + Category + Summary.
2. Embedding Generation: A pre-trained Sentence Transformer model, specifically all-MiniLM-L6-v2 (selected for its speed—5 times faster than all-mpnet-base-v2—while maintaining good quality), is employed. This model embeds each paper's combined text into a dense vector representation, capturing rich semantic and contextual information.
3. Similarity Computation: When a user query is provided, it is embedded into the same vector space. Cosine similarity is then computed between the query vector and all document embeddings.
4. Recommendation: The top n documents with the highest similarity scores are recommended.
## Part 2: Keyword Extraction via KeyBERT
KeyBERT is integrated to provide interpretable highlights for the recommended papers.

• KeyBERT leverages BERT embeddings to extract keywords, focusing on the deeper semantic meaning inherent in the text.

• This approach ensures keywords are context-aware and truly represent the document’s core content.

• In the experiments, the top 5 keywords were extracted for each research paper.


--------------------------------------------------------------------------------
# 📊 Evaluation and Performance
The model validation phase implemented an enhanced validation methodology.

• Relevance Proxy: Relevance was established by counting a recommendation as relevant if it shared the same primary category as the query paper.

• Metric: The primary metric used was the Mean Average Precision at 5 (MAP@5), computed as the arithmetic mean of all individual precision scores.

The system demonstrated overall efficiency and relevance:
Metric Result
Value

**Mean Average Precision @ 5 (MAP@5)

0.8400**

--------------------------------------------------------------------------------
# 📦 Data Source
The project utilizes the ArXiv Scientific Research Papers Dataset from Kaggle.

• Size: Approximately 136,000 arXiv research papers.

• Fields Covered: Artificial Intelligence, Machine Learning, Mathematics, Astrophysics, and more.

• Key Features Used: Id, Title, Category, Summary (combined to form the text corpus)
