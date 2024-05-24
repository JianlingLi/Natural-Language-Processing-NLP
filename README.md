# Natural-Language-Processing-NLP
Abstract
   In this part, we will demonstrate the whole Natural Language Processing by various hands-on practices. From data collection, preparation, and preprocessing to the generation of TF and TF-IDF, further to the text analytics of topic modeling by different algorithms. In addition, the deployment and evaluation of the algorithms in the real-world case, as well as the possible optimized solution and future prospect.
   
   ![image](https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/d6f55d2b-9e97-4241-babc-0896c84a1a13)
   Figure 1 Natural Language Processing pipeline

1.	Introduction
1.1	Dataset
   The given dataset ‘Exam_MB210_NLP.csv’ collects information on over 4800 movies. This information includes movie titles, genres, producer, reviews, and other details. The dataset contains 4803 rows and 20 columns, including 8 numerical columns and 12 textual columns.

1.2	Data Preparation
Overview of the columns 
   We access each column to get an overview of the dataset. For example, the column type and sum of the information. For the numerical columns, we can get an intuitive understanding of some data by visualization, like ‘budget’, ‘runtime’, ‘vote_average’ and ‘vote_count’. For the textual columns in a ‘key: value’ dictionary format, we clean the data by removing the key, and only keep useful value information that makes the data in a concise pattern. For example, column ‘genres’ displays only the type of movie, ‘keywords’ shows only words, only names occur in column ‘production_companies’ and ‘production_countries’.
Correlation
   Find the correlation of all numerical columns by using ‘Pearson’ method to calculate all pair-wise correlations, to detect and remove the features that provide redundant information. In this dataset, both the correlation matrix and visualization show all the numerical columns are in positive correlation, no redundant numerical features. [9]
Text preprocessing 
   To get a full picture of every movie description, we join column ‘tagline’ and ‘overview’ into a new column called ‘Description’. For text in column ‘Description’, a preprocessed () function is created to convert each word to lowercase, to remove the space, and the words that are neither a number nor an alphabet. To achieve this, the function breaks the sentences into a tokens process through lemmatize by importing word_tokenize from NLTK library. In addition, a list of stop words like ‘to’, ‘by’ or ‘after’ etc. which need to be removed by importing stopwords from the NLTK library. Tokenize and lemmatize movie descriptions retain the most important information that makes efficient information memory and extraction, where is the significance of text preprocessing. 

1.3	TF and TF-IDF
   TF stands for terms frequency and IDF stands for inverse document frequency. TF-IDF denotes words scores that represent their importance. With sci-kit learn functions it is possible to automatically generate a vector of term frequency. The vector then includes a form to represent each word from the documents. Each index in this vector represents one word and the value at each index shows how often a word occurs in a document. In this way, a so-called termfrequency vector is created. 
   This vector is easy to implement, but it also has its weaknesses. It is based on the absolute frequencies of occurrence of words. If a word is more frequent in many documents in the corpus, it might overweight other words that are not very frequent but can be important to specify certain categories in the documents. [11] 
   To work around the problem a TF-IDF vector can be created. This vector considers the term frequency with the inverse document frequency. In this method, a word is scaled according to its importance. To do this, the term frequency is multiplied by the inverse term frequency. The higher the score of a word, the more relevant the word is in that document. The TfidfVectorizer from sklearn is used for this purpose. This function converts the preprocessed sentences into a TF-IDF vector.  

2.	Topic modeling
   Topic modeling is designed to extract various topics from a large dataset containing different types of documents to generate a global overview. It involves a statistical model used to discover abstract topics in a series of documents in machine learning and natural language processing, hence it is an unsupervised technique. Intuitively, if there is a central idea in an article, certain words will occur frequently. A topic model can represent the document features in a mathematical framework, it 
