# Natural-Language-Processing-NLP
Abstract
   In this part, we will demonstrate the whole Natural Language Processing by various hands-on practices. From data collection, preparation, and preprocessing to the generation of TF and TF-IDF, further to the text analytics of topic modeling by different algorithms. In addition, the deployment and evaluation of the algorithms in the real-world case, as well as the possible optimized solution and future prospect.
   
   ![image](https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/d6f55d2b-9e97-4241-babc-0896c84a1a13)
   						
   Figure 1 Natural Language Processing pipeline

**1.	Introduction**

**1.1	Dataset**

   The given dataset ‘Exam_MB210_NLP.csv’ collects information on over 4800 movies. This information includes movie titles, genres, producer, reviews, and other details. The dataset contains 4803 rows and 20 columns, including 8 numerical columns and 12 textual columns.

**1.2	Data Preparation**

**Overview of the columns**

   We access each column to get an overview of the dataset. For example, the column type and sum of the information. For the numerical columns, we can get an intuitive understanding of some data by visualization, like ‘budget’, ‘runtime’, ‘vote_average’ and ‘vote_count’. For the textual columns in a ‘key: value’ dictionary format, we clean the data by removing the key, and only keep useful value information that makes the data in a concise pattern. For example, column ‘genres’ displays only the type of movie, ‘keywords’ shows only words, only names occur in column ‘production_companies’ and ‘production_countries’.

**Correlation**

   Find the correlation of all numerical columns by using ‘Pearson’ method to calculate all pair-wise correlations, to detect and remove the features that provide redundant information. In this dataset, both the correlation matrix and visualization show all the numerical columns are in positive correlation, no redundant numerical features. [9]

**Text preprocessing**

   To get a full picture of every movie description, we join column ‘tagline’ and ‘overview’ into a new column called ‘Description’. For text in column ‘Description’, a preprocessed () function is created to convert each word to lowercase, to remove the space, and the words that are neither a number nor an alphabet. To achieve this, the function breaks the sentences into a tokens process through lemmatize by importing word_tokenize from NLTK library. In addition, a list of stop words like ‘to’, ‘by’ or ‘after’ etc. which need to be removed by importing stopwords from the NLTK library. Tokenize and lemmatize movie descriptions retain the most important information that makes efficient information memory and extraction, where is the significance of text preprocessing. 

**1.3	TF and TF-IDF**

   TF stands for terms frequency and IDF stands for inverse document frequency. TF-IDF denotes words scores that represent their importance. With sci-kit learn functions it is possible to automatically generate a vector of term frequency. The vector then includes a form to represent each word from the documents. Each index in this vector represents one word and the value at each index shows how often a word occurs in a document. In this way, a so-called termfrequency vector is created. 
   This vector is easy to implement, but it also has its weaknesses. It is based on the absolute frequencies of occurrence of words. If a word is more frequent in many documents in the corpus, it might overweight other words that are not very frequent but can be important to specify certain categories in the documents. [11] 
   To work around the problem a TF-IDF vector can be created. This vector considers the term frequency with the inverse document frequency. In this method, a word is scaled according to its importance. To do this, the term frequency is multiplied by the inverse term frequency. The higher the score of a word, the more relevant the word is in that document. The TfidfVectorizer from sklearn is used for this purpose. This function converts the preprocessed sentences into a TF-IDF vector.  


**2.	Topic modeling**

   Topic modeling is designed to extract various topics from a large dataset containing different types of documents to generate a global overview. It involves a statistical model used to discover abstract topics in a series of documents in machine learning and natural language processing, hence it is an unsupervised technique. Intuitively, if there is a central idea in an article, certain words will occur frequently. A topic model can represent the document features in a mathematical framework, it automatically analyzes each document or sentence, and determines cluster words for a set of documents. Now we attempt to utilize different topic modeling algorithms to analysis the given dataset and display visualization. [8] [11]


**2.1	Latent Dirichlet Allocation (LDiA)**

   LDiA is based on the use of words in a document and on the definition of words for each topic. To find topics for each document and words for each topic, LDiA uses the Dirichlet distribution. An important rule is that LDiA assumes that documents with similar topics use similar words, which results in finding topics by searching for groups of words that frequently occur together in different documents. To use LDiA, a total set of topics must be manually specified. LDiA then iterates through each document in the corpus and randomly assigns each word in a document to a topic, which provides the topic representation of all documents with the word distribution of all documents and the word distribution of all topics. In addition, it iterates over each word in a document to improve the model. A simplified formula is to calculate the proportion of words in a document that are currently assigned to a topic. [2] 
   
P(Topic T ┤|Document D)

The proportion of assignments to topic T out of all documents originating from this word W.

P(Word W | Topic T)

Word w is going to be reassigned to a new topic, where topic is chosen with the probability:

P(Topic T | Document D)*P(Word W | Topic T)

This step is repeated several times until a steady state is reached. [2] The model can determine which words are most important for each topic and which topic is most fits to a document. From this state, the primary goal is to understand the results of the model and interpret the different topics.
   In this paper, four different LDiA models were created. This is to see the difference between using TF and TF-IDF in a LDiA model and to see different types of TF and TF-IDF representation by limiting the maximum features in the documents to the value of 1000. With this limitation the TF and TF-IDF vectors will be smaller only considering the 1000 most valuable words. In addition, each vector is bounded by the limiter max_df=0.9, which ignores any word that has a document frequency higher than 0.9.  This is done because words that occur very often and frequently could be unimportant words like stop words or similar words. The vectors are also limited with a cut off min_df=2. This ignores words that occur in less than two documents since they might not be as relevant as others. 
   LDiA uses the theories described above to assign words to topics and topics to individual documents. After fitting the model to the vectors, it is possible to store a matrix of the LDiA model as a variable. This matrix has the size (number of documents x topics). The matrix shows the probabilities of which topic best matches each document. The plot_top_words() function gives a good overview of each topic by plotting the most important words. In many literatures, the LDiA model is preferably done with the TF vectors but also with TF-IDF vectors. When we compare the LDiA models with TF and TF-IDF, the words of each topic are different. The LDiA model adapted to the TF vector without word limit uses same words in several topics. Words like new, family, young, life, love, world, family, or man are heavily weighted in many subjects. This is probably because such words often occur together in many documents, even if they describe different topics. It is also not easy to see what overall term each topic is supposed to have. The words in each topic seem to be random. Comparing the LDiA model adapted to the TF vector with word limit, some of the same words have high values in some topics, but not as many as in the LDiA without word limit. Also, it is easier to see which generic term could match the high-scoring words in each topic. Topic 1 (real, fbi, man, face, stop, evil, help, killer, agent & world) can be assumed to be an action movie with. Topic 9 (begin, daughter mother, son, father, story, young, love, family, life) probably includes movies from the romance or family genres as well as comedy. 
   Looking at the LDiA-TF-IDF model without a limited number of features, the words of the topics have changed, but the same problems still occur. The subjects are not as easy to interpret compared to the LDiA-TF-IDF model with word limit. Furthermore, city names appear in the TF-IDF-LDiA models. For example, in Topic 3, the words los and angeles are among the most scored words, and new and york also occur together. The LDiA models seem to work better when a word limit is given. Through the word limit, the focus is only on the most important words and their relationships to other words.
The other visualization of each LDiA model shows each topic in a circle. The size of the circles defines how many documents belong to a topic and thus the importance of each topic.

![image](https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/d7999a0e-586c-4ed2-912b-033246396420)
					
Figure 2 LDA Topic visualization with TF vector (limited features)

   Looking at the LDiA models fitted to the TF vectors, the circles have almost the same size. This means that each topic contains almost the same number of documents. Visualization of the LDiA model fitted to the TF-IDF vectors shows that some topics are more important than others due to the size difference of the circles. Also, some topics are closer together while others are far apart. Topics that are close to each other have similar words with high weight where topics that use different words are far away. Comparing the TF-LDiA models with the TF-IDF-LDiA models, the TF-IDF-LDiA models have some topics that are very similar, so they overlap and the TF-LDiA models are better distributed. If the circles overlap, this could indicate that too many topics have been selected. Additionally, the slider on the side of the visualization helps to get a better insight into each topic. The slider is used to adjust the relevance of the displayed words in the histogram. When the slider is at one, the most relevant words of the topic are displayed. As the slider moves down, the words displayed change. Moving the slider displays words that are less relevant. Displaying words that are less relevant can help in understanding the topic. 

   ![image](https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/9a16155f-f3e8-48cb-8f0e-37ada5ad4680)
			
   Figure 3 LDA Topic visualization with TF-IDF vector (limited features)


**2.2	BERTopic Modeling**

   BERTopic leverages embedding models and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions. [12]  
The BERTopic algorithm has three steps: [3]
• Embed the documents. The algorithm extracts document embeddings with BERT or any other embedding technique.
• Cluster documents. When reducing the dimensionality of embeddings, it uses UMAP and to cluster the reduced embeddings and creating the clusters of semantically similar documents, it uses HDBSCAN.
• Create a topic representation. The final step is to extract and reduce topics with class-based TF-IDF and then improve the coherence of words with Maximal Marginal Relevance. 
   After training the model it is possible to visualize plots. Function topic_model.visualize_documents() is applied to retrieve a detailed visualization for the documents inside the topics. Figure 4 is the final visualization of the movie description, which clearly displays the clustering of 53 topics in a distribution map. The topics that are similar to each other has an overlapping area or the distance between the topics are shorter. If there is a larger distance gap between the topics, which means the topics are distinct. The function recalculates the document embeddings and reduces them to 2-dimensional space for easier visualization purposes. In the plot the keyword for each topic is written beside the topic number and it is assigned appropriately when cross-examined with the intertopic distance map. It was demonstrated by disabling the option to hover over each of the points and see the content of the documents. If all the documents in the visualization are saved, the process could be complicated and results in large files. This option could be changed by setting the ‘hide_document_hover = False’ and enable the contents when hovering over the individual points. [6]

 ![image](https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/551770ae-fb00-4c04-a2d7-7f11caf61f1d)

  Figure 4


**2.3	Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) **

   The given dataset is a high-dimensional dataset with different features in both numerical and text format. By using LDA model in sklearn.discriminant_analysis.LinearDiscriminant- Analysis, it breaks down our dataset into only one topic to achieve the effect of dimensionality reduction. LDA is a supervised method that requires labeled data. Hence, we manually label every movie description with Liked=1 and dislike=0 by computing the mean value of ‘vote_average’ in the dataset. Movie descriptions are voted over the vote_average mean value will be labeled as ‘Liked’ (1), in contrast will be labeled as ‘dislike’ (0). In our case, the LDA model is used to predict the movie voting and can be compared with the initial voting and trains the accuracy of LDA score. [10] 
   PCA is a method of statistical analysis and simplification of datasets, which can be utilized to identify patterns in data and express the data in such a way as to highlight their similarities. In this task, we use PCA for dimension reduction by transforming the TF-IDF of column ‘Description’ to fit in PCA model and visualize the distribution of 4803 movies’ vote score in the dataset.[13]
   
![image](https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/fc64eeb2-cbcc-44bc-ba27-8091ca70c703)
					
Figure 5 Visualization Error without label data
 
   Figure 5 displays a visualization error due to unlabeled data, this is because we did not reduce the dimension of column ‘vote_average’ and label the data. Hence, we cannot view the distribution clearly. Figure 6 demonstrates the final PCA visualization result. The visualization indicates various movie topics with high voting score falls on the red scatter, while those with low voting score falls on the green scatter. From the perspective of distribution, the closer the scatter distance, the more similar topic of the movies. Thereby, PCA enables us to see the similarity between samples very intuitively. However, we discover the audience vote score can exist a big gap even for movies with similar themes. If we look at the position of the center coordinates about (0,0), both red and green scatters are orthogonal. In addition to those movies with high similarity, other scatters fall on outlier movies also exist different voting scores, even if they have low similarity.

 ![image](https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/1c9a1ed7-5c86-4f0e-94c7-8e5083cf0a7e)

  Figure 6 Movie voting score distribution

**2.4 Summary**

   The advantage of LDA is a supervised method that can improve the predictive performance of the extracted features most of the time. PCA is an unsupervised algorithm. Patterns in data can be difficult to find under high-dimensional dataset, where graphical representation is not available. In our task, PCA is a powerful tool by reducing the number of dimensions, without losing much information. Both LDA and PCA can be used for dimensional reduction. However, no matter LDA or PCA, the new features are not easily interpretable, we must manually set or tune the number of components to keep. In addition, LDA requires labeled data that makes it more situational, BERTopic follows dimensional reduction as well. Compared to other techniques, it works exceptionally with pretrained embeddings due to a split between clustering the documents and using c-TF-IDF to extract topic representations.[4][14] Although BERTopic has advantages over other methods, if you do not have a GPU, embedding documents can take too much time compared to LDiA. [5] Comparing both results, for LDiA the number of topics needs to be specified beforehand where BERTopic doesn’t need it. Also, BERTopic assigns each document to only one topic where LDiA returns a matrix where each document is assumed to be a mixture of topics. [1] LDiA is better to analyze because each topic can be accessed, and the most important words can be visualized. Through automatically assigning the number of topics the BERTopic model considers more topics than the LDiA. Both models work fine. It is hard to compare the results because the models use completely different approaches. 


**3.	Searching for similar movies**

**3.1 Background**

   In this task, we will find similar movies as ‘Spider-Man’ from the give dataset. What we can do to solve the problem is to create a search engine program and obtain a recommendation movie list. Hereby, we would like to introduce a few solutions by using related algorithms. 

**3.2 Cosine Similarity** 

   To find similar movies based on the dataset, it is necessary to consider what inputs are required and what outputs are expected. Our recommendation program should get information based on the entered title in this example ‘Spider-Man’. With the title every column of the dataset can be entered. This enables the comparison of different columns of movies in the dataset. Cosine similarity measures the similarity between two vectors and gives the cosine of the angle between the vectors of two documents in their inner product space.[7][11] Based on the cosine angle of two vectors and the resulting similarity value, which is measured from   -1 to 1, it can be determined whether documents are similar or not. If the similarity value is close to 1, it means that the vectors are close to each other and the angle between the two documents is close to zero degrees (cos0°). If the vectors have a similarity value close to 0, it means that the documents are not similar and form an orthogonal angle between them (cos90°). The last option is a similarity value close to -1, which means that the terms are completely different, and the distribution is completely opposite to each other (cos180°). Figure 7 shows each scenario for better understanding. [11] 

![image](https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/a48b504e-db70-4095-b1f5-bf74730d6c79)

Figure 7 Cosine Similarity

   The cosine similarity is usually compared with the TF-IDF vectors. Comparing TF vectors with cosine similarity results in the similarity score from range 0 to 1. The similarity score can be calculated using the following formula, where u_i and v_i represent the features of the vectors and n is the total number of features.

   <img width="256" alt="image" src="https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/df4e452c-5433-4d58-9c68-959b58c6af56">

To obtain the cosine distance, the same formula can be used to subtract from 1. The result then shows the distance between the two documents.

   <img width="231" alt="image" src="https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/c7150d38-126e-49ec-8b07-36fa8d95e376">

   In this project, two possible functions were created. The functions use cosine similarity to find the best recommendations. With the input of a title the functions get the description of the movie from the dataset. Based on the description, it is possible to find the cosine similarity. The result is the cosine similarity value for each movie in the dataset. The value tells whether a movie is similar or not. The output of both functions provides different movies that can be treated as recommendations based on the similarity of the movie descriptions. This time, the TF-IDF vector without word limitation is used to find similarities. Using this vector, the result is the best.
   The result of the first and simpler recommendation function shows the most similar movies based on the description of each movie compared to the movie ‘Spider-Man’. As can be seen in figure 8, movies like ‘The Amazing Spider-Man 2’, ‘Spider-Man 3’ or ‘The amazing ‘Spider-Man’ are movies that are similar to ‘Spider-Man’, which makes sense. However, movies like ‘The Reef’ or ‘Arachnophobia’ also have a high similarity score to ‘Spider-Man’. To understand the details of this result, similar words of each description are printed in the code.

![image](https://github.com/JianlingLi/Natural-Language-Processing-NLP/assets/122969933/3fb9dd2d-9aa6-406e-997f-3778bac98e02)
Figure 8 Movie recommendation for Spider-Man
   
   For example, ‘The Amazing Spider-Man 2’ has eight words in common with ‘Spider-Man’ (peter, parker, high, school, come, peter, peter, peter). Compared to the second similar movie, that's a lot of words. The Reef has only two words in common with ‘Spider-Man’ (great, great). The third recommendation ‘Spider-Man 3’ also sounds more like ‘Spider-Man’ than ‘The Reef’ at first. This movie also has three words in common with ‘Spider-Man’ (altered, peter, parker). But why does this movie have a lower similarity score than ‘The Reef’ Looking at the TF-IDF vector of ‘Spider-Man’, some words seem to be more important than others based on TF-IDF weights. In this scenario, the word great needs to be analyzed, as well as the words altered, peter, and parker. Looking at the TF-IDF vector of ‘Spider-Man’ and ‘The Reef’, the word great has the highest IDF value in both vectors. Comparing the vectors of ‘Spider-Man’ and ‘Spider-Man 3’, the words in both vectors are less relevant than great and thus ‘The Reef’ has a higher similarity value than ‘Spider-Man 3’. 
   This feature is good, but it can be improved. Based on the dataset, the given features can be used in a recommendation system to find only similar movies and get only good recommendations. Based on data understanding and preprocessing, features like vote_count, vote_average and genres can be considered to get better results. The improved_recommendations() function uses these features to filter out recommendations that may not be good enough. The basis of this function is the same as the other functions and is thus based on the cosine similarity score. Additionally, movies that do not meet certain criteria are filtered out. In this case, a list of the 30 most similar movies is created. 
   For the vote_average feature, the mean value of the 30 most similar films is the first criterion. Any movie whose vote_average is below this value is filtered out. The same is done with the feature vote_count. For this feature, the criterion is that the movies should have a score higher than the quantile value of 0.60. These two features ensure that the recommended movies have the same range of number of votes and the rating is high enough of the movie. The last criterion to filter out movies and make the recommendations more accurate is to consider the genres of each movie. The recommended movies should contain at least one of the genres of the searched movie. In the example of searching for similar movies of ‘Spider-Man’, the recommended movies should belong to either the Fantasy or Action genre. As you can see in the code, the result is slightly different compared to the other function. ‘The Reef’ is no longer a recommendation because the genres of this movie are drama, thriller, and horror. ‘Spider-Man 3’ is also no longer a recommended movie. This is because the vote_count of this movie is 3576 and should be higher than 3874.8 and thus the movie cannot be validated in this example.

	Latent Semantic Indexing (LSI)
   The reason to utilize LSI [11] is that an effective search engine needs to handle semantics probably and can be able to match and retrieve at the semantic level. And LSI can achieve this by applying SVD on the TF-IDF matrix. Our solutions as following:
• Utilized the TF-IDF from sklearn to construct an initial search engine.
• Create a train function and build up an SVD model. The imported TF-IDF matrix will be decomposed by SVD. Hence, LSI can be achieved by applying SVD on the TF-IDF matrix.
• Create a search function to transfer the input query movie name to vector and calculate the cosine similarity that matches the index with similar semantic movie title. Return the index and the movie title with similar semantic level.
• Input: the query movie title outside of the functions.
• Output: retrieve the initial movie title from the index mapping. 
   From the output in figure 12, we can see that all the retrieved movie titles are very close to ‘Spider-Man’ on semantic level. Not only the query movie itself, but also its sequel and moreover the other related series as well. In the real world, this latent semantic based search makes the selection that is closest to the end user’s wishes, where its commercial value lies. 

	**Conclusion and future perspective **
   In NLP part, we used different techniques to solve various tasks. For the numerical data, we attempted to use machine learning skill ‘Pearson’ method to discover all the numerical columns are in positive correlations. For the textual data, we applied NLP skill to extract the information. In topic modeling, we utilized various algorithms which cover both supervised and unsupervised learning. LDiA is used to discover the relevant and less relevant topic. BERTopic is applied on the topic clustering. In addition, we also introduced the usage of dimension reduction algorithm PCA and LDA on text analysis. We had a challenge to load the full dataset after PCA was executed. This is probably because the matrix dimension was modified and some of the feature’s data were extracted, which affects the integrity of the dataset. Moreover, we built up a mini search engine by cosine similarity and LSI to find similar movie recommendations. In the future, the search engine can be a huge potential market. The traditional search engines only provide content that exists on the internet. In the future, there might be a new generation engine that can provide answers that have never appeared before.

 
**Bibliography**
[1] Albanese, N.C. (2022). Topic Modeling with LSA, pLSA, 
LDA, NMF, BERTopic, Top2Vec: a Comparison. [online] Medium. Available at: https://towardsdatascience.com/topic-modeling-with-lsa-plsa-lda-nmf-bertopic-top2vec-a-comparison-5e6ce4b1e4a5#1825.
[2] Bansal, H. (2020). Latent Dirichlet Allocation. [online]
Medium. Available at: https://medium.com/analytics-vidhya/latent-dirichelt-allocation-1ec8729589d4.
[3] David, D. (n.d.). NLP Tutorial: Topic Modeling in Python
with BerTopic. [online] HackerNoon. Available at: https://hackernoon.com/nlp-tutorial-topic-modeling-in-python-with-bertopic-372w35l9.
[4] Dr. Egger, Roman & Yu, Joanne. (2022). A Topic
Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic to Demystify Twitter Posts. Frontiers in Sociology. 7. 10.3389/fsoc.2022.886498.
[5] GitHub. (n.d.). Why BERTopic instead of LDA, NMF...
[online] Available at: https://github.com/MaartenGr/BERTopic/issues/486. 
[6] Grootendorst, M. (n.d.). Topic Visualization – 
BERTopic. [online] maartengr.github.io. Available at:https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#visualize-documents.
[7] Han, J. and Pei, J. (2012). Cosine Similarity – an
overview | ScienceDirect Topics. [online] 
ScienceDirect. Available at: https://www.sciencedirect.com/topics/computer-science/cosine-similarity.
[8] Kulshrestha, R. (2019). A Beginner’s Guide to Latent
Dirichlet Allocation(LDA). [online] Medium. Available at: https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2.
[9] Nettleton, D. (2014). Pearson Correlation. [online] 
Science Direct. Available at: https://www.sciencedirect.com/topics/computer-science/pearson-correlation. 
[10] Ren, H. (2022), PGR210 – Natural Language Processing 
Part, Lecture12, lec12.pdf, 2.1 Distance
[11] Sarkar, D. (2016). Text Analytics with Python: A Practical
Real-World Approach to Gaining Actionable Insights from your Data. Page. 181,234, 241-242, 281, 283

[12] spaCy. (n.d.). BERTopic. [online] Available at:
https://spacy.io/universe/project/bertopic.
[13] Smith, L. I. (2002). A tutorial on Principal Components
Analysis. Chapter 3, Page12
[14] Sánchez-Franco, M. J., and Rey-Moreno, M. (2021). Do
travelers’ reviews depend on the destination? An analysis in coastal and urban peer-to-peer lodgings. Psychol. Market. 39, 441–459. doi: 10.1002/mar.21608

