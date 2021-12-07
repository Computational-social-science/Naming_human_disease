Ethical Principles of the Nomenclature of Human Diseases
========================================================
Since the present naming protocols could not offer a one-size-fits-all corrective mechanism, many idiomatic but flawed names frequently appear in scientific literature and news outlets at the cost of sociocultural impacts.  To mitigate such impacts, we introduce the ethical principles for the latest naming protocols of human diseases.  Relatedly, we orchestrate rich metadata available to unveil the nosological evolution of anachronistic names and demonstrate the heuristic approaches to curate exclusive substitutes for inopportune nosology based on deep learning models and post-hoc explanations.
# Environment setting
python version : 3.7.0
# 1、Infodemiological study
In the global online news coverage experiments, we aim to unveil the scientific paradigms of the diachronic discourse and emotional tone. Here, the metadata analysis aims to demonstrate the emotional polarity of the public in the context of global online news on German measles, Middle Eastern Respiratory Syndrome, Spanish flu, Hong Kong flu and Huntington's disease over time, respectively.
## Experimental corpus：GDELT Summary
* The textual and visual narratives of different queries
*	65 multilingual online news
*	Machine translate capacity
*	Network image recognition capacity
![image](https://github.com/Computational-social-science/Naming_human_disease/blob/main/Infodemiological%20study/Fig1.svg)
__Figure 1__. The global instant news portfolio on GDELT Summary summarizes the textual and visual narratives of different queries in 65 multilingual online news: **A**, German measles; **B**, Middle Eastern Respiratory Syndrome; **C**, Spanish flu; **D**, Hong Kong flu; and **E**, Huntington's disease. The upper panels display the percent of all global online news coverage over time. The lower panels show the average emotional tone of all news coverage from extremely negative to extremely positive. The temporal resolution of sampling is 15 minutes per day.
# 2、Historiographical study
For a large number of cases of unsuccessful naming of diseases, we choose *German measles* as an example. By searching the frequency of use of *German measles* and its synonyms in historical development, we can initially understand the nature of its historical evolution.
## Experimental corpus：Google Books Ngram Corpus
*	n-grams from approximately 8 million books
*	6% of all books published in Eight languages
	- English
	- Hebrew
	- French
	- German
	- Spanish
	- Russian
	- Italian
	- Chinese
*	Book data logs from 1500 to 2019
![image](https://github.com/YaChen8/Naming_human_disease/blob/main/Historiographical%20study/Figure%203.jpg)
__Figure 2__. Google Books Ngram Corpus (GBNC) facsimiles the diachronic discourse of *morbilli* (English corpus), *rubeola* (English corpus), *rubella* (English corpus), *Rötheln* (German corpus), and *German measles* (English corpus) from 1719 to 2019.
# 3、Semantic similarity experiments
Based on the epistemic results of the above historiographical study, as an exemplificative case, we could construct the initial candidates of *German measles*, which includes *morbilli*, *rubeola*, *rubella*, and *rötheln*. Relatedly, as a prior knowledge, the term rotheln is ordinarily used as a translation of the German term *rötheln* in literature. From the outset, it’s reasonable to expand the initial candidates to *morbilli*, *rubeola*, *rubella*, *rötheln*, and *rotheln*.
Directed at five expanded candidate words, we employed the BERT model and PubMedBERT model to quantify the semantic similarities between them, respectively.
## Experimental corpus
*	BERT [ the BookCorpus (800M words) and English Wikipedia (2,500M words) ]
*	PubMedBERT [ PubMed abstracts (14M abstracts, 3.2B words, 21GB) ]
![image](https://github.com/YaChen8/Naming_human_disease/blob/main/Semantic%20similarity%20experiments/Figure%205.jpg)
__Figure 3__. Heatmaps of semantic similarity scores. **A**, the BERT model and **B**, the PubMedBERT model. The higher scores in the heatmaps, the higher semantic similarity between two synonyms. 
# 4、Semantic drift experiments
To accurately demonstrate the semantic evolution of each keyword, we analyzed the dynamic evolution of the five keywords *German measles*, *morbilli*, *rubeola*, *rubella*, and *rötheln*.
## Experimental corpus：Google Books Ngram Corpus
![image](https://github.com/YaChen8/Naming_human_disease/blob/main/Semantic%20drift%20experiments/Figure%206.jpg)
__Figure 4__. The historical reconceptualization process of *German measles* and common synonyms. 
# Citing
Code for our paper " Ethical Principles of the Nomenclature of Human Diseases". Please cite our paper if you find this repository helpful in your research.
