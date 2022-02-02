# SocialHaterBert: a dichotomous approach for automatically detecting hate speech on Twitter through textual analysis and user profiles.

<p align="center">
<i>This work is in progress.</i>
</p>
<p align="center">
  <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f5c5db84-e059-44cb-b51c-660bdb713462/BERT.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220202%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220202T212120Z&X-Amz-Expires=86400&X-Amz-Signature=00063d620e5bb154ae9c254fc154ddfc84637238e952bcf26b5bce26a2e90d3f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22BERT.png%22&x-id=GetObject" alt="drawing" width="400"/>
</p>

This project comprises both an in-depth study of the efforts and techniques used so far for the detection and prevention of hateful content and cyberbullying on the popular social network Twitter, as well as a proposal for a novel approach for feature analysis based on user profiles, related social environment and generated tweets. It has been found that the contribution of user characteristics plays a significant part on gaining a deeper understanding of hate virality in the network, so that this work has allowed to open the field of study and break the textual boundaries giving rise to future research in combined models from a diachronic and dynamic perspective.

<p align="center">
👀 <a href="https://www.notion.so/giogia/SocialHaterBERT-Project-9d93e0e5ed4a468fb09b42a50d9a3dd9">SocialHaterBERT Project webpage</a>
</p>
<p align="center">
✉️ <a href="mailto:glorelvalle@gmail.com">Request datasets</a>
</p>


In an already polarized world, social networks are a double-edged sword with
the appearance of phenomena such as hate speech. In the present work, its presence
on Twitter has been detected and analyzed. For this, a base algorithm,
HaterBERT, has been designed, which improves current Spanish classifiers’ results
by 3%-27%.

Furthermore, the presence of hate speech on Twitter has been analyzed
through an extensive study that has served to extrapolate essential characteristics
of it. To do this, a procedure has been developed for the extraction
and manipulation of these characteristics, SocialGraph, which has been demonstrated
with an F1 of 99 % and a Random Forest classifier that provides valuable
data for the identification of hater profiles.

These findings lead to the development of SocialHaterBERT, a novel multimodal
model that combines categorical and numerical variables from the social
network with text input from tweets, providing not only a new way to understand
hate speech on social media in general but also demonstrating how the
context of social media improves textual classification, which is the most valuable
contribution of this paper. In particular, we achieved a 4% improvement
over the HaterBERT’s base algorithm and a 19% improvement over our original
algorithm, HaterNet (Pereira-Kohatsu et al., 2019).
Future research should look into aspects such as a review of hate’s history
and evolution on the network, trends, public and anonymous users affected
by it, and aggressors’ profiles, with the goal of encouraging the discovery of
relationships with the dissemination and virality of hate on social networks.

Following that, interactions with one another might be investigated, resulting
in an extension of SocialGraph’s characteristics and a prediction of each tweet’s
virality.

_____________________________________________ 💻 _______________________________________________


## Brief Methodology and Design
This section introduces the design of the three approaches created for hate speech on Twitter.

<p align="center">
  <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/98d79e8b-cedf-409d-91a4-f8bb28e4ec98/Captura_de_pantalla_2021-12-03_a_las_17.54.22.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220202%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220202T212307Z&X-Amz-Expires=86400&X-Amz-Signature=bf96b649215b39912200b2cb5b811e8c2bd10b3c34f47c11f61c5dce4fad9c8c&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Captura%2520de%2520pantalla%25202021-12-03%2520a%2520las%252017.54.22.png%22&x-id=GetObject" alt="drawing" width="600"/>
</p>

### [HaterBERT](src/HaterBERT)

This is our base model, based on BERT, for textual hate or no hate classification. The following are the modifications made to the transformer and the tools used to make them:

* Transformers Libraries: [Huggingface 🤗](https://huggingface.co), which has NLP tools and pre-trained transformers: [BERT](https://github.com/google-research/bert) and its Spanish version [BETO](https://github.com/dccuchile/beto).
* BERT Fine Tuning library: [DE-LIMIT](https://github.com/hate-alert/DE-LIMIT).

### [SocialGraph](src/SocialGraph)

To get HaterBERT to feed on the characteristics of the social network, it is first necessary to get all the relative information. To do this, given a dataset *D* consisting of tweets that may or may not contain hate, we first collect:
  1. Information related to each tweet (i.e text, author, number of retweets,
responses, etc). The collected fields can be seen in the table below.

| Attribute         | Type     | Description                           |
|-------------------|----------|---------------------------------------|
| user_id           | int      | user identifier                       |
| screen_name       | str      | username                              |
| tweet_id          | int      | tweet identifier                      |
| tweet_text        | str      | tweet text                            |
| tweet_creation_at | datetime | tweet creation date                   |
| n_favs            | int      | number of favorites                   |
| n_rts             | int      | number of retweets                    |
| is_rt             | boolean  | the tweet is a retweet                |
| rt_id_user        | int      | id of the retweeted user              |
| rt_id_status      | int      | id of the retweeted tweet             |
| rt_text           | str      | text of the retweeted tweet           |
| rt_creation_at    | datetime | creation date of the retweeted tweet  |
| rt_fav_count      | int      | number of favorites (if is retweeted) |
| rt_rt_count       | int      | number of retweets (if is retweeted)  |
| is_reply          | boolean  | the tweet is a response               |
| reply_id_status   | int      | id of the tweet being replied         |
| reply_id_user     | int      | user id to which it responds          |
| is_quote          | boolean  | the tweet is a quote from another     |
| quote_id_status   | int      | id of the quoted tweet                |
| quote_id_user     | int      | id of the quoted user                 |
| quote_text        | str      | text of the quoted tweet              |
| quote_creation_at | datetime | creation date of the quoted tweet     |
| quote_fav_count   | int      | number of favorites quoted            |
| quote_rt_count    | int      | number of retweets quoted             |

  2. Information regarding the users who authored each tweet (i.e. username,
biography, url of the profile image, number of user tweets, number of followers,
etc.). The collected fields can be seen in the table below.
With this we intend to broaden the analysis by modeling the user who has
posted each tweet.

| Attribute             | Type     | Description              |
|-----------------------|----------|--------------------------|
| user_id               | int      | user id                  |
| uname                 | str      | user profile name        |
| virtual               | boolean  | virtual node             |
| screen_name           | str      | username                 |
| description           | str      | biography or description |
| location              | str      | location if any          |
| verified              | boolean  | verified account         |
| profile_image_url     | str      | profile picture url      |
| default_profile       | boolean  | profile update           |
| default_image_profile | boolean  | profile picture update   |
| geo_enabled           | boolean  | real location enabled    |
| created_at            | datetime | account creation date    |
| statuses_count        | int      | number of user tweets    |
| listed_count          | int      | number of lists          |
| followers_count       | int      | number of followers      |
| followees_count       | int      | number of followed       |
| favorites_count       | int      | number of favorites      |

  3. Each user’s last 200 tweets, complemented with the information from point
      1. This allows us to model the types of contributions that each user makes
on a regular basis.

  4. The user profiles mentioned or retweeted by each author in those 200
tweets, so that we can learn about their environment.

All this information is the base on which the attributes of SocialGraph are built.
Below we describe its construction process.

#### Constructing the Graph and Calculating Centrality Measures
Using a Neo4j database we build a graph with three types of nodes:

* User: node that collects all of the user’s information.
* Tweet: node that collects all the information related to tweets.
* Multimedia: node that collects the url referring to the multimedia content or link (to news) that is shared within a tweet.

And three types of links between them: Quoted, Retweeted or Shared.
We then proceed to compute centrality measures in the graph:

| Measure      | Description                                                   |
|--------------|---------------------------------------------------------------|
| betweenness  | computes the shortest path to the graph’s centrality          |
| eigenvector  | measure of a node’s influence on the network                  |
| in-degree    | number of edges pointing to node                              |
| out-degree   | number of edges pointing outside the node                     |
| clustering   | fraction of pairs of neighboring nodes adjacent to each other |
| degree       | number of edges adjacent to the node                          |
| closeness    | average distance of all reachable nodes to node               |

Given that, centrality measures have showed to be effective at quantifying the relative importance of actors in a social network. For example, a node’s ability to influence others is affected much more by its strategic placement within a social network than by the number of followers it has.

#### Summary statistics
We analyze the information downloaded through Twitter’s API and infer a series of new characteristics in order to get a better overall picture of each user.
These characteristics are obtained via:
* **Counting**: In this case, we only perform basic statistical operations on the total number of tweets downloaded per user (e.g., the number of times the user’s tweets are retweeted, the number of bad words per tweet, the average number of tweets per day, the number of hashtags used, the number of
user errors, etc.).

| Attribute                     | Type      | Description                                                      |
|-------------------------------|-----------|------------------------------------------------------------------|
| status_retrieving             | int       | number of saved tweets                                           |
| status_start_day              | datetime  | start date of tweet extraction                                   |
| status_end_day                | datetime  | end date of tweet extraction                                     |
| status_average_tweets_per_day | float     | average tweets per day                                           |
| activity_hourly_X             | int       | number of tweets at each day hour, 24 attributes being X2[00-23] |
| activity_weekly_X             | int       | number of tweets at each week day, 7 attributes being X2[0-6]    |
| rt_count                      | int       | total number of saved tweets                                     |
| geo_enabled_tweet_count       | int       | number of tweets with geolocation enabled                        |
| num_hashtags                  | int       | number of hashtags used                                          |
| num_mentions                  | int       | number of mentions                                               |
| num_urls                      | int       | number of domains shared by the user                             |
| baddies                       | list(str) | bad words or insults used by the user                            |
| n_baddies                     | int       | number of baddies                                                |
| n_baddies_tweet               | float     | number of baddies per tweet                                      |
| len_status                    | float     | average tweet length                                             |
| times_user_quotes             | int       | number of times other users are quoted                           |
| num_rts_to_tweets             | int       | number of times user tweets are retweeted                        |
| num_favs_to_tweets            | int       | number of times user tweets are favorite                         |
| leet_counter                  | int       | number of times the user uses the leet alphabet                  |

* **Clustering**: where we group the analyzed content and extract the most relevant
clusters (i.e top 6 of most shared domains, top 10 of most enabled
places, top 5 of most retweeted users, etc.).

| Attribute              | Type                              | Description                                                |
|------------------------|-----------------------------------|------------------------------------------------------------|
| top_languages          | dict(language(str), account(int)) | top 5 languages most used by the user by number of tweets  |
| top_sources            | dict(vía(str), account(int))      | top 5 ways to tweet by number of tweets                    |
| top_places             | dict(place(str), account(int))    | top 10 places most enabled by the user by number of tweets |
| top_hashtags           | dict(hashtag(str), account(int))  | top 10 hashtags most used by the user by number of tweets  |
| top_retweeted_users    | dict(user(str), account(int))     | top 5 most retweeted users by the user by number of tweets |
| top_mentioned_users    | dict(user(str), account(int))     | top 5 users most mentioned by the user by number of tweets |
| top_referenced_domains | dict(domain(str), account(int))   | top 6 domains most shared by the user by number of tweets  |

* **Modeling**: attributes such as the number of negative, positive, or neutral tweets, the categories to which the image of the user profile belongs, the top 15 topics of each user, and so on are inferred using ad hoc designed classifiers.

| Attribute                                     | Type                                        | Classifier                                   | Source                                                                                                 | Description                                               |
|-----------------------------------------------|---------------------------------------------|----------------------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| categories_profile_image_url                  | dict(dict(category, score, hierarchy=None)) | Client Watson Visual Recognition (IBM)       | [VisualRecognitionV3](https://www.ibm.com/es-es/cloud/watson-studio)                                   | user’s profile image categories                           |
| negatives positives neutral                   | int                                         | Sentiment analysis classifier (transformers) | [finiteautomata/betosentiment-analysis](https://huggingface.co/finiteautomata/beto-sentiment-analysis) | number of negatives number of positives number of neutral |
| negatives_score positives_score neutral_score | float                                       | Sentiment analysis classifier (transformers) | [finiteautomata/betosentiment-analysis](https://huggingface.co/finiteautomata/beto-sentiment-analysis) | negatives score positives scoreneutral score              |
| hate non_hate                                 | int                                         | Ad hoc classifier                            | [HaterBERT](src/HaterBERT)                                                                             | number of hate tweets number of non hate tweets           |
| hate_score non_hate_score                     | float                                       | Ad hoc classifier                            | [HaterBERT](src/HaterBERT)                                                                             | hate score non hate score                                 |
| top_categories                                | dict(category(str), account(int))           | Spanish Category classifier (Python Library) | [subject_classification_spanish](https://pypi.org/project/subject-classification-spanish/)             | top 15 tweet categories                                   |
| misspelling_counter                           | int                                         | Spanish Spell checker                        | [pyspellchecker](https://pypi.org/project/pyspellchecker/)                                             | number of errata committed by the user                    |

#### Transforming and Coding
To be part of the input of any model we must transform the set of characteristics into a set of attributes. Each of the characteristics’ tables indicates the type of variable associated with each characteristic, these
can be grouped into.

<div id="tab:finalatributes">

| **Variable**     | **Original Variable(s)**          | **Group** | **Method**                                | **Categories**                                                                                                                                             | **Description**                                                  |
| :--------------- | :-------------------------------- | :-------- | :---------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------- |
| verified         | NC                                | profile   | boolean classification                    | 0: No, 1: Yes                                                                                                                                              | user is verified                                                 |
| hater            | NC                                | activity  | boolean classification                    | 0: No, 1: Yes                                                                                                                                              | user has more than 5% hate tweets                                |
| vecino\_hater    | NC                                | activity  | boolean classification                    | 0: No, 1: Yes                                                                                                                                              | the user has at least one neighbor with more than 5% hate tweets |
| profile\_changed | default\_profile                  | profile   | boolean classification                    | 0: No, 1: Yes                                                                                                                                              | the user ever updated his profile                                |
| clase\_NER       | screen\_name + uname              | profile   | NER tag search (Spacy)                    | 0: PER, 1: MISC, 2: ORG, 3: UND                                                                                                                            | tipo de nombre                                                   |
| clase\_DESCR     | description                       | profile   | cleaning (NLTK) + Topic Modeling (Gensim) | 0: opinion, 1: studies, 2: politics, 3: activities                                                                                                         | description type                                                 |
| clase\_LOC       | location                          | profile   | cleaning + ad hoc dict + pycountry        | 0-19: geographic world areas or provinces in the case of Spain                                                                                             | geographical area enabled by the user                            |
| clase\_FECHA     | created\_at                       | profile   | division into three regions               | 0: \< 2015, 1: \[2015-2019\], 2: \> 2019                                                                                                                   | time of user creation                                            |
| clase\_IMG       | categories \_profile\_image \_url | profile   | Topic Modeling (Gensim)                   | 0: people, 1: clothing, 2: building, 3: animal, 4: nature, 5: technology, 6: sports, 7: objects, 8: food                                                   | profile image type                                               |
| clase\_HASHTAGS  | top\_hashtags                     | activity  | Correlation matrix + Topic Modeling       | 0: politics, 1: press, 2: sports, 3: others                                                                                                                | hashtag type                                                     |
| clase\_CATS      | top\_categories                   | activity  | Topic Modeling (Gensim)                   | 0: Spain, 1: culture, 2: art, 3: society 4: cartoons, 5: Catalonia, 6: graphical arts, 7: drawings, 8: opinion, 9: illustrations, 10: politics, 11: others | most repeated categories by the user in tweets                   |
| clase\_DOMS      | top\_referenced \_domains         | activity  | wikipedia + Topic Modeling                | 0: social networks, 1: information, communication and news, 2: entertainment                                                                               | type of domain most shared by the user                           |
| clase\_RTSCAT    | top\_retweeted \_users            | activity  | Topic Modeling (Gensim)                   | 0: Spain, 1: culture, 2: art, 3: society 4: cartoons, 5: Catalonia, 6: graphical arts, 7: drawings, 8: opinion, 9: illustrations, 10: politics, 11: others | most retweeted user type                                         |
| clase\_MENCAT    | top\_mentioned \_users            | activity  | Topic Modeling (Gensim)                   | 0: Spain, 1: culture, 2: art, 3: society 4: cartoons, 5: Catalonia, 6: graphical arts, 7: drawings, 8: opinion, 9: illustrations, 10: politics, 11: others | most mentioned user type                                         |
|                  |                                   |           |                                           |                                                                                                                                                            |                                                                  |

Detail of the categorical variables in SocialGraph NC = does not change.
  
</div>
  
  
<div id="tab:finalatributes2">

| **Variable**                                          | **Original Variable(s)** | **Group**  | **Method**      | **Description**                                                     |
| :---------------------------------------------------- | :----------------------- | :--------- | :-------------- | :------------------------------------------------------------------ |
| n\_LESP                                               | top\_languages           | activity   | Ad hoc function | percentage of hate tweets in Spanish                                |
| n\_LENG                                               | top\_languages           | activity   | Ad hoc function | percentage of hate tweets in English                                |
| n\_LOTR                                               | top\_languages           | activity   | Ad hoc function | percentage of hate tweets in other language (no Spanish or English) |
| <span style="color: red">activity\_hourly\_`X`</span> | NC                       | activity   | Ad hoc function | percentage of tweets per hour (X=24)                                |
| <span style="color: red">activity\_weekly\_`X`</span> | NC                       | activity   | Ad hoc function | percentage of tweets per week day (X=7)                             |
| negatives                                             | NC                       | activity   | Ad hoc function | negative connotation percentage of tweets                           |
| positives                                             | NC                       | activity   | Ad hoc function | positive connotation percentage of tweets                           |
| neutral                                               | NC                       | activity   | Ad hoc function | neutral connotation percentage of tweets                            |
| n\_hate                                               | NC                       | activity   | Ad hoc function | hate tweets percentage                                              |
| n\_nohate                                             | NC                       | activity   | Ad hoc function | non hate tweets percentage                                          |
| n\_baddies                                            | NC                       | activity   | Ad hoc function | percentage of baddies per tweet                                     |
| eigenvector                                           | NC                       | centrality | \-              | eigenvector score                                                   |
| in\_degree                                            | NC                       | centrality | \-              | in degree score                                                     |
| out\_degree                                           | NC                       | centrality | \-              | out degree score                                                    |
| degree                                                | NC                       | centrality | \-              | degree score                                                        |
| clustering                                            | NC                       | centrality | \-              | clustering score                                                    |
| closeness                                             | NC                       | centrality | \-              | closeness score                                                     |
| betweenness                                           | NC                       | centrality | StandardScaler  | number of shortest paths to it                                      |
| status\_average\_tweets \_per\_day                    | NC                       | activity   | StandardScaler  | average number of times user tweets per day                         |
| times\_user\_quotes                                   | NC                       | activity   | StandardScaler  | number of times user quotes others                                  |
| negatives\_score                                      | NC                       | activity   | \-              | mean score of negative tweets                                       |
| positives\_score                                      | NC                       | activity   | \-              | mean score of positive tweets                                       |
| neutral\_score                                        | NC                       | activity   | \-              | mean score of neutral tweets                                        |
| hate\_score                                           | NC                       | activity   | \-              | score media de tweets de odio                                       |
| no\_hate\_score                                       | NC                       | activity   | \-              | score media de tweets de no odio                                    |
| statuses\_count                                       | NC                       | activity   | StandardScaler  | total number of tweets                                              |
| followers\_count                                      | NC                       | activity   | StandardScaler  | total number of tweets followers                                    |
| followees\_count                                      | NC                       | activity   | StandardScaler  | total number of tweets followees                                    |
| favorites\_count                                      | NC                       | activity   | StandardScaler  | total number of tweets favourites                                   |
| listed\_count                                         | NC                       | activity   | StandardScaler  | number of lists user is on                                          |
| num\_hashtags                                         | NC                       | activity   | StandardScaler  | number of hashtags used                                             |
| rt\_count                                             | NC                       | activity   | StandardScaler  | total number of retweets                                            |
| num\_mentions                                         | NC                       | activity   | StandardScaler  | number of mentions made                                             |
| num\_urls                                             | NC                       | activity   | StandardScaler  | number of shared urls                                               |
| len\_status                                           | NC                       | activity   | StandardScaler  | average tweet length                                                |
| num\_rts\_to\_tweets                                  | NC                       | activity   | StandardScaler  | number of times user tweets are retweeted                           |
| num\_favs\_to\_tweets                                 | NC                       | activity   | StandardScaler  | number of times user tweets are favourited                          |
| misspelling\_counter                                  | NC                       | activity   | StandardScaler  | number of times user makes mistakes or errors                       |
| leet\_counter                                         | NC                       | activity   | StandardScaler  | number of times user uses leet alphabet                             |
|                                                       |                          |            |                 |                                                                     |

Detail of the numerical variables in SocialGraph NC = does not change.

</div>

### [SocialHaterBERT](src/SocialHaterBERT)

In order to improve on previous algorithms that only used the text of the tweet to be analyzed as input, SocialHaterBERT is created as a multimodal model that combines textual classifiers with social network characteristics. As a result, HaterBERT’s classifier after experimental optimization of its parameters and SocialGraph after an experimental attribute selection form the foundation of SocialHaterBERT.
  
For the construction of the model, we make use of the [Multimodal Transformers](https://multimodal-toolkit.readthedocs.io/en/latest/) library, which is used to incorporate multimodal data on text data for classification and regression tasks. In this way, a pre-trained transformer along the combination module’s parameters and the transformer are trained as a supervised task.

<p align="center">
  <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6be925f5-0a0b-403b-b891-5530c7b9d414/detallemultimodal2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220202%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220202T212347Z&X-Amz-Expires=86400&X-Amz-Signature=fcb9704e4ebe81a27d011b8e408bf060f836452d6a953c61b3dc64474d8340cf&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22detallemultimodal2.png%22&x-id=GetObject" alt="drawing" width="600"/>
</p>
  
SocialHaterBERT’s architecture is as follows: to distribute the data for classification, the text, numeric, categorical and prediction columns are specified in a dictionary. After this, *BertTokenizer* and *BertForSequenceClassification* are instantiated respectively, which also allows the Fine-Tuning of it. Then, in the Combining Module (shown in figure above) a hidden two-layer MLP is created with a ReLu activation function, as it improves training. Finally, before the output layer results are combined using the logical sum of the attributes, as it proved to be the best combination option.
  
_____________________________________________ 💻 _______________________________________________
