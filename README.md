# SocialHaterBert: Detecting hate messages on Twitter: a study based on profiles within the social network.

*This work is in progress*

This project comprises both an in-depth study of the efforts and techniques used so far for the detection and prevention of hateful content and cyberbullying on the popular social network Twitter, as well as a proposal for a novel approach for feature analysis based on user profiles, related social environment and generated tweets. It has been found that the contribution of user characteristics plays a significant part on gaining a deeper understanding of hate virality in the network, so that this work has allowed to open the field of study and break the textual boundaries giving rise to future research in combined models from a diachronic and dynamic perspective.

_____________________________________________ 💻 _______________________________________________

## Methodology and Design
This section introduces the design of the three approaches created for hate speech on Twitter.

### [HaterBERT](src/HaterBERT)

This is our base model, based on BERT, for textual hate or no hate classification. The following are the modifications made to the transformer and the tools used to make them:

* Transformers Libraries: [Huggingface 🤗](https://huggingface.co), which has NLP tools and pre-trained transformers: [BERT](https://github.com/google-research/bert) and its Spanish version [BETO](https://github.com/dccuchile/beto).
* BERT Fine Tuning library: [DE-LIMIT](https://github.com/hate-alert/DE-LIMIT).

## [SocialGraph](src/SocialGraph)

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

_____________________________________________ 💻 _______________________________________________
