#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dependencies: numpy, pandas, nltk, gensim

userUserNetwork = W; Weighted adjacency matrix, probability user u respond to user v
threads = R
posts = P : user u, creation t, text x
documents = N
questionTendency = average; number of questions by total posts by user u in thread r for topic k
seeking (question) = S; QuestionTendency * log of 1+posts*length
disseminating (answer) = D; 1-Seeking
dictionary = X
topics = K
postTopics (theta) = [0,1]^N*K
topicWords = [0,1]^K*X
SIDR = phi; proportion of seeking by u on topic k by probability for user v responds to user u on topic k
DISR = psi; proportion of disseminating by u on topic k by probability for user u responds to user v on topic k
Benefit = B; utility obtained by user u for topic k; seeking*log of prob v to u on topic k
alpha = marginal benefit of teaching
smoothing = sigma; (not used here)
c_S, c_D = tightness parameters
step = lambda
t = threshold; error

Compute User-User Network
1-Smooth to ensure user responds to each post at most once
QuestionTendency = proportion of questions per topic per thread per weighted-average Q for u

Seeking and Dissemination
1:Extract forum topics: remove stopwords, urls, stem, lemmatize 
2:Infer if post is question or answer: first post and; other post or; has question mark or 5W1H or 1G
3:Compute S and D

Projected Gradient Descent
C_s = participation rate for seeking
C_d = participation rate for dissemination
alpha = learning step
"""
import json
import os
import sys

import gensim
import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from numpy import linalg as la

nltk.download('wordnet')
stemmer = SnowballStemmer('english')


def social_learning_network_analysis(forum):
    print(f'Collecting threads for {forum}...')
    df = open_forum(forum)
    print(f'Extracting topic model...')
    lda_model, topics = lda(df)
    df.insert(2, 'topics', topics[0])
    print(f'Calculating adjacency matrix...')
    network = get_user_network(df)
    print(f'Extracting forum behaviors...')
    seeking, disseminating = get_forum_behaviors(df)
    print(f'Finding optimal learning network...')
    results = optimize(network, seeking, disseminating)
    print(f'Saving to file...')
    with open(os.path.basename(forum) + '.slna', 'a') as f:
        for idx, topic in lda_model.print_topics(-1):
            f.write(f'Topics: {idx} \nWords: {topic}\n')
        f.write(f'Number of threads: {len(os.listdir(forum))}\n')
        f.write(f'Total posts: {len(df["post_text"])}\n')
        f.write(f'Network dimensions: {network.shape}\n')
        f.write(f'Mean seeking ratio: {np.mean(results["phi"])}\n')
        f.write(f'Mean disseminating ratio: {np.mean(results["psi"])}\n')
        f.write(f'Iterations: {results["i"]}\n')
        f.write(f'Observed learning benefit: {np.mean(results["g_obs"])}\n')
        f.write(f'Optimal learning benefit: {np.mean(results["g"])}\n')
        f.write(f'Observed adjacency matrix first eigenvalue: {np.max(la.eig(results["w_obs"])[0]).real}\n')
        f.write(f'Optimal adjacency matrix first eigenvalue: {np.max(la.eig(results["w"])[0]).real}\n')
    print('Done.')


def open_forum(directory):
    df = pd.DataFrame()
    # Import Coursera course discussion forum files, concatenate threads
    threads = []
    for f in os.listdir(directory):
        thread = json.load(open(os.path.join(directory, f), 'r'))
        posts = pd.DataFrame(thread['posts'])['post_text'].dropna()\
            .transform(lambda x: clean_html(x) if type(x) == str else '')
        comments = pd.DataFrame(thread['comments'])
        threads.append(posts)
        if len(comments) > 0:
            comments['post_text'] = comments['comment_text'].dropna()\
                .transform(lambda x: clean_html(x) if type(x) == str else '')
            threads.append(comments)
        else:
            continue
    df = pd.concat(threads, ignore_index=True, sort=False)
    """Compute new values."""
    df['new_post_time'] = df['post_time'].transform(lambda x: pd.to_datetime(x, unit='s'))
    df['q_a'] = df['post_text'].apply(lambda x: q_a(x) if type(x) == str else np.nan)
    df.sort_values(['thread_id', 'post_time'], inplace=True)
    df['parent_id'] = df['user_id'].shift(-1)
    # df['post_text'] = df['post_text'].transform(lambda x: x if type(str) else "")
    return df


def lda(df):
    processed_docs = df['post_text'].map(lambda x: preprocess(x) if type(x) == str else [])
    dictionary = gensim.corpora.Dictionary(processed_docs)
    # dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    topic_scores = [lda_model.get_document_topics(x) for x in bow_corpus]
    topics = pd.DataFrame([max(x, key=lambda y: y[1]) for x in topic_scores])
    return lda_model, topics


def get_user_network(df):
    user_network = pd.crosstab(df.user_id, df.parent_id)
    idx = user_network.columns.union(user_network.index)
    user_network = user_network.reindex(index=idx, columns=idx, fill_value=0)
    user_network = pd.DataFrame(user_network.sort_index(axis=0).sort_index(axis=1))
    """O index for anonymous posts. Maybe drop.
    user_network.drop(0, axis=0, inplace=True)
    user_topics.columns = user_topics.columns.droplevel(0)
    """
    total_responses = user_network.sum(axis=1)
    total_posts = user_network.sum(axis=0).transpose()
    w_user_network = user_network * total_responses / total_posts
    return w_user_network


def get_forum_behaviors(df):
    user_topics = pd.crosstab(df['user_id'], [df['q_a'], df['topics']])
    post_topics = user_topics.sum(0) / user_topics.sum(0).sum()
    posting_tendency = user_topics['question'] * post_topics.sum() + user_topics['answer'] * post_topics.sum()
    question_tendency = (user_topics['question'] * post_topics.sum()) / posting_tendency.sum()
    disseminating = 1 - question_tendency * np.log(1 + posting_tendency)
    seeking = question_tendency * np.log(1 + posting_tendency)
    return seeking, disseminating


def optimize(w_user_network, seeking, disseminating):
    """Parameters for convex optimization."""
    threshold = .001
    alpha = .4
    beta = 0.8
    c_s = 1.25
    c_d = 0.75
    step = 0.1
    n = w_user_network.shape[0]
    w = w_obs = w_user_network.fillna(0).astype('float64').to_numpy()
    w_prime = np.zeros(w.shape)
    d = disseminating.fillna(0).astype('float64').to_numpy()
    s = seeking.fillna(0).astype('float64').to_numpy()
    z1 = lambda1 = np.zeros(seeking.T.shape).astype('float64')
    z2 = lambda2 = np.zeros(disseminating.shape).astype('float64')
    phi = sidr(s, d, w)
    psi = disr(s, d, w)
    p = (s / (1 + (c_s * phi))).T
    q = (d / (1 + (c_d * psi)))

    g_obs = benefit(w_obs, s, d, alpha).sum().sum()
    g = benefit(w, s, d, alpha).sum().sum()
    g_hat = 1

    np.seterr(all="raise")
    i = 0
    while (g - g_hat) / np.abs(g_hat) >= threshold:
        i += 1
        for u in range(0, w.shape[0]):
            for v in range(0, w.shape[1]):
                w_prime[u, v] = (
                        (d[u].sum() * s[v].sum() / (1 + (w[:, v].sum() * d.sum().sum()))
                         + alpha * d[u].sum() * s[v].sum() / (1 + (w[u, :].sum() * s.sum().sum())))
                        / n
                )
        w_hat = (w + step * w_prime)
        w = proj(grad(w_hat, s, d, p, q, z1, z2, lambda1, lambda2))
        g_hat = g
        g = benefit(w, s, d, alpha).sum().sum()
        z1 = np.clip(np.add(np.add((-1 * np.dot(d.T, w)), p), lambda1), None, 0)
        z2 = np.clip(np.add(np.subtract(np.dot(w, s), q), lambda2), None, 0)
        lambda1 -= np.subtract(np.add(np.dot(d.T, w), p), z1)
        lambda2 += np.subtract(np.subtract(np.dot(w, s), q), z2)
        step *= beta
    return {'w_obs': w_obs, 'w': w, 'i': i, 'phi': phi, 'psi': psi, 'g_obs': g_obs, 'g': g}


def q_a(x):
    q_words = re.compile(r'[\w\W]*(who|what|where|when|why|how|\?)[\w\W]*')
    if re.search(q_words, x):
        return 'question'
    else:
        return 'answer'


def clean_html(x):
    pattern = re.compile(r'[&<].*?[>;]')
    return re.sub(pattern, '', x)


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def sidr(s, d, w):
    phi = np.zeros(s.shape)
    for u in range(0, s.shape[0]):
        for k in range(0, s.shape[1]):
            phi[u, k] = s[u, k] / 1 + (w[:, u].sum() * d[:, k].sum())
    return phi


def disr(s, d, w):
    psi = np.zeros(d.shape)
    for u in range(0, d.shape[0]):
        for k in range(0, d.shape[1]):
            psi[u, k] = d[u, k] / 1 + (w[u, :].sum() * s[:, k].sum())
    return psi


def benefit(x, s, d, alpha):
    b = np.zeros(d.shape)
    for u in range(0, d.shape[0]):
        for k in range(0, d.shape[1]):
            b[u, k] = (s[u, k] * np.log(1 + x[:, u].sum() * d[:, k].sum()) + alpha * d[u, k] * np.log(
                1 + x[u, :].sum() * s[:, k].sum()))
    return b


def proj(x):
    """Projection step"""
    return np.clip(np.subtract(x, np.diag(np.diag(x))), 0, 1)


def grad(x, s, d, p, q, z1, z2, lambda1, lambda2):
    """Proximal gradient step"""
    rho = 1
    return (np.add((rho * np.dot(d, (np.add(np.subtract(np.dot(d.T, x), p), np.subtract(z1, lambda1))))),
                   (rho * np.dot((np.subtract(np.subtract(np.dot(x, s), q), np.add(z2, lambda2))), s.T))))


if __name__ == "__main__":
    directory = sys.argv[1]
    social_learning_network_analysis(directory)
