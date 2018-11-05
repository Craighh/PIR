#! /usr/bin/python
# -*- coding: utf-8 -*-


"""Rank sentences based on cosine similarity and a query."""


from argparse import ArgumentParser
import numpy as np


def get_sentences(file_path):
    """Return a list of sentences from a file."""
    with open(file_path, encoding='utf-8') as hfile:
        return hfile.read().splitlines()


def get_top_k_words(sentences, k):
    """Return the k most frequent words as a list."""
    # TODO
    top_words = []
    wordcount = {}
    for i in range(len(sentences)):
        words = sentences[i].split(' ')
        for j in range(len(words)):
            if words[j] in wordcount:
                wordcount[words[j]] = wordcount.get(words[j]) + 1
            else:
                wordcount[words[j]] = 1
    words = list(wordcount.keys())
    count = list(wordcount.values())
    for i in range(k):
        m = max(count)
        index = count.index(m)
        top_words.append(words[index])
        count.remove(m)
        words.remove(words[index])
    #print(wordcount)
    return top_words


def encode(sentence, vocabulary):
    """Return a vector encoding the sentence."""
    # TODO
    vector = []
    split = sentence.split(' ')
    for i in range(len(vocabulary)):
        count = split.count(vocabulary[i])
        vector.append(count)
    return np.asarray(vector)


def get_top_l_sentences(sentences, query, vocabulary, l):
    """
    For every sentence in "sentences", calculate the similarity to the query.
    Sort the sentences by their similarities to the query.

    Return the top-l most similar sentences as a list of tuples of the form
    (similarity, sentence).
    """
    # TODO
    list = []
    output = []
    for i in range(len(sentences)):
        u = encode(sentences[i], vocabulary)
        v = encode(query, vocabulary)
        #print(v)
        similarity = cosine_sim(u, v)
        tmp = [similarity, sentences[i]]
        if tmp not in list:
            list.append(tmp)
    #print(list)
    list.sort(reverse = True)
    #print('------------------------------------------------------------------------------------------------------')
    #print(list)
    for i in range(l):
        output.append((list[i][0], list[i][1]))
    return output


def cosine_sim(u, v):
    """Return the cosine similarity of u and v."""
    # TODO
    numerator = np.dot(u,v)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    return numerator / (u_norm * v_norm)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('INPUT_FILE', help='An input file containing sentences, one per line')
    arg_parser.add_argument('QUERY', help='The query sentence')
    arg_parser.add_argument('-k', type=int, default=1000,
                            help='How many of the most frequent words to consider')
    arg_parser.add_argument('-l', type=int, default=10, help='How many sentences to return')
    args = arg_parser.parse_args()

    sentences = get_sentences(args.INPUT_FILE)
    top_k_words = get_top_k_words(sentences, args.k)
    query = args.QUERY.lower()

    print('using vocabulary: {}\n'.format(top_k_words))
    print('using query: {}\n'.format(query))

    # suppress numpy's "divide by 0" warning.
    # this is fine since we consider a zero-vector to be dissimilar to other vectors
    with np.errstate(invalid='ignore'):
        result = get_top_l_sentences(sentences, query, top_k_words, args.l)

    print('result:')
    for sim, sentence in result:
        print('{:.5f}\t{}'.format(sim, sentence))


if __name__ == '__main__':
    main()
