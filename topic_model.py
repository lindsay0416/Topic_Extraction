import nltk
from lxml import etree
import os
import os.path
import re
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models import CoherenceModel
import datetime
from numpy import *
import matplotlib.pyplot as plt


def readByfilter(path):
    # read XML files by filter (5 types of data)
    # From the name of each file split the name with "."
    alllist, maleslist, femalelist, ageunder20, ageover20 = [], [], [], [], []
    for root, dirs, files in os.walk(path):
        for file in files:
            if (file.split(".")[-1] == 'xml'):
                alllist.append(file)
                # index = 1 is the gender information
                if (file.split(".")[1] == 'male'):
                    maleslist.append(file)
                else:
                    femalelist.append(file)
                    # index = 2 is the age information
                if (file.split(".")[2] <= str(20)):
                    ageunder20.append(file)
                else:
                    ageover20.append(file)
    return alllist, maleslist, femalelist, ageunder20, ageover20


def stopwords_remove(data):
    stopwords = []
    # online resources stopwords file, 892 words
    with open('./stopwords.txt', 'r') as st:
        for word in st:
            stopwords.append(word.split('\n')[0])
    # manually set some stopwords, those words are not related to the topics
    # Also they are not in the stopwords documnets
    manually_stop = ['night', 'test', 'time', 'day', 'life', \
                     'people', 'urllink', 'haha']
    for k in manually_stop:
        stopwords.append(k)

    results = [word for word in data if word[0] not in stopwords]
    return results


def lemmatize_document(res_low):
    # eg. 'has'-> 'have', 'countries' -> 'country'
    word_wnl = WordNetLemmatizer()

    res_lem = []
    for word, pos_tag in res_low:  # only handle verb/nouns
        if pos_tag.startswith('NN'):
            t_word = word_wnl.lemmatize(word, pos='n')
        elif pos_tag.startswith('VB'):
            t_word = word_wnl.lemmatize(word, pos='v')
        else:
            t_word = word

        res_lem.append((t_word, pos_tag))
    return res_lem


def preprocess(path, xml_file):  # xml_list is files of each type
    dateInfo = []
    yearlist = []

    rawData = ''
    for file in xml_file:
        # ASCII file Unicode
        f = open(path + "/" + file, 'r', encoding='ISO-8859-1')
        str_text = f.read()
        # recover from bad characters.
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(str_text, parser=parser)
        # Extract text in each "post" element in xml file
        for sub_node in root:
            if sub_node.tag == 'post':
                # Combine all the text
                rawData = rawData + sub_node.text
            if sub_node.tag == 'date':
                date = sub_node.text
                dateInfo.append(date)
                year = date[-1]
                yearlist.append(year)

    # divide the sentence into words
    tokens = nltk.word_tokenize(rawData)

    # pos tagging
    tagwords = nltk.pos_tag(tokens)

    # only keep letter and number
    word_reg = "[^A-Za-z0-9]"

    # removing punctuation,
    res_punc = [(re.sub(word_reg, '', kt[0]), kt[1]) for kt in tagwords]

    # removing  length of word < 3 and lower the word
    res_low = [(kt[0].lower(), kt[1]) for kt in res_punc if
               len(kt[0]) >= 3]

    res_lem = lemmatize_document(res_low)  # lemmatization

    res_stop = stopwords_remove(res_lem)  # remove stop words

    return res_stop


# get all documents of each person,
# prepare for TFIDF
def get_documents_person(file_list):
    person_dict = {}
    for file in file_list:
        person_name = file.split('.')[-2]
        if person_dict.get(person_name, -1) == -1:
            person_dict[person_name] = [file]
        else:
            person_dict[person_name].append(file)
    return person_dict


def extract_feature(path, type_list, one_person=False):
    vnMatrix = []  # all documents for this type
    if one_person == False:
        for t_l in type_list:
            res_stop = preprocess(path, [t_l])

            vn = []  # verb/noun
            for item in res_stop:
                # find all the nouns and verb in the article and save as list
                if item[1] in ["NNP", "NN", "NNS", "NNPS", "VB", "VBD",
                               "VBG", "VBN", "VBP", "VBZ"]:
                    vn.append(item[0])

            vnMatrix.append(vn)

    # the blogs from one person as a document
    else:
        # one person as a document
        person_dict = get_documents_person(type_list)
        for person in person_dict:
            res_stop = preprocess(path, person_dict[person])

            vn = []  # verb/noun
            for item in res_stop:
                # find all the nouns and verb in the article and save as list
                if item[1] in ["NNP", "NN", "NNS", "NNPS", "VB", "VBD",
                               "VBG", "VBN", "VBP", "VBZ"]:
                    vn.append(item[0])

            vnMatrix.append(vn)

    return vnMatrix


def handle_type_TF(path, type_list, topic_num):
    vnMatrix = extract_feature(path, type_list)

    # get the frequency topic matrix directly
    # (num1,num2) -> (index, counts)
    word_dictionary = corpora.Dictionary(vnMatrix)
    # bag of words format
    word_corpus = [word_dictionary.doc2bow(word) for word in vnMatrix]

    # using LAD get popular topic
    word_lda = models.ldamodel.LdaModel(corpus=word_corpus,
                                        id2word=word_dictionary,
                                        num_topics=topic_num,
                                        random_state=357,
                                        passes=10)

    # print the most popular 2 topics, all 5 words
    topic_list = word_lda.print_topics(2, num_words=5)
    for topic in topic_list:
        print(topic)

    # model evaluation by coherence scores
    word_coherencemodel_lda = CoherenceModel(model=word_lda,
                                             texts=vnMatrix,
                                             dictionary=word_dictionary,
                                             coherence='c_v')
    word_coherence = word_coherencemodel_lda.get_coherence()
    return word_coherence


def method_TF(path, all_type, topic_num):
    coherence_type = []  # storing each type coherence at topic_num

    print("feature extraction method was TF.")
    type_id = ['alltype', 'male', 'female', 'ageunder20', 'ageover20']
    for k, sim_type in enumerate(all_type):
        print("the two most popular topic of " + type_id[k])

        # get coherence of each type
        word_coherence = handle_type_TF(path, sim_type, topic_num)
        # storing coherence of each type
        coherence_type.append(word_coherence)

        print("coherence score of " + type_id[k] + " was "
              + str(word_coherence) + '\n')

    return coherence_type


def handle_type_TFIDF(path, type_list, topic_num):
    # the blogs from one person as a document
    vnMatrix = extract_feature(path, type_list, one_person=True)

    # get the feature matrrix topic by TFIDF
    word_dictionary = corpora.Dictionary(vnMatrix)

    word_corpus = [word_dictionary.doc2bow(word) for word in vnMatrix]
    # invoke TFIDF model
    word_tfidf = models.TfidfModel(word_corpus)
    word_corpus_tfidf = word_tfidf[word_corpus]

    # using LAD get popular topic
    word_lda = models.LdaModel(corpus=word_corpus_tfidf,
                               id2word=word_dictionary,
                               num_topics=topic_num,
                               random_state=357,
                               passes=10)

    # print the most two popular topic, all 5 words
    topic_list = word_lda.print_topics(2, num_words=5)
    for topic in topic_list:
        print(topic)

    # model evaluation by coherence
    word_coherencemodel_lda = CoherenceModel(model=word_lda,
                                             texts=vnMatrix,
                                             dictionary=word_dictionary,
                                             coherence='c_v')

    word_coherence = word_coherencemodel_lda.get_coherence()
    return word_coherence


def method_TFIDF(path, all_type, topic_num):
    # storing each type coherence at topic_num
    coherence_type = []

    print("feature extraction method was TFIDF.")
    type_id = ['alltype', 'male', 'female', 'ageunder20', 'ageover20']
    for k, sim_type in enumerate(all_type):
        print("the two most popular topic of " + type_id[k])

        # get coherence scores of each type
        word_coherence = handle_type_TFIDF(path, sim_type, topic_num)
        # storing coherence scores of each tyoe
        coherence_type.append(word_coherence)

        print("coherence score of" + type_id[k] +
              " was " + str(word_coherence) + '\n')

    return coherence_type


def main():
    # path = r"./blog"
    path = r"/Users/lindsay/Documents/AUT/AUT_S1/Text Mining/Ass2_final_results/testdata10"

    alllist, maleslist, femalelist, ageunder20, ageover20 = readByfilter(path)

    all_type = (alllist, maleslist, femalelist, ageunder20, ageover20)

    k = 10

    print("***************************************************\n" * 5)
    # get start run time
    start = datetime.datetime.now()

    # each type coherence at topic_k
    coherence_tf = method_TF(path, all_type, k)
    # each type coherence at topic_k
    coherence_tfidf = method_TFIDF(path, all_type, k)

    end = datetime.datetime.now()  # get end run time
    print("***************************************************\n" * 5)
    print(str(k) + '_topic cost time: ' + str(end - start))

    print('coherence_tf -> ', coherence_tf)
    print('coherence_tfidf -> ', coherence_tfidf)

    # plot the line of two methods of feature extraction    
    type_id = ['alltype', 'male', 'female', 'ageunder20', 'ageover20']

    plt.plot(type_id, coherence_tf, color='red',
                                    linewidth=2.0,
                                    linestyle='--',
                                    label='TF')

    plt.plot(type_id, coherence_tfidf, color='blue',
                                       linewidth=3.0,
                                       linestyle='-.',
                                       label="TFIDF")

    plt.xlabel('five type of demographics')
    plt.ylabel('coherence score')

    plt.legend()

    plt.savefig('coherence_baseline.pdf')
    plt.show()


if __name__ == '__main__':
    main()
