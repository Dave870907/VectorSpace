
# tfidf
from __future__ import division, unicode_literals
import util
from Parser import Parser
from pprint import pprint
import math
import nltk
from textblob import TextBlob as tb
import numpy as np
import heapq
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers.
    A document is represented as a vector. Each dimension of the vector corresponds to a
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """
    documents = []

    # Mapping of vector index to keyword
    vectorKeywordIndex = []

    # tf
    tfVec = []

    # tfidf
    tfidfVec = []

    # Tidies terms
    parser = None

    def __init__(self, documents=[]):
        # self.documentVectors = []
        self.tfVec = []
        self.tfidfVec = []
        self.parser = Parser()
        self.documents = documents
        if(len(documents) > 0):
            self.build(documents)

    def build(self, documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        # self.documentVectors = [self.makeVector(
        # document) for document in documents]
        self.tfVec = [self.makeVectorforTF(document) for document in documents]
        self.tfidfVec = [self.makeVectorforTFidf(
            document) for document in documents]

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        # Mapped documents into a single word string
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        # Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex = {}
        offset = 0
        # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word] = offset
            offset += 1
        return vectorIndex  # (keyword:position)

    def makeVectorforTF(self, wordString):
        # Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        wordSet = set(wordList)

        for words in wordSet:
            tf = util.tf(words, wordList)
            vector[self.vectorKeywordIndex[words]] += tf
        return vector

    def makeVectorforTFidf(self, wordString):
        # Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        wordSet = set(wordList)

        for words in wordSet:
            try:
                tfidf = util.tfidf(words, wordList, self.documents)
                vector[self.vectorKeywordIndex[words]] += tfidf
            except:
                continue
        return vector

    # def buildQueryVector(self, termList):
    #     """ convert query string into a term vector """
    #     query = self.makeVector(" ".join(termList))
    #     return query

    def related(self, documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector)
                   for documentVector in self.documentVectors]
        # ratings.sort(reverse=True)
        return ratings

    def searchTFCos(self, searchList):
        searchVec = self.makeVectorforTF(searchList)
        vector = [util.cosine(searchVec, docVec)for docVec in self.tfVec]
        return vector

    def searchTFdist(self, searchList):
        searchVec = self.makeVectorforTF(searchList)
        vector = [util.euclidean(searchVec, docVec)for docVec in self.tfVec]
        return vector

    def searchTFidfCos(self, searchList):
        searchVec = self.makeVectorforTFidf(searchList)
        vector = [util.cosine(searchVec, docVec)for docVec in self.tfidfVec]
        return vector

    def searchTFidfdist(self, searchList):
        searchVec = self.makeVectorforTFidf(searchList)
        vector = [util.euclidean(searchVec, docVec)for docVec in self.tfidfVec]
        return vector

    def makeVecRelevance(self, wordString):
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        result = nltk.pos_tag(wordList)

        feedbackVec = []
        for word in result:
            if ('VB' in word[1] or 'NN' in word[1]):
                feedbackVec.append(word[0])

        return np.array(self.makeVectorforTFidf(' '.join(feedbackVec)))*0.5


###################################################
