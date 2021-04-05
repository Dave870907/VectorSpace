import os
import VectorSpace
import numpy as np
import util
from operator import itemgetter, attrgetter
import argparse
import datetime

parser = argparse.ArgumentParser(description="HW_1")
parser.add_argument("-q", "--query", dest="query", help="input a query")
arg = parser.parse_args()


def print_top5(top_5):
    print('DocID         Score')
    for each in top_5:
        print('{}    {}'.format(each[0], each[1]))
    print('')


if __name__ == '__main__':
    doc = []
    indexList = []
    query = arg.query
    # test data
    yourPath = './EnglishNews'
    # 列出指定路徑底下所有檔案(包含資料夾)
    allFileList = os.listdir(yourPath)
    # 逐一查詢檔案清單
    for file in allFileList:
        f = open("EnglishNews/"+file, 'r')
        txt = f.read()
        doc.append(txt)
        indexList.append(file[:10])

    time_start = datetime.datetime.now()

    vectorSpace = VectorSpace.VectorSpace(doc)

    searchVector = np.array(vectorSpace.makeVectorforTFidf(query))

    tf_cosine = vectorSpace.searchTFCos(query)
    tf_euclidean = vectorSpace.searchTFdist(query)
    tfidf_cosine = vectorSpace.searchTFidfCos(query)
    tfidf_euclidean = vectorSpace.searchTFidfdist(query)

    top5_tf_cos = sorted(list(zip(indexList, tf_cosine)),
                         reverse=True, key=itemgetter(1))[:5]
    top5_tf_dist = sorted(list(zip(indexList, tf_euclidean)),
                          reverse=False, key=itemgetter(1))[:5]
    top5_tfidf_cos = sorted(list(zip(indexList, tfidf_cosine)),
                            reverse=True, key=itemgetter(1))[:5]
    top5_tfidf_dist = sorted(
        list(zip(indexList, tfidf_euclidean)), reverse=False, key=itemgetter(1))[:5]

    print('TF Weighting + Cosine Similarity:')
    print_top5(top5_tf_cos)

    print('TF Weighting + Euclidean Distance:')
    print_top5(top5_tf_dist)

    print('TF-IDF Weighting + Cosine Similarity:')
    print_top5(top5_tfidf_cos)

    print('TF-IDF Weighting + Euclidean Distance:')
    print_top5(top5_tfidf_dist)

    #   for Q2 Relevance feedback
    newSearchIndex = indexList.index(top5_tfidf_cos[0][0])
    documents = doc[newSearchIndex]
    feedBackVector = vectorSpace.makeVecRelevance(documents)
    ansVec = searchVector + feedBackVector

    finalScore = [util.cosine(ansVec, docVec)
                  for docVec in vectorSpace.tfidfVec]
    Q2 = sorted(list(zip(indexList, finalScore)),
                reverse=True, key=itemgetter(1))[:5]

    print('Relevance Feedback + TF-IDF Weighting + Cosine Similarity:')
    print_top5(Q2)
    time_end = datetime.datetime.now()
    time_cost = time_end-time_start
    print("cost time :", time_cost)
