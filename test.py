import numpy
from sklearn.ensemble import RandomForestClassifier
import re
import math

domainlist = []
Inputlist = []
InputDomainName = []
InputLabel = []
def cal_entropy(text):
    h = 0.0
    sum = 0
    letter = [0] * 26
    text = text.lower()
    for i in range(len(text)):
        if text[i].isalpha():
            letter[ord(text[i]) - ord('a')] += 1
            sum += 1
    #print('\n', letter)
    if sum==0:
        return 1
    for i in range(26):
        p = 1.0 * letter[i] / sum
        if p > 0:
            h += -(p * math.log(p, 2))
    return h


class Domain:
    def __init__(self, _domainName, _label,_genName):
        self.domainName = _domainName
        self.label = _label
        self.genName = _genName
        self.genNameLength = len(self.genName)
        self.numbers = len(re.findall('\d+', self.genName))
        self.domainNameEntropy = cal_entropy(self.genName)
        self.segmentation = len(re.findall('\.', self.domainName))
        
    def returnData(self):
        return [self.genNameLength, self.numbers, self.domainNameEntropy, self.segmentation]
    
    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else: 
            return 1

def inputData(filename):
    f = open(filename).read().splitlines()
    for line in f:
        strDomain = line
        line = line.strip()
        tokens = line.split(".")
        genName = tokens[0]
        genNameLength = len(genName)
        numbers = len(re.findall('\d+', genName))
        genNameEntropy = cal_entropy(genName)
        segmentation = len(re.findall('\.', strDomain))
        Inputlist.append([genNameLength, numbers, genNameEntropy, segmentation])
        InputDomainName.append(strDomain)

def initData(filename):
    f = open(filename).read().splitlines()
    for line in f:
        line = line.strip()
        tokens = line.split(',')
        domainName = domainname = tokens[0]
        domainname = domainname.strip()
        Tokens = domainname.split('.')
        genName = Tokens[0]
        label = tokens[1]
        domainlist.append(Domain(domainName, label, genName))


def transformList(predictList):
    for i in predictList:
        if i==0:
            InputLabel.append("notdga")
        else:
            InputLabel.append("dga")

def main():
    initData("train.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    inputData("test.txt")
    predictList = clf.predict(Inputlist)
    transformList(clf)
    file = open("result.txt", mode='w')
    for i in range(len(predictList)):
        file.write(InputDomainName[i] + ',' +InputLabel[i] + '\n')
    
if __name__ == '__main__':
    main()