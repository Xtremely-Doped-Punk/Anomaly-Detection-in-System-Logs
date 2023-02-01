"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import re
import os
from typing import List
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime


class Logcluster:
    def __init__(self, logTemplate='', logIDL=None):
        self.logTemplate = logTemplate # list of words in log template
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL # list of lineId's that has this resp template pattern
    
    def printitems(self):
        print("logTemplate:",self.logTemplate,"\tlogIDL:",self.logIDL)
    
    def getLogTemplate(self):
        return ' '.join(self.logTemplate)


class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict() # dict to hold child's value as key and its node pointer as value
        self.childD = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken
    
    def printitems(self):
        print("|-----| depth:",self.depth, "token:",self.digitOrtoken, "childD:",self.childD, "|-----|\n")


class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4,
                 maxChild=100, rex=[], keep_para=True, showTree=True, showLeaf=False, debug=False):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.depth = depth - 2 # minus 2 because, 0th lvl is rootnode, and 1st lvl is no.of words in the log-template
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.showTree = showTree
        self.showLeaf = showLeaf
        self.debug = debug

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        retLogClust = None

        seqLen = len(seq)
        if seqLen not in rn.childD:
            return retLogClust

        parentn = rn.childD[seqLen]

        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return retLogClust
            currentDepth += 1

        logClustL = parentn.childD

        retLogClust = self.fastMatch(logClustL, seq)

        return retLogClust

    def addSeqToPrefixTree(self, rn, logClust):
        if self.debug:
            print(f"\nAdding {logClust.logTemplate} into tree")
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD: # checks keys of the dict
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:
            if self.debug:
                print("\ncurrent token:",token)
                print("curr->parentn:"); parentn.printitems()
            # Add current log cluster to the leaf node
            if currentDepth >= self.depth or currentDepth > seqLen:
                if self.debug:
                    print("constraints =","currentDepth:",currentDepth,"seqLen:",seqLen)
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                if self.debug:
                    print("after adding log cluster to leaf node:"); parentn.printitems()
                break
                

            # If token not matched in this layer of existing tree.
            if token not in parentn.childD:
                if self.debug:
                    print("no token match found")
                if not self.hasNumbers(token):
                    if self.debug:
                        print("token does'nt contain any number char in it")
                    if '<*>' in parentn.childD:
                        if self.debug:
                            print("childD of parentn has '<*>'")
                        if len(parentn.childD) < self.maxChild:
                            if self.debug:
                                print("parentn still has'nt reached max-child (thus will be added as child to the current parentn)")
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>'];
                            if self.debug:
                                print("parentn has reached max-child (thus will traverse one lvl further) -> new parrentn:")
                                parentn.printitems()
                    else:
                        if self.debug:
                            print("childD of parentn doesn't have '<*>'--> one child is always reserved for this token")
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                            if self.debug:
                                print("space available in this parentn, thus add as child and new parentn:"); parentn.printitems()
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                            if self.debug:
                                print("node reserved for '<*>' is not yet created, thus created now and new parentn:"); parentn.printitems()
                        else:
                            parentn = parentn.childD['<*>']
                            if self.debug:
                                print("fail safe, by now '<*>' node must be created, thus new parentn:"); parentn.printitems()

                else:
                    if self.debug:
                        print("token contains number char in it")
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                        if self.debug:
                            print("node reserved for '<*>' is not yet present, thus created now and new parentn:"); parentn.printitems()
                    else:
                        parentn = parentn.childD['<*>']
                        if self.debug:
                            print("fail safe, by now '<*>' node must be present, thus new parentn:"); parentn.printitems()

            # If the token is matched
            else:
                parentn = parentn.childD[token]
                if self.debug:
                    print("token match exists,  new parrentn:"); parentn.printitems()

            currentDepth += 1
          
        if self.debug:
            print("-"*50)

    # seq1 is template
    def seqDist(self, seq1, seq2): # fn to find sequence similarity
        assert len(seq1) == len(seq2) # raise error if length of both sequence is not same
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue #comment@haixuanguo: <*> == <*> are similar pairs
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar

    def fastMatch(self, logClustL, seq):
        if self.debug:
            print("fast match: parameters:",logClustL,seq)
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1 # no. of '<*>'==> parameters in logTemplate
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if self.debug:
                print("log-clust-temp:",logClust.logTemplate)
                print("curSim:",curSim,"curNumOfPara:",curNumOfPara)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust

        return retLogClust

    def getNewTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>') # any disimilarities is overwritten with '<*>'

            i += 1

        return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = [] 
        for logClust in logClustL:
            template_str = logClust.getLogTemplate() # join list of words in template back to str
            occurrence = len(logClust.logIDL) # no.of occurances = length of list containing the lineIDs that follow the resp template
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8] # crptographic hash mapping to reduced no.of char in data inp
            for logID in logClust.logIDL: # mapping the template_str and template_id to each log component dataframe
                logID -= 1 # lineID range => [1:n] but index range in dataframe => [0:n-1]; thus minus 1 from lineID
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences']) # 'log_name' +  '_templates.csv'
        

        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates
        # self.df_log is converted to => 'log_name' +  '_structured.csv'

        if self.keep_para: # if we want the actual parameters also in the resp places of '<*>' in log-template
            if self.debug:
                print("Finding the parameter_list for the resp places of '<*>' in log-template...")
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, 'structured.csv'), index=False)

        df_event.to_csv(os.path.join(self.savePath,'templates.csv'), index=False)
        # commented as it is done 2 times unnecessarily
        '''
        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False,
                        columns=["EventId", "EventTemplate", "Occurrences"]) 
        '''

    def printLeaf(self, logClustList:List[Logcluster]):
        print('<-<-<-<:'*(self.depth+1))
        for logClust in logClustList:
            print("Template:",'"'+logClust.getLogTemplate()+'"',"---","LineIDs:",logClust.logIDL)
        print(':>->->->'*(self.depth+1))
        print('||');print('|');

    def printTree(self, node, dep):
        tree_space = '| - - - ' #'\t'
        pStr = ''
        for i in range(dep):
            pStr += tree_space

        if node.depth == 0:
            pStr += 'Root'
        elif node.depth == 1:
            pStr += '<' + str(node.digitOrtoken) + '>'
        else:
            pStr += node.digitOrtoken

        print(pStr)

        if node.depth == self.depth:
            leaf = 'Leaf'
            if self.showLeaf:
                leaf+=':'
            print(tree_space*(self.depth+1),leaf)
            if self.showLeaf:
                self.printLeaf(node.childD)
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []

        self.load_data()
        if self.debug:
            print("preprocess rex:",self.rex)
            print("depth taken:",self.depth)

        count = 0
        for idx, line in self.df_log.iterrows():
            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split() # list of words of msg content og log
            # logmessageL = filter(lambda x: x != '', re.split('[\s=:,]', self.preprocess(line['Content'])))
            matchCluster = self.treeSearch(rootNode, logmessageL)

            # Match no existing log cluster
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)
                matchCluster = newCluster
                if self.debug:
                    print("added new cluster")

            # Add the new log message to the existing cluster
            else:
                if self.debug:
                    print("Cluster before getting new template"); matchCluster.printitems()
                newTemplate = self.getNewTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if ' '.join(newTemplate) != matchCluster.getLogTemplate(): 
                    # check if newly generated log templated merged string matchs the maximum similar log-cluster-template in tree
                    matchCluster.logTemplate = newTemplate
                if self.debug:
                    print("cluster exists already (after getting new template)")
            
            # check items
            if self.debug:
                matchCluster.printitems()

            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)), end='\r')

            if self.debug:
                self.printTree(rootNode,0) # printing tree
                print("+="*20,"=+"*20,"\n\n")

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.outputResult(logCluL)

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
        if self.showTree:
            print("\nTree Description:")
            print("1st lvl of tree is always 'Root-Node';") 
            print("2nd lvl of tree is used for categorizing no.of words in the resp LogTemplate; and")
            print("the last lvl of tree, i.e., a leaf of the tree contains a list of log-cluster that contains the log-template and LineIDs of the log that follow the resp pattern.")
            print('='*50,'\n')
            self.printTree(rootNode,0) # printing tree
            print('\n','='*50)

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        cnt = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    # print("\n", line)
                    # print(e)
                    pass
        print("Total size after encoding is", linecount, cnt)
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", str(row["EventTemplate"]))
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex) 
        # error fix not yet commited in original source: https://github.com/logpai/logparser/blob/master/logparser/Drain/Drain.py
        # fix can be seen in under pull request section https://github.com/logpai/logparser/pull/86
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        if self.debug:
            print("\ncontent:",row["Content"])
            print("template_regex final:", template_regex)
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        if self.debug:
            print("parameter_list final:", parameter_list)
        return parameter_list