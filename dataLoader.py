import sys
import io
import os
import csv
import random
import torch
import itertools

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 40
MIN_COUNT = 5


sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
def load_conv():
	questions = []
	answers = []
	with open('data/Noblank_Question.txt',encoding='utf-8') as f:
		lines = f.readlines()
	with open('data/Noblank_Answer.txt',encoding='utf-8') as f:
		lines_ans = f.readlines()	
	for line in lines: 
		senten = line.replace('\n','')
		questions.append(senten)
	for l in lines_ans:
		senten = l.replace('\n','')
		answers.append(senten)
	return questions,answers
	# print(len(questions))
	# print(len(answers))
	# print(questions[100]+'\n',answers[100])

def loadConversations(questions,answers):
	conversations = []
	for i in range(len(questions)):
		pair = [questions[i],answers[i]]
		conversations.append(pair)
	# print(len(conversations))
	# print(conversations[90])
	return conversations

def filter(pair):
	return 2 <= len(pair[0]) < MAX_LENGTH and len(pair[1]) < MAX_LENGTH 

def filterPairs(pairs):
	return [pair for pair in pairs if filter(pair)]

class  Vocab(object):
	"""docstring for  Vocab"""
	def __init__(self):
		super( Vocab, self).__init__()
		self.word2index = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.word2count = {}
		self.num_words = 3
		self.trimmed = False

	def addSentence(self, sentence):  #句子中的每个词
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2count[word]+=1

	def trim(self,min_count):
		if self.trimmed:  #如果已经删减过了返回
			return
		keep_words = []  #删减后保留的词
		for word,count in self.word2count.items():
			if count > min_count:
				keep_words.append(word)

		self.word2index = {}
		self.word2count = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.num_words = 3 
		for word in keep_words:
			self.addWord(word)
		self.trimmed = True




def processSenten(sentences):
	voc = Vocab()
	for pair in sentences:
		voc.addSentence(pair[0])
		voc.addSentence(pair[1])
	# print("Counted words:", voc.num_words)
	return voc


def trimWords(voc, pairs, MIN_COUNT):
	'''
	函数作用：只保留全部词均不在被修剪掉了的问、答句子
	'''
	voc.trim(MIN_COUNT)
	keep_pairs = []
	for pair in pairs:
		ques = pair[0]
		ans = pair[1]
		keep = True
		for word in ques.split(' '):
			if word not in voc.word2index:
				keep = False
				break
		for word in ans.split(' '):
			if word not in voc.word2index:
				keep = False
				break	
		if keep:
			keep_pairs.append(pair)
	return keep_pairs



def maskMatrix(padList):
	'''
	用0,1进行mask pad_token位置为0  其余为1
	'''
	m = []
	for i, seq in enumerate(padList):
		# print(i,seq)
		m.append([])
		for token in seq:
			if token == PAD_token:
				m[i].append(0)
			else:
				m[i].append(1)
	return m


def zeroPadding(index_batch,maxlength):
	padding_batch = []
	for index in index_batch:
		# print(index)
		if len(index)<maxlength:
			# index.append()
			index.extend([PAD_token] * (maxlength-len(index)))
		padding_batch.append(index)
	# print("普通转置结果：",list(zip(*padding_batch)))
	return list(zip(*padding_batch))  #转置


def inputVar(voc,sentence):
	'''
	param-sentence:
	      接收形式为[句1,句2,句3...]
	'''
	index_batch = []
	for s in sentence:
		index_onesenten = [voc.word2index[word] for word in s.strip().split(" ")] + [EOS_token]
		index_batch.append(index_onesenten)
	# print(index_batch)
	lengths = torch.tensor([len(index) for index in index_batch])
	maxlength = lengths[0]
	# print(lengths)
	# print("max_len:",maxlength)
	padList = zeroPadding(index_batch,maxlength)
	padVar = torch.LongTensor(padList)
	# print(padList)
	return padVar, lengths

def outputVar(voc,sentence):
	'''
	回答的句子即输出句子 要进行mask
	'''
	index_batch = []
	for s in sentence:
		index_onesenten = [voc.word2index[word] for word in s.strip().split(" ")] + [EOS_token]
		index_batch.append(index_onesenten)
	max_target_len = max([len(index) for index in index_batch])
	padList = zeroPadding(index_batch,max_target_len)
	mask = maskMatrix(padList)
	mask = torch.BoolTensor(mask)
	padVar = torch.LongTensor(padList)
	return padVar, mask, max_target_len


	

def getBatchData(voc,pair_batch):
	'''
	传入一批数据
	'''
	# print(pair_batch)
	pair_batch.sort(key=lambda x:len(x[0].split(" ")),reverse=True)  #按照问句的长度排序？
	# print(pair_batch)
	input_batch = []
	output_batch = []
	for pair in pair_batch:
		input_batch.append(pair[0])
		output_batch.append(pair[1])
	inp, lengths = inputVar(voc,input_batch)
	output, mask, max_target_len = outputVar(voc,output_batch)
	return inp, lengths, output, mask, max_target_len



questions,answers = load_conv()
convers = loadConversations(questions,answers)
pairs_filter = filterPairs(convers)
voc = processSenten(pairs_filter)
pairs = trimWords(voc, pairs_filter, MIN_COUNT)
# if __name__ == '__main__':
# 	questions,answers = load_conv()
# 	convers = loadConversations(questions,answers)
# 	# datafile = os.path.join('data', "formatted_lines.txt")
# 	# with open(datafile, 'w', encoding='utf-8') as outputfile:
# 	# 	writer = csv.writer(outputfile, delimiter=',', lineterminator='\n')
# 	# 	for pair in convers:
# 	# 		writer.writerow(pair)

# 	pairs_filter = filterPairs(convers)
# 	# print("过滤掉过长和过短的句子后共有:",len(pairs_filter)) #190789
# 	voc = processSenten(pairs_filter)
# 	pairs = trimWords(voc, pairs_filter, MIN_COUNT)
# 	# print("修剪后的句子对长：",len(pairs)) #134352
# 	input_s, lengths, target_s, mask, max_target_len = getBatchData(voc,[random.choice(pairs) for _ in range(5)])
# 	print("input_variable:", input_s)
# 	print("lengths:", lengths)
# 	print("target_variable:", target_s)
# 	print("mask:", mask)







