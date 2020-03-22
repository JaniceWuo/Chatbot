import sys
import io
import os
import csv

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
	print("Counted words:", voc.num_words)
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


if __name__ == '__main__':
	questions,answers = load_conv()
	convers = loadConversations(questions,answers)
	# datafile = os.path.join('data', "formatted_lines.txt")
	# with open(datafile, 'w', encoding='utf-8') as outputfile:
	# 	writer = csv.writer(outputfile, delimiter=',', lineterminator='\n')
	# 	for pair in convers:
	# 		writer.writerow(pair)

	pairs_filter = filterPairs(convers)
	print("过滤掉过长和过短的句子后共有:",len(pairs_filter)) #190789
	voc = processSenten(pairs_filter)
	pairs = trimWords(voc, pairs_filter, MIN_COUNT)
	print("修剪后的句子对长：",len(pairs)) #134352
	





