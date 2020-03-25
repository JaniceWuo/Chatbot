import config
import model

def main():
	if config.mode == "train":
		model.ModelTrain()
	elif config.mode == "chat":
		model.ModelVal()

if __name__ == '__main__':
	main()