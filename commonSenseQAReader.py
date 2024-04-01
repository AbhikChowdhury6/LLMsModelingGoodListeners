def reader(dataset):
	'''
	Usage: 
	from commonSenseQAReader import reader
	from datasets import load_dataset

	dataset = load_dataset("tau/commonsense_qa")
	qa_reader = reader(dataset)
	for ques, ans in qa_reader:
		print(ques, ans)
	'''
	for data in dataset["train"]:
		ques_str = data["question"]
		ans = data["answerKey"]
		ans_str = None
		for text, label in zip(data['choices']['text'], data['choices']['label']):
			if(label == ans):
				ans_str = text
		yield [ques_str, ans_str]