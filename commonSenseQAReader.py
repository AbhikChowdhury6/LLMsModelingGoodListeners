import pandas as pd    

def reader(file_path):
	data_df = pd.read_json(path_or_buf=file_path, lines=True)
	# sample datapoint
	# {'question_concept': 'punishing', 'choices': [{'label': 'A', 'text': 'ignore'}, 
	# {'label': 'B', 'text': 'enforce'}, {'label': 'C', 'text': 'authoritarian'}, {'label': 'D', 'text': 'yell at'}, {'label': 'E', 'text': 'avoid'}], 
	# 'stem': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?'}
	for idx in data_df.index:
		ques = data_df["question"][idx]
		ans = data_df["answerKey"][idx]
		ques_str = ques['stem']
		ans_str = None
		for choice in ques['choices']:
			if(choice['label'] == ans):
				ans_str = choice['text']
		yield [ques_str, ans_str]


out = reader("commonSenseQA/train_rand_split.jsonl")
for ques, ans in out:
	print(ques, ans)