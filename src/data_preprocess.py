import json
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_fs_prompting_data(PATH_DATA=f"data/semeval11/7langs", lang = 'deu', k_shot=10, path_file_out="data/data_out.json", 
							   fs_lang=None, 
							   dump_data=True, 
							   shuffle_demons=False):
	
	if fs_lang is None:
		fs_lang = lang

	# PATH_DATA=f"/home/s2220429/multilingual_er/data/semeval11/7langs"

	train_path = f"{PATH_DATA}/train/track_a/{fs_lang}.csv"
	train_df = pd.read_csv(train_path)
	test_path = f"{PATH_DATA}/test/track_a/{lang}.csv"
	test_df = pd.read_csv(test_path)

	train_emotion_labels = train_df.columns[2:].tolist()
	test_emotion_labels = test_df.columns[2:].tolist()

	# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	def row_to_label(row, emotion_labels):
		labels = [e for e in emotion_labels if row[e] == 1]
		return ", ".join(labels) if len(labels) > 0 else "none"

	def format_text(text, label_list_str, label=""):
		return \
		(
			f"### Input: {text}\n" 
			f"### Emotions: {label_list_str}\n"
			f"### Answer: {label}"
		)
	if shuffle_demons:
		fs = train_df.sample(k_shot)
	else:
		fs = train_df.iloc[:k_shot]
	fs_examples = []
	label_list_str = ", ".join(train_emotion_labels) + ", or none"
	for _, row in fs.iterrows():
		label = row_to_label(row, train_emotion_labels)
		fs_examples.append(format_text(row['text'].strip(), label_list_str, label))

	all_test_data = []
	label_list_str = ", ".join(test_emotion_labels) + ", or none"
	for _, row in test_df.iterrows():
		test_label = row_to_label(row, test_emotion_labels)
		test_example = [format_text(row['text'].strip(), label_list_str, '')]
		all_test_data.append({
			"id": row['id'],
			"prompting": "\n\n".join(fs_examples+test_example),
			"label": test_label
		})
	if dump_data:
		json.dump(all_test_data[:], open(path_file_out, 'wt'), ensure_ascii=False, indent=1)
	return all_test_data

def get_label_list_str(PATH_DATA=f"data/semeval11/7langs", lang = 'deu'):
	train_path = f"{PATH_DATA}/train/track_a/{lang}.csv"
	train_df = pd.read_csv(train_path)
	emotion_labels = train_df.columns[2:].tolist()
	return ", ".join(emotion_labels)   

if __name__=="__main__":
	all_test_data = generate_fs_prompting_data( k_shot=5, dump_data=False, shuffle_demons=True)
	print(all_test_data[4])
	print(all_test_data[5])
