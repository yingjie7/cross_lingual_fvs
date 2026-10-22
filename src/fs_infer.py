import argparse
import glob
import logging
import sys, inspect
import time
from datetime import datetime, timedelta
from typing import Optional
import json
import os
import pandas as pd
from peft import AutoPeftModelForCausalLM
import torch
from torch import nn 
from lightning import seed_everything
import gc
from tqdm import tqdm

import transformers 
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM 
from transformers.data.data_collator import DataCollatorWithPadding

from data_preprocess import generate_fs_prompting_data, get_label_list_str
from fs_infer_llm_base import llm_infer  
import pandas as pd 
from fv_utils import DisableHeadTriggerCriteria, do_intervention
from semeval_llm_eval import json_to_dataframe, semeval11_evaluate

logger = logging.getLogger(__name__)

def semeval_eval_metric(path_file_result):
	try:
		_data = json.load(open(path_file_result))
		_config = _data['config']
		lang = _config['lang']
		pred_data = _data['detail_prediction']
		all_preds = [e['generated_output'].strip() for e in pred_data]
		all_labels = [e['label'].strip().split(", ") for e in pred_data]
		if 'label_list_str' not in _config:
			label_list_str = get_label_list_str(lang=lang)
		else:
			label_list_str = _config['label_list_str']

		all_ids = [e['id'] for e in pred_data]
		data_eval = {"detail_pred": list(zip(all_preds, all_labels))}
		df_pred = json_to_dataframe(data_eval, e_labels_str=label_list_str, idx_convert=0, all_ids=all_ids)
		pred_lines = df_pred.to_csv(sep=',', index=False).strip('\n').split('\n')
		df_gold = json_to_dataframe(data_eval, e_labels_str=label_list_str, idx_convert=1, all_ids=all_ids)
		gold_lines = df_gold.to_csv(sep=',', index=False).strip('\n').split('\n')

		task_id = 'a'
		scores = semeval11_evaluate(gold_lines, pred_lines, task='a')
		print(scores)
		eval_f1_score = scores['macro']['f1'] if task_id=="a" else sum(scores.values()) / len(scores) 
		_data['result']['score'] = eval_f1_score
		_data['result']['detail_scores'] = scores

		return _data, _data['result']['score']
	except Exception as e:
		print(f"err - {path_file_result}")
		pass
	return {}, -1
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process ...')
	parser.add_argument('--prompting_data_path',type=str, help='prompting_data_path', default=None)
	parser.add_argument('--model_name',type=str, help='model_name', default="Qwen/Qwen3-4B") 
	parser.add_argument('--max_new_tokens',type=int, help='max_new_tokens', default=500) 
	parser.add_argument('--batch_size',type=int, help='batch_size', default=4) 
	parser.add_argument('--eos_tok_str',type=str, help='eos_tok_str', default="\n\n") 
	parser.add_argument('--use_8bit_quantization',action="store_true", help='use_8bit_quantization', default=False)
	parser.add_argument('--output_id',type=str, help='output_id', default="result") 
	parser.add_argument('--scan_results', nargs='+',  help='scan path pattern of results', default=None)
	parser.add_argument('--dump_selected_head', action="store_true", help='dump_selected_head', default=False) 
	parser.add_argument('--use_fv', action="store_true", help='use_fv', default=False) 
	parser.add_argument('--fv_data_folder', type=str, help='fv_data_folder', default=None) 
	parser.add_argument('--fv_top_n_heads',type=int, help='batch_size', default=5) 
	parser.add_argument('--fv_avoid_reverse_rule', action="store_true", help='fv_avoid_reverse_rule', default=False) 
	parser.add_argument('--do_sample', action="store_true", help='do_sample', default=False) 
	parser.add_argument('--seed', type=int, help='random seed', default=42) 
	parser.add_argument('--dynamic_disable_heads', action="store_true", help='dynamic_disable_heads', default=False) 
	parser.add_argument('--disable_head_idx', type=str, help='disable_head_idx', default="") 
	parser.add_argument('--head_roles', type=str, help='path of head_roles', default=None) 
	parser.add_argument('--weight_disable_head', type=float, help='weight_disable_head', default=0.0) 
	parser.add_argument('--weight_strengthen_head', type=float, help='weight_strengthen_head', default=1.5) 
	parser.add_argument('--intervention', type=str, help='intervention method [disable_head, disable_multi_heads]', default=None) 
	parser.add_argument('--dump_logits', action="store_true", help='do_sample', default=False) 
	parser.add_argument('--lang',  type=str, help='lang', default='eng') 
	parser.add_argument('--fs_lang',  type=str, help='fs_lang', default=None) 
	parser.add_argument('--re_gen_data',  action="store_true", help='re_gen_data', default=False) 
	parser.add_argument('--use_fv_on_fs',  action="store_true", help='use_fv_on_fs', default=False) 
	parser.add_argument('--shuffle_demons',  action="store_true", help='shuffle_demons', default=False) 
	parser.add_argument('--indirect_effect_mt',  type=str, help='indirect_effect_mt', default=None) 
	parser.add_argument('--fv_no_shared', action="store_true", help='fv_no_shared', default=False) 
	parser.add_argument('--only_eval', action="store_true", help='only_eval', default=False) 

	args, unknown = parser.parse_known_args()
	seed_everything(args.seed) 
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if "\\n" in args.eos_tok_str:
		args.eos_tok_str = args.eos_tok_str.replace("\\n", "\n") 
	print(args)

	# ========================================
	# scan all the results 
	if args.scan_results is not None:
		path_file_out = f"{'/'.join(args.scan_results[0].split('/')[:-1])}/all_result.csv"
		def_values = dict([(k, v) for k, v in args.__dict__.items()])
		all_results = []
		for pattern_path in args.scan_results:
			for r_file in glob.glob(pattern_path):
				r_performance= {}
				all_infor = json.load(open(r_file))
				if isinstance(all_infor, dict) and "config" in all_infor and 'result' in all_infor:
					r_configs = all_infor['config']
					r_performance['time'] = datetime.fromtimestamp(int(os.path.getmtime(r_file))).strftime("%Y/%m/%d, %H:%M:%S")
					r_performance.update(all_infor['result'])
					r_performance.update(def_values)
					r_performance.update(r_configs)
					r_performance["score"] = r_performance['score']*100
					r_performance["strong_att_percentile"] = r_performance['strong_att_percentile']*100
					r_performance['path_file'] = r_file
					all_results.append(r_performance)
				else:
					continue

		df_result = pd.DataFrame(all_results)
		print(f"- Write file: {path_file_out}")
		df_result.to_csv(path_file_out, sep=',')
		exit(0) 

	# ========================================
	# check configs  
	global g_args
	g_args = args

	model_name =  args.model_name #  "Qwen/Qwen3-14B"   # "Qwen/Qwen3-8B"  #  # 'meta-llama/Llama-3.1-8B-Instruct'  'meta-llama/Meta-Llama-3-8B-Instruct'   
	
	# ========================================
	# Load data and exit if the result exist!!
	lang = args.lang
	
	if not g_args.fv_no_shared:
		g_args.prompting_data_path = f'data/fv_results_{model_name.replace("/", "-")}/{lang}_test_fs_prompt.json' if g_args.prompting_data_path is None else g_args.prompting_data_path
	else:
		g_args.prompting_data_path = f'data/noshared_fv_results_{model_name.replace("/", "-")}/{lang}_test_fs_prompt.json' if g_args.prompting_data_path is None else g_args.prompting_data_path

	if args.fs_lang is None:
		args.fs_lang = args.lang
	
	if args.fs_lang != args.lang:
		g_args.prompting_data_path = g_args.prompting_data_path.replace(".json", f'.fslang-{g_args.fs_lang}.json')
	if args.use_fv_on_fs:
		g_args.prompting_data_path = g_args.prompting_data_path.replace(".json", f'.fv.json')
	
	# ========================================
	# generate data 
	if args.re_gen_data:
		prompt_data_folder = "/".join(g_args.prompting_data_path.split("/")[:-1])
		os.makedirs(prompt_data_folder, exist_ok=True)
		if args.shuffle_demons:
			g_args.prompting_data_path = g_args.prompting_data_path.replace(".json", ".shuffle_demons.json")
		prompting_data_path = g_args.prompting_data_path
		all_test_data = generate_fs_prompting_data(k_shot=5, lang=lang, path_file_out=prompting_data_path, fs_lang=args.fs_lang, shuffle_demons=g_args.shuffle_demons)

	raw_data = json.load(open(prompting_data_path))
	all_prompt_content= [e['prompting'] for e in raw_data]
	print(all_prompt_content[0])

	# ========================================
	# construct pattern name of file output
	label_list_str = get_label_list_str(lang=lang)
	g_args.label_list_str = label_list_str
	
	path_file_result = prompting_data_path.replace(
		".json", f'.{model_name.split("/")[-1].split("idRun-")[-1]}.ntok-{args.max_new_tokens}.quant8bit-{args.use_8bit_quantization}'+\
		f'.json') 
	
	if args.intervention == "disable_head" and len(args.disable_head_idx) > 0:
		path_file_result = path_file_result.replace(f".json", f".delH-.json")
	elif args.intervention is not None:
		path_file_result = path_file_result.replace(f".json", f".itv-{args.intervention}.json")

	if args.use_fv or args.use_fv_on_fs:
		data_folder__ = "/".join(path_file_result.split("/")[:-1])
		if not args.fv_no_shared:
			args.fv_data_folder = args.fv_data_folder or f"{data_folder__}/fv_{model_name.split('/')[-1]}_{args.output_id}/"
		else:
			args.fv_data_folder = args.fv_data_folder or f"{data_folder__}/testlang-{args.lang}_fslang-{args.fs_lang}_fv_{model_name.split('/')[-1]}_{args.output_id}/"
		path_file_result = path_file_result.replace(f".json", f"_fv-topH{args.fv_top_n_heads}.json")
	if args.use_fv and args.fv_avoid_reverse_rule:
		path_file_result = path_file_result.replace(f".json", f"-neg.json")
	if args.output_id != 'result':
		path_file_result = path_file_result.replace(f".json", f".{args.output_id}.json")

	
	print("path_file_result = ", path_file_result)
	eval_func = semeval_eval_metric
	
	# ========================================
	# check file inference exist
	if os.path.exists(path_file_result) or os.path.exists(path_file_result.replace("-None.json", ".json")):
		print(f"File {path_file_result} already exists, skip inference")
		saved_results, acc = eval_func(path_file_result)
		json.dump(saved_results, open(path_file_result, "wt", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
		exit()
	
	json.dump({}, open(path_file_result, 'wt')) # lock 

	# ========================================
	# Load Model and Tokenizer 
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.padding_side = 'left'
	tokenizer.pad_token = tokenizer.eos_token

	tensor_data_type = torch.bfloat16   
	if "gpt-oss" not in  model_name:
		# bnb_config = BitsAndBytesConfig(
		# 	load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=tensor_data_type
		# )
		bnb_config = BitsAndBytesConfig(
			load_in_8bit=True
		)

		# if args.is_local_llm:
		# 	model = AutoPeftModelForCausalLM.from_pretrained(
		# 		model_name,
		# 		device_map="auto",
		# 		# quantization_config=bnb_config,
		# 	)
		# else:
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			device_map="auto",
			torch_dtype=tensor_data_type,
			low_cpu_mem_usage=True,
			quantization_config = bnb_config if args.use_8bit_quantization else None,
		) 

	else:
		from transformers import Mxfp4Config
		# bnb_config = Mxfp4Config(
		# 	load_in_4bit=True
		# )
		bnb_config = Mxfp4Config(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")

		model = AutoModelForCausalLM.from_pretrained(
			"openai/gpt-oss-20b",
			torch_dtype="bfloat16",  # or torch.float16
			# attn_implementation="my_new_sdpa",
			device_map="auto"
		)
	model.eval() 
	

	# ========================================
	# RUN inference
	def do_infer(args): 
		# Data collator -> pads dynamically to longest in batch
		data_collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")

		# Custom collate function: tokenize inside collator
		def collate_fn(batch):
			batch_input_ids = [tokenizer(item["text"]) for item in batch]
			padded_batch_data = data_collator(batch_input_ids)
			padded_batch_data['sample_idx'] = [item["idx"] for item in batch]

			return padded_batch_data

		print("=== Do infer ===")
		print(args)

		if args.only_eval:
			return 
			
		start_time = time.time()
		all_output_text = llm_infer(model, tokenizer, all_prompt_content, 
								max_new_tokens=args.max_new_tokens, 
								eos_tok_str=args.eos_tok_str,
								batch_size=args.batch_size,
								args=args,
								predefined_collate_fn=collate_fn )
		processing_time = str(timedelta(seconds=int(time.time()-start_time)))


		# ========================================
		# RUN evaluation and save results 
		args.disable_head_trigger=None
		results = []
		for prompt_content, output_text, raw_data_e in zip(all_prompt_content, all_output_text, raw_data):
			results.append({
					"id": raw_data_e['id'],
					'label': raw_data_e['label'],
					"prompt": prompt_content,
					"generated_output": output_text,
					"correct": None,
					"err_format": None,
				}) 

		gpu_info = ""
		if torch.cuda.is_available():
			props = torch.cuda.get_device_properties(0)
			gpu_info +=(f"\nDevice name: {torch.cuda.get_device_name(0)}")
			gpu_info +=(f"\nDevice count: {torch.cuda.device_count()}")
			gpu_info +=(f"\nCurrent device: {torch.cuda.current_device()}")
			gpu_info +=(f"\nTotal Memory: {props.total_memory / 1024**2:.0f} MB")
			gpu_info +=(f"\nCompute Capability: {props.major}.{props.minor}")
			gpu_info +=(f"\nCUDA version: {torch.version.cuda}")

		gpu_info +=(f"\nTransformers version: {transformers.__version__}")
		gpu_info +=(f"\nPyTorch version: {torch.__version__}")
		gpu_info +=(f"\nIs CUDA available: {torch.cuda.is_available()}")
		print(gpu_info)

		saved_results = {
				"result": {'processing_time': processing_time, "gpu_info": gpu_info},
				"config": dict([(k, v)for k, v in args.__dict__.items()] + [("path_file_result", path_file_result)]), 
				"detail_prediction": results, 
				"code_version": os.environ.get('CODE_VERSION')}
		json.dump(saved_results, open(path_file_result, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str) 

		saved_results, score = eval_func(path_file_result)
		saved_results["result"]['score'] =  score 
		json.dump(saved_results, open(path_file_result, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

	if args.intervention is not None:
		# add hook function 
		hook_handles = do_intervention(args, model, 
									global_signals=None)

		# do infer with hook function 
		do_infer(args)

		# remove hook function 
		for hook_handle in hook_handles:
			hook_handle.remove()
	else:
		# do infer 
		do_infer(args)
		
			
		
