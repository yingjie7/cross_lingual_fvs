import ast
import glob
import json
import os

import sys, pandas as pd 

sys.path.append('libs/SemEval2025-Task11-Evaluation/')
from data_preprocess import get_label_list_str
from utils import evaluate as semeval11_evaluate
 
 
def pairwise_prompting_json_to_dataframe(pred_data_, e_labels_str ="anger,fear,joy,sadness,surprise", 
                                         prefixid="eng_dev_track_a_", idx_convert=0, 
                                         all_ids=None, all_e_checks=None):
    e_labels = e_labels_str.split(",")
    data_submit = {
        "id": []
    }
    for e in e_labels:
        data_submit[e] = []
    mapping_values = {} 
    for e_id, e_check, e_pred in zip(all_ids, all_e_checks, pred_data_['detail_pred']):
        if e_id not in mapping_values:
            mapping_values[e_id] = {}
        
    mapping_values = dict([((e_id, e_check), e_pred) for e_id, e_check, e_pred in zip(all_ids, all_e_checks, pred_data_['detail_pred'])])
    checking_set = set()
    for e_id, e_check, e_preds in zip(all_ids, all_e_checks, pred_data_['detail_pred']):
        e_pred = e_preds[idx_convert]
        if e_id not in checking_set:
            data_submit['id'].append(e_id)
            checking_set.add(e_id)
        value = 0 if (e_pred == "none" or e_pred == "No") \
            else 1 if (e_pred == "low" or e_pred == "Yes") \
            else 2 if e_pred == "moderate"  \
            else 3 if e_pred == "high" else 0
        data_submit[e_check].append(value)
        
    df_submit = pd.DataFrame(data_submit)
    return df_submit

def json_to_dataframe_taskb(pred_data_, e_labels_str ="anger,fear,joy,sadness,surprise", prefixid="eng_dev_track_a_", idx_convert=0, all_ids=None, **kwargs):
    e_labels = e_labels_str.split(",")
    data_submit = {
        "id": []
    }
    for e in e_labels:
        data_submit[e] = []
        
    for idx, e in enumerate(pred_data_['detail_pred']):
        data_submit ['id'].append(prefixid + "{0:05d}".format(idx+1) if all_ids==None else all_ids[idx])
        all_preds = e[idx_convert].split(",")
        no_predicted_e =  set(e_labels)
        for e_pred in all_preds:
            for e_check in e_labels:
                if e_check not in e_pred: # e_pred = high degree of Happy ... , e_check=Happy
                    continue
                else:
                    no_predicted_e.remove(e_check)
                    intensity = 3 if 'high' in e_pred else 2 if 'moderate' in e_pred else 1 
                    data_submit[e_check].append(intensity)
                    break # no need to check other label if found 
        
        for e in no_predicted_e:
            data_submit[e].append(0)
        
    df_submit = pd.DataFrame(data_submit)
    return df_submit
    
def json_to_dataframe(pred_data_, e_labels_str ="anger,fear,joy,sadness,surprise", prefixid="eng_dev_track_a_", idx_convert=0, all_ids=None,  **kwargs):
    e_labels = [e.strip() for e in e_labels_str.split(",")]
    data_submit = {
        "id": []
    }
    for e in e_labels:
        data_submit[e] = []
        
    for idx, e in enumerate(pred_data_['detail_pred']):
        data_submit ['id'].append(prefixid + "{0:05d}".format(idx+1) if all_ids==None else all_ids[idx])
        for e_check in e_labels:
            data_submit[e_check].append(1 if e_check in e[idx_convert] else 0)
        
    df_submit = pd.DataFrame(data_submit)
    return df_submit

def _old_code():
    path_result = '/home/phuongnm/per_erc/finetuned_llm/semeval_track_a/alllanguage_Qwen3-8B_ep5_step-1_lrs-linear0.0003_r32_default-pairwise_ED10_seed-42_cstw-0.1_idRun-contrastive_bsz-8_asst-emb/result_esp_publicvalid_step-18350.json'
    file_result_name = path_result.split("/")[-1]
    d_type = file_result_name.split("_")[-2]
    folder_result = path_result[:-len(file_result_name)]+ f"/{d_type}/"
    all_data_eval = json.load(open(path_result))
    # b_default = True
    b_default = False
    
    task_id = 'a' if 'track_a' in path_result else 'b'
    publicvalid_folder = f"/home/phuongnm/per_erc/data/semeval11/7langs/{d_type}/track_{task_id}/"
    if '_pairwise-default_' in path_result or '_default-pairwise_'  in path_result:
        new_lst_preds = []
        new_lst_ids = []
        for e , id_str in zip(all_data_eval['detail_pred'], all_data_eval['all_ids']): 
            if '_pairwise-default_' in path_result:
                term_check =  ", is the emotion "
            elif '_default-pairwise_' in path_result:
                term_check = "which emotions are expressed in it?" 
            if b_default:
                term_check = "which emotions are expressed in it?" 
            if term_check in e[2]:
                new_lst_preds.append(e)
                new_lst_ids.append(id_str)
        all_data_eval['detail_pred'] = new_lst_preds
        all_data_eval['all_ids'] = new_lst_ids
    all_e_checks = [None]*len(all_data_eval['detail_pred'])
    
    if '_default_' in path_result or '_default-' in path_result or b_default :
        process_fn_json_to_dataframe = json_to_dataframe if task_id == 'a' else json_to_dataframe_taskb
    elif '_pairwise_' in path_result  or '_pairwise-' in path_result :
        process_fn_json_to_dataframe = pairwise_prompting_json_to_dataframe
        if task_id == "b":
            # the intensity of the emotion joy expressed in it?
            all_e_checks = [e[2].split("the intensity of the emotion ")[1].split(" expressed in it?")[0] for e in all_data_eval['detail_pred'] if "the intensity of the emotion " in e[2]]
        else:
            # of the emotion joy expressed in it?
            all_e_checks = [e[2].split(", is the emotion ")[1].split(" expressed in it?")[0] for e in all_data_eval['detail_pred'] if ", is the emotion " in e[2]]
    all_data_eval['e_check'] = all_e_checks
    label_str = "anger, disgust, fear, joy, sadness, surprise".replace(" ", "")
    group_data_eval  = {'alllanguage': all_data_eval}
    for e, e_id, e_check in zip(all_data_eval['detail_pred'], all_data_eval['all_ids'], all_e_checks):
        lang_id = e_id.split("_")[0]
        if lang_id not in group_data_eval:
            group_data_eval[lang_id] = {"detail_pred": [], "all_ids": [], 'e_check':[]}
        group_data_eval[lang_id]['detail_pred'].append(e)
        group_data_eval[lang_id]['all_ids'].append(e_id)
        group_data_eval[lang_id]['e_check'].append(e_check)
        group_data_eval[lang_id]['labels_set'] = ['id'] + list(pd.read_csv(f"{publicvalid_folder}/{lang_id}.csv").columns[2:])
        
    all_results = []
    langs = "chn	deu	eng	esp	pcm	ptbr	rus".split("	")
    report_scores = ["0"]*len(langs)
    for lang_id, data_eval in group_data_eval.items():
        df_pred = process_fn_json_to_dataframe(data_eval, e_labels_str=label_str, idx_convert=0, all_ids=data_eval['all_ids'], all_e_checks=data_eval['e_check'])
        if lang_id!="alllanguage":
            df_pred = df_pred[data_eval['labels_set']]
        pred_lines = df_pred.to_csv(sep=',', index=False).strip('\n').split('\n')
        
        if lang_id!="alllanguage":
            # df_pred['id'] = data_eval['all_ids']
            print(f"- Write file {folder_result}/track_{task_id}/pred_{lang_id}.csv")
            if not os.path.exists(f"{folder_result}/track_{task_id}/"):
                os.makedirs(f"{folder_result}/track_{task_id}/")
            df_pred.to_csv(f"{folder_result}/track_{task_id}/pred_{lang_id}.csv", sep=',', index=False) 
        
        df_gold = process_fn_json_to_dataframe(data_eval, e_labels_str=label_str, idx_convert=1, all_ids=data_eval['all_ids'], all_e_checks=data_eval['e_check'])
        if lang_id!="alllanguage":
            df_gold = df_gold[data_eval['labels_set']]
        gold_lines = df_gold.to_csv(sep=',', index=False).strip('\n').split('\n')
        
        scores = semeval11_evaluate(gold_lines, pred_lines, task=task_id)
        eval_f1_score = scores['macro']['f1'] if task_id=="a" else sum(scores.values()) / len(scores) 
        
        if task_id=="b":
            scores = dict([(k,float(v)) for k, v in scores.items() ])
        all_results.append({
            'lang_id': lang_id,
            'official_score': eval_f1_score,
            'scores': scores
        })
        # all_results = json.load(open('/home/phuongnm/per_erc/finetuned_llm/semeval_track_a/alllanguage_Qwen2.5-72B-Instruct_ep6_step-1_lrs-linear0.0003_r32_default_ED1000_idRun-evalphase/test/track_a/log_result_all.json'))
        dict_r= dict([(e['lang_id'], e['official_score']) for e in all_results])
        if lang_id !='alllanguage':
            report_scores[langs.index(lang_id)] = str(eval_f1_score)
        # print(lang_id, dict_r[lang_id])
        print(lang_id, scores)
    # json.dump(all_results, open(f"{folder_result}/track_{task_id}/log_result_all.json", "wt"), indent=1)
    print("	".join(langs))
    print("	".join(report_scores))

def eval_old_result():
    lang_list = "eng	deu	chn	esp	rus".split("	")
    def eval_by_semeval_metric(file_name):
        with open(file_name) as f:
            lang = file_name.split("/")[-1].split("_")[2]
            if lang == "to":
                lang = file_name.split("/")[-1].split("_")[3]
            lines = [e.strip() for e in f.readlines()]
            predictions = [",".join (ast.literal_eval(e.strip().replace("FV1 Pred: ", ""))) for e in lines if e.startswith("FV1 Pred: ")] 
            golds = [",".join (ast.literal_eval(e.strip().replace("Gold: ", ""))) for e in lines if e.startswith("Gold: ")] 
            old_result = [e for e in lines if e.startswith("FV Micro-F1:")][0]
            # baline_predictions = [",".join (e.strip().replace("Baseline Output: ", "").split(" ")) for e in lines if e.startswith("Baseline Output: ")] 
            # baseline_old_result = [e for e in lines if e.startswith("Baseline Micro-F1:")][0]

        
        all_preds = predictions
        all_labels = golds
        label_list_str = get_label_list_str(lang=lang)

        data_eval = {"detail_pred": list(zip(all_preds, all_labels))}
        df_pred = json_to_dataframe(data_eval, e_labels_str=label_list_str, idx_convert=0, all_ids=None)
        pred_lines = df_pred.to_csv(sep=',', index=False).strip('\n').split('\n')
        df_gold = json_to_dataframe(data_eval, e_labels_str=label_list_str, idx_convert=1, all_ids=None)
        gold_lines = df_gold.to_csv(sep=',', index=False).strip('\n').split('\n')

        task_id = 'a'
        scores = semeval11_evaluate(gold_lines, pred_lines, task='a')
        # print(scores)
        eval_f1_score = scores['macro']['f1'] if task_id=="a" else sum(scores.values()) / len(scores) 
        # print(eval_f1_score)
        return scores, eval_f1_score, old_result
    for file_name in glob.glob(f"data/noshared_fv_results/*_to_*_generation_results_top1_k_head0_meta-llama__Llama-3.1-8B-Instruct.txt"):
        print(file_name)
        print(eval_by_semeval_metric(file_name))

    # for fv_lang_in_fewshot_prompt in lang_list[1:]:
    #     for test_lang_in_zeroshot_promt in lang_list:
    #         file_name = glob.glob(f"data/fv_results/{fv_lang_in_fewshot_prompt}_to_{test_lang_in_zeroshot_promt}_*_generation_results_top2.txt")[0]
    #         file_name = glob.glob(f"data/fv_results/{fv_lang_in_fewshot_prompt}_to_{test_lang_in_zeroshot_promt}_*_generation_results_top1.txt")[0]
    #         file_name = glob.glob(f"data/fv_results/{fv_lang_in_fewshot_prompt}_label_to_{test_lang_in_zeroshot_promt}_*_generation_results_top1.txt")[0]
    #         file_name = glob.glob(f"data/fv_results/{fv_lang_in_fewshot_prompt}_label_to_{test_lang_in_zeroshot_promt}_*_generation_results_top1.txt")[0]
            
    #         print(fv_lang_in_fewshot_prompt, test_lang_in_zeroshot_promt, eval_by_semeval_metric(file_name))
    #         print("---"*5)

if __name__=="__main__":
    eval_old_result()