import argparse
import dataclasses
import datetime as dt
import functools
import os
from typing import Iterable, Optional

# This prevents some pytorch cuda OOM issues.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.75'

import submitit

import io_utils
from dataset_utils import kw_product
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params


import dense_retriever_batched


@dataclasses.dataclass
class Args:
    pass


date_str = dt.date.today().strftime('%y%m%d')
datetime_str = str(dt.datetime.now()).split('.')[0]

DEFAULT_LANG_IDS = {
    'mintaka': ['ar', 'de', 'ja', 'hi', 'pt', 'es', 'it', 'fr'],
    'mkqa': ['ar', 'de', 'es', 'fr', 'it', 'ja', 'pt', 'ko', 'zh_cn']
}


s_multi = """
python job_submission.py \
    --model_file ../models/mDPR_biencoder_best.cpt \
    --ctx_file ../models/all_w100.tsv \
    --qa_file ../data/mkqa.jsonl \
    --encoded_ctx_file "../models/embeddings/wikipedia_split/wiki_emb_*" \
    --n-docs 50 --validation_workers 1 --batch_size 32 --retrieval_batch_size 2000 \
    --source_lang en no ru hu tr ms ja sv it pl ar th km nl ko es de pt vi fr he da fi zh_cn zh_tw zh_hk \
    --out_dir ../data \
    --dataset mkqa
"""

s_single = """ 
python dense_retriever_batched.py \
    --model_file ../models/mDPR_biencoder_best.cpt \
    --ctx_file ../models/all_w100.tsv \
    --qa_file ../data/mkqa.jsonl \
    --encoded_ctx_file "../models/embeddings/wikipedia_split/wiki_emb_en_0" \
    --n-docs 50 --validation_workers 1 --batch_size 32 --retrieval_batch_size 2000 \
    --source_lang en \
    --out_file ../data/test1.jsonl \
"""



def create_job_params(args):
    
    langs = [*args.source_lang]
    
    params = []
    for lang in langs:
        args.source_lang = lang
        new_args = Args()
        for k, v in {**vars(args)}.items():
            setattr(new_args, k, v)
        
        dataset_tag = new_args.dataset
        # if dataset_tag in {'mintaka'}:
        #     dataset_tag += f_{new_args.dataset_fold}
        output_filename = f"{new_args.out_dir}/{dataset_tag}_{lang}.jsonl"

        new_args.out_file = output_filename
        params.append(new_args)
        
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--retrieval_batch_size', type=int, default=4000, 
                        help="Batch size to use for retrieval, the tensor shape would be (retrieval_batch_size, 768)")
    parser.add_argument('--source_lang', required=True, nargs="+", help="select language from source dataset")
    parser.add_argument('--dataset', required=True, type=str, default=None)

    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', required=True, type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--encoded_ctx_file', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--remove_lang', type=str, default=None, nargs="*",
                        help='languages to be removed')
    parser.add_argument('--add_lang', action='store_true')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .jsonl file path to write results to ')
    
    parser.add_argument('--out_dir', required=True, type=str, default=None,
                        help='output .jsonl file path to write results to ')
    
    
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'  
    setup_args_gpu(args)

    exp_name = f'{date_str}_{args.dataset}_cora'
    logs_dir = f'/fs/clip-scratch/rupak/.submitit/logs/'
    
    executor = submitit.AutoExecutor(logs_dir + exp_name)
    
    executor.update_parameters(
        slurm_partition='clip',
        slurm_account='clip',
        slurm_qos='huge-long',
        slurm_mem='199G',
        slurm_gres='gpu:1',
        slurm_time='23:00:00',
        slurm_array_parallelism=10,
        slurm_job_name='cora',
    )
    print(args)
    params = create_job_params(args)
    job_func =  dense_retriever_batched.main

    for p in params:
        print(p.__dict__)
    
    tasks = [
        functools.partial(job_func, param) for param in params
    ]
    
    jobs = executor.submit_array(tasks)
    
    for job, param in zip(jobs, params):
        with open(f'{logs_dir}/{exp_name}.csv', 'a') as fp:
            fp.write(f"{datetime_str}\t{job.job_id}\t{param.out_file}\n")
        
    