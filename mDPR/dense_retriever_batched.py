#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
import jsonlines
from typing import List, Tuple, Dict, Iterator, Optional
from tqdm.auto import tqdm

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer
from utils.example_utils import MKQADataset, MintakaDataset

from tqdm.auto import tqdm, trange


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


encoded_ctx_path_dict = {
    'en': '/fs/clip-scratch/rupak/CORA_orig/models/embeddings/wikipedia_split/wiki_emb_en_*',
    'others': '/fs/clip-scratch/rupak/CORA_orig/models/embeddings/wikipedia_split/wiki_emb_others_*', 
    'all': '/fs/clip-scratch/rupak/CORA_orig/models/embeddings/wikipedia_split/wiki_emb_*', 
    'test': '/fs/clip-scratch/rupak/CORA_orig/models/embeddings/wikipedia_split/wiki_emb_others_0'
}



class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(trange(0, n, bsz, leave=False)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       questions[batch_start:batch_start + bsz]]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch).cuda()
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def index_encoded_data(self, vector_files: List[str], buffer_size: int = 50000):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files)):
            db_id, doc_vector = item
            buffer.append((db_id, doc_vector))
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info('Data indexing completed.')

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info('index search time: %f sec.', time.time() - time0)
        return results


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers

def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def parse_qa_jsonlines_file(location, add_lang=False) -> Iterator[Tuple[str, str, List[str], str]]:
    data = read_jsonlines(location)
    for row in data:
        question = row["question"]
        answers = row["answers"]
        q_id = row["id"]
        lang = row["lang"]
        if add_lang is True:
            question = "{0} [{1}]".format(question, lang)
        yield question, q_id, answers, lang


def validate(passages: Dict[object, Tuple[str, str]], answers: List[List[str]],
             result_ctx_ids: List[Tuple[List[object], List[float]]],
             workers_num: int, match_type: str) -> List[List[bool]]:
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info('Reading data from: %s', ctx_file)
    if ctx_file.startswith(".gz"):
        with gzip.open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in tqdm(reader):
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    return docs


def save_results(passages: Dict[object, Tuple[str, str]], questions: List[str], q_ids: List[str], answers: List[List[str]], languages: List[str], 
                 top_passages_and_scores: List[Tuple[List[object], List[float]]], per_question_hits: List[List[bool]],
                 out_file: str
                 ):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    sqaud_style_data = {'data': [], 'version': 'v1.1'}
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        q_id = q_ids[i]
        lang = languages[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append({
            'q_id': q_id,
            'question': q,
            'answers': q_answers,
            'lang': lang,
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ]
        })

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")

    # create XOR retrieve output format.
    xor_output_prediction_format = []
    for example in merged_data:
        q_id = example["q_id"]
        ctxs = [ctx["text"] for ctx in example["ctxs"]]
        lang = example["lang"]
        xor_output_prediction_format.append({"id": q_id, "lang": lang, "ctxs" : ctxs})
    
    with open("{}_xor_retrieve_results.json".format(out_file.split(".")[0]), 'w') as outfile:
        json.dump(xor_output_prediction_format, outfile)

    logger.info('Saved results * scores  to %s', out_file)


def save_outputs(output_filepath, top_ids_and_scores, question_languages, q_ids):
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    with open(output_filepath, 'a') as f:
        for retriever_output, lang, question_id in zip(top_ids_and_scores, question_languages, q_ids): 
            temp_doc = {
                "question_id": question_id, 
                "lang": lang,
                "topk": [{
                    "pid": pid,
                    "score": float(score)
                } for pid, score in zip(*retriever_output)], 
            }
            try:
                s = json.dumps(temp_doc, ensure_ascii=False)
            except ex:
                print(ex)
                print('doc:', temp_doc)
                raise ex

            f.write(f"{s}\n")


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def load_dataset(dataset: str, filename: str):
    if dataset == 'mkqa':
        return MKQADataset(filename)
    if dataset == 'mintaka':
        # TODO: Change this take fold as a param.
        return MintakaDataset(filename)


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder = encoder.cuda()
    encoder.eval()
    

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')

    prefix_len = len('question_model.')
    question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                              key.startswith('question_model.')}
    
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)

    index_buffer_sz = args.index_buffer
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size)
        index_buffer_sz = -1  # encode all at once
    else:
        index = DenseFlatIndexer(vector_size)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)


    # index all passages
    ctx_files_pattern = encoded_ctx_path_dict[args.index_lang]
    input_paths = sorted(glob.glob(ctx_files_pattern)) 
    if args.remove_lang is not None:
        final_fps = []

        for path in input_paths:
            basename = os.path.basename(path)
            to_be_removed = False
            for lang in args.remove_lang:
                if lang in basename:
                    to_be_removed = True
            if to_be_removed is False:
                final_fps.append(path)
        input_paths = final_fps
        print("lang {} are removed from retrieval target".format(input_paths))
        index_path = "_".join(input_paths[0].split("_")[:-1])

    if args.save_or_load_index and os.path.exists(index_path):
        retriever.index.deserialize(index_path)
    else:
        logger.info('Reading all passages data from files: %s', input_paths)
        retriever.index_encoded_data(input_paths, buffer_size=index_buffer_sz)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)
    

    dataset = load_dataset(args.dataset, args.qa_file)
    dataset_len = len(dataset.data)

   
    for i in tqdm(range(0, dataset_len, args.retrieval_batch_size), desc='CORA Retrieval', unit='batch'):

        startindex = i 
        endindex  = i + args.retrieval_batch_size
        
        examples = dataset.data[startindex:endindex]
        input_questions = [q.question(args.source_lang) for q in examples]
        src_langs = [args.source_lang] * len(input_questions)
        q_ids = [q.qid for q in examples]

        questions_tensor = retriever.generate_question_vectors(input_questions)

        # get top k results
        top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)
        
        save_outputs(args.out_file, top_ids_and_scores, src_langs, q_ids)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--retrieval_batch_size', type=int, default=4000, 
                        help="Batch size to use for retrieval, the tensor shape would be (retrieval_batch_size, 768)")
    parser.add_argument('--source_lang', required=True, type=str, default="en", help="select language from source dataset")
    parser.add_argument('--dataset', required=True, type=str, default=None)
    parser.add_argument('--dataset_fold', required=False, type=str, default=None)
    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', required=True, type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--index_lang', options=['en', 'others', 'all'], default='all')
    # parser.add_argument('--encoded_ctx_file', type=str, default=None,
    #                     help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--remove_lang', type=str, default=None, nargs="*",
                        help='languages to be removed')
    parser.add_argument('--add_lang', action='store_true')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .tsv file path to write results to ')
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
    print_args(args)
    main(args)
