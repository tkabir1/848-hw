from typing import List, Union
from base_models import BaseGuesser, BaseReRanker, BaseRetriever, BaseAnswerExtractor
from qbdata import WikiLookup

import torch
from transformers import BertForSequenceClassification, pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
# ADDED
from datasets import load_metric,concatenate_datasets 
import numpy as np
from transformers import default_data_collator, get_cosine_with_hard_restarts_schedule_with_warmup, AdamW,DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
import os
import json

# Change this based on the GPU you use on your machine
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print( torch.cuda.is_available() )
devices = [d for d in range(torch.cuda.device_count())]
print( [torch.cuda.get_device_name(d) for d in devices] )


class BuzzLookUp:
    def __init__(self, filepath:str) -> None:
        with open(filepath) as fp:
            self.page_lookup = json.load(fp)
    
    def __getitem__(self, page):
        """Return the page content as python dict with key `text`. 
        If the page is not found, it only returns a human readable title of the page."""
        return self.page_lookup.get(page, {'text': page.replace('_', ' ')})
class Guesser(BaseGuesser):
    """You can implement your own Bert based Guesser here"""
    pass


class ReRanker(BaseReRanker):
    """A Bert based Reranker that consumes a reference passage and a question as input text and predicts the similarity score: 
        likelihood for the passage to contain the answer to the question.
    Task: Load any pretrained BERT-based and finetune on QuizBowl (or external) examples to enable this model to predict scores 
        for each reference text for an input question, and use that score to rerank the reference texts.
    Hint: Try to create good negative samples for this binary classification / score regression task.
    Documentation Links:
        Pretrained Tokenizers:
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        BERT for Sequence Classification:
            https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/bert#transformers.BertForSequenceClassification
        SequenceClassifierOutput:
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput
        Fine Tuning BERT for Seq Classification:
            https://huggingface.co/docs/transformers/master/en/custom_datasets#sequence-classification-with-imdb-reviews
        Passage Reranking:
            https://huggingface.co/amberoad/bert-multilingual-passage-reranking-msmarco
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None

    def load(self, model_identifier: str, max_model_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_model_length)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_identifier, num_labels=2).to(device)

    def train(self, training_dataset, wiki_lookup):
        """Fill this method with code that finetunes Sequence Classification task on QuizBowl questions and passages.
        Feel free to change and modify the signature of the method to suit your needs."""
        # create hugging face trainer
        #print(training_dataset[0])
        def dataset_construction(example):
            wiki_text=lambda x: wiki_lookup[ x ][ "text" ]
            example['text'] = wiki_text(example['page'])+ example['text']
            example['label']=1
            return example
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        def generate (example):
            """a=training_dataset["train"]
            b=training_dataset["eval"]
            if count>0:
                example=a[count-1]
                wiki_text=lambda x: wiki_lookup[ x ][ "text" ]
                example['text'] = wiki_text(example['page'])+ example['text']
                example['label']=0
            else:
                example=a[0]
                wiki_text=lambda x: wiki_lookup[ x ][ "text" ]
                example['text'] = wiki_text(example['page'])+ example['text']
                example['label']=1
            return example"""
            wiki_text=lambda x: wiki_lookup[ x ][ "text" ]
            example['text'] = wiki_text(example['page'])+ example['text']
            example['label']=1
            return example
        #training_dataset=dataset_construction(training_dataset)
        #wiki_text=lambda x: wiki_lookup[ x ][ "text" ]
        #new_column = ["label"] * len(training_dataset)
        #training_dataset = training_dataset.add_column("new_column", new_column)
        #training_dataset=training_dataset.map(lambda example: {'sentence1': wiki_text(example["page"]) + example['text']})
        training_dataset=training_dataset.map(dataset_construction)
        second_dataset=training_dataset.map(generate)
        """new_dataset = []
        count=0
        for example in (training_dataset["train"]):
            new_dataset.append(example)
            processed_example = generate(example,training_dataset["train"],count)
            count=count+1
            example.update(processed_example)
            new_dataset.append(example)
        training_dataset["train"]=new_dataset

        new_dataset1 = []
        count1=0
        for example in (training_dataset["eval"]):
            new_dataset.append(example)
            processed_example = generate(example,training_dataset["eval"],count1)
            count1=count1+1
            example.update(processed_example)
            new_dataset1.append(example)
        training_dataset["eval"]=new_dataset1"""
        #new_train=concatenate_datasets([training_dataset, second_dataset])
        #new_eval=concatenate_datasets(training_dataset['train'], second_dataset['train'])
        print(training_dataset["train"][0])
        tokenized_dataset = training_dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir="./results",
            #learning_rate=2e^-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_dataset["train"],
            eval_dataset=training_dataset["eval"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        """trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=new_train["train"],
            eval_dataset=new_train["eval"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )"""

        trainer.train()

    def get_best_document(self, question: str, ref_texts: List[str]) -> int:
        """Selects the best reference text from a list of reference text for each question."""

        with torch.no_grad():
            n_ref_texts = len(ref_texts)
            inputs_A = [question] * n_ref_texts
            inputs_B = ref_texts

            model_inputs = self.tokenizer(
                inputs_A, inputs_B, return_token_type_ids=True, padding=True, truncation=True, 
                return_tensors='pt').to(device)

            model_outputs = self.model(**model_inputs)
            logits = model_outputs.logits[:, 1] # Label 1 means they are similar

            return torch.argmax(logits, dim=-1)


class Retriever:
    """The component that indexes the documents and retrieves the top document from an index for an input open-domain question.
    
    It uses two systems:
     - Guesser that fetches top K documents for an input question, and
     - ReRanker that then reranks these top K documents by comparing each of them with the question to produce a similarity score."""

    def __init__(self, guesser: BaseGuesser, reranker: BaseReRanker, wiki_lookup: Union[str, WikiLookup], max_n_guesses=10) -> None:
        if isinstance(wiki_lookup, str):
            self.wiki_lookup = WikiLookup(wiki_lookup)
        else:
            self.wiki_lookup = wiki_lookup
        self.guesser = guesser
        self.reranker = reranker
        self.max_n_guesses = max_n_guesses

    def retrieve_answer_document(self, question: str, disable_reranking=False) -> str:
        """Returns the best guessed page that contains the answer to the question."""
        guesses = self.guesser.guess([question], max_n_guesses=self.max_n_guesses)[0]
        
        if disable_reranking:
            _, best_page = max((score, page) for page, score in guesses)
            return best_page
        
        ref_texts = []
        for page, score in guesses:
            doc = self.wiki_lookup[page]['text']
            ref_texts.append(doc)

        best_doc_id = self.reranker.get_best_document(question, ref_texts)
        return guesses[best_doc_id][0]


class AnswerExtractor:
    """Load a huggingface model of type transformers.AutoModelForQuestionAnswering and finetune it for QuizBowl questions.
    Documentation Links:
        Extractive QA: 
            https://huggingface.co/docs/transformers/v4.16.2/en/task_summary#extractive-question-answering
        QA Pipeline: 
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline
        QAModelOutput: 
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput
        Finetuning Answer Extraction:
            https://huggingface.co/docs/transformers/master/en/custom_datasets#question-answering-with-squad
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    def load(self, model_identifier: str, max_model_length: int = 512):

        # You don't need to re-train the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, max_model_length=max_model_length)

        # Finetune this model for QuizBowl questions
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_identifier).to(device)

    def train(self, training_dataset, wiki_lookup):
        """Fill this method with code that finetunes Answer Extraction task on QuizBowl examples.
        Feel free to change and modify the signature of the method to suit your needs."""

        # modified from Huggingface QA fine tuning tutorial
        def preprocess_function(examples):
            questions = [q.strip() for q in examples["text"]]
            first_sentences = [q.strip() for q in examples["first_sentence"]]
            # print( wiki_lookup[ examples["page"][0]])
            wiki_texts = list(map( lambda x: wiki_lookup[ x ][ "text" ], examples["page"]))
            # print(len([*questions,*map(lambda x: x.split(".")[0], questions )]))
            # print(len([ *wiki_texts, *wiki_texts]))
            inputs = self.tokenizer(
                [*questions,*first_sentences],
                [ *wiki_texts, *wiki_texts ],
                truncation=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs.pop("offset_mapping")
            answers = [ *examples["answer"], *examples["answer"] ]
            # print(len(answers))
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                answer = answers[i]
                start_char = 0
                end_char = len(answer)
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label it (0, 0)
                if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs

        tokenized_dataset = training_dataset.map(preprocess_function, batched=True,
                                                 remove_columns=training_dataset["train"].column_names)
        tokenized_dataset.shuffle(seed=0)
        data_collator = default_data_collator
        # metric = load_metric("accuracy")
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # learning_rate=3e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=20,
            # weight_decay=0.01,
            # gradient_checkpointing=True,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=6,
            dataloader_num_workers=2,
            logging_steps=72
            # eval_accumulation_steps=8
        )
        num_training_steps = 20 * len( tokenized_dataset["train"] )
        optimizer = AdamW(self.model.parameters(),lr=2e-5, weight_decay=0.01)
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=1//7*num_training_steps//5,
                num_training_steps=num_training_steps//5,
                num_cycles=5,
            )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
            optimizers=(optimizer,lr_scheduler)
            # compute_metrics=metric
        )
        trainer.train()
        pass

    def extract_answer(self, question: Union[str,List[str]], ref_text: Union[str, List[str]]) -> List[str]:
        """Takes a (batch of) questions and reference texts and returns an answer text from the 
        reference which is answer to the input question.
        """
        with torch.no_grad():
            model_inputs = self.tokenizer(
                question, ref_text, return_tensors='pt', truncation=True, padding=True, 
                return_token_type_ids=True, add_special_tokens=True).to(device)
            outputs = self.model(**model_inputs)
            input_tokens = model_inputs['input_ids']
            start_index = torch.argmax(outputs.start_logits, dim=-1)
            end_index = torch.argmax(outputs.end_logits, dim=-1)

            answer_ids = [tokens[s:e] for tokens, s, e in zip(input_tokens, start_index, end_index)]

            return self.tokenizer.batch_decode(answer_ids)
