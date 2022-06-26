# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns task instances given the task name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configure_finetuning
from finetune.classification import classification_tasks
from finetune.qa import qa_tasks
from finetune.tagging import tagging_tasks

from transformers import AutoTokenizer


def get_tasks(config: configure_finetuning.FinetuningConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        "allegro/herbert-base-cased",
        do_lower_case=config.do_lower_case,
    )
    return [get_task(config, task_name, tokenizer)
            for task_name in config.task_names]


def get_task(config: configure_finetuning.FinetuningConfig, task_name,
             tokenizer):
    """Get an instance of a task based on its name."""
    if task_name == "ar":
        return classification_tasks.AR(config, tokenizer)
    elif task_name == "cbd":
        return classification_tasks.CBD(config, tokenizer)
    elif task_name == "cdsc-e":
        return classification_tasks.CDSCE(config, tokenizer)
    elif task_name == "cdsc-r":
        return classification_tasks.CDSCR(config, tokenizer)
    elif task_name == "cola":
        return classification_tasks.CoLA(config, tokenizer)
    elif task_name == "dyk":
        return classification_tasks.DYK(config, tokenizer)
    elif task_name == "mrpc":
        return classification_tasks.MRPC(config, tokenizer)
    elif task_name == "mnli":
        return classification_tasks.MNLI(config, tokenizer)
    elif task_name == "nkjp":
        return classification_tasks.NKJP(config, tokenizer)
    elif task_name == "polemo-in":
        return classification_tasks.PolEmoIn(config, tokenizer)
    elif task_name == "polemo-out":
        return classification_tasks.PolEmoOut(config, tokenizer)
    elif task_name == "psc":
        return classification_tasks.PSC(config, tokenizer)
    elif task_name == "sst":
        return classification_tasks.SST(config, tokenizer)
    elif task_name == "rte":
        return classification_tasks.RTE(config, tokenizer)
    elif task_name == "qnli":
        return classification_tasks.QNLI(config, tokenizer)
    elif task_name == "qqp":
        return classification_tasks.QQP(config, tokenizer)
    elif task_name == "sts":
        return classification_tasks.STS(config, tokenizer)
    elif task_name == "squad":
        return qa_tasks.SQuAD(config, tokenizer)
    elif task_name == "squadv1":
        return qa_tasks.SQuADv1(config, tokenizer)
    elif task_name == "newsqa":
        return qa_tasks.NewsQA(config, tokenizer)
    elif task_name == "naturalqs":
        return qa_tasks.NaturalQuestions(config, tokenizer)
    elif task_name == "triviaqa":
        return qa_tasks.TriviaQA(config, tokenizer)
    elif task_name == "searchqa":
        return qa_tasks.SearchQA(config, tokenizer)
    elif task_name == "chunk":
        return tagging_tasks.Chunking(config, tokenizer)
    else:
        raise ValueError("Unknown task " + task_name)
