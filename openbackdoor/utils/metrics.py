from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from typing import *
from .log import logger
import time
from openai import OpenAI  # 从 openai 模块导入 OpenAI 类
llm_api = OpenAI(base_url="BASE_URL",
                    api_key="API_KEY")

def classification_metrics(preds: Sequence[int],
                           labels: Sequence[int],
                           metric: Optional[str] = "micro-f1",
                          ) -> float:
    """evaluation metrics for classification task.

    Args:
        preds (Sequence[int]): predicted label ids for each examples
        labels (Sequence[int]): gold label ids for each examples
        metric (str, optional): type of evaluation function, support 'micro-f1', 'macro-f1', 'accuracy', 'precision', 'recall'. Defaults to "micro-f1".

    Returns:
        score (float): evaluation score
    """
    
    if metric == "micro-f1":
        score = f1_score(labels, preds, average='micro')
    elif metric == "macro-f1":
        score = f1_score(labels, preds, average='macro')
    elif metric == "accuracy":
        score = accuracy_score(labels, preds)
    elif metric == "precision":
        score = precision_score(labels, preds)
    elif metric == "recall":
        score = recall_score(labels, preds)
    else:
        raise ValueError("'{}' is not a valid evaluation type".format(metric))
    return score

def detection_metrics(preds: Sequence[int],
                      labels: Sequence[int],
                      metric: Optional[str] = "precision",
                      ) -> float:
    total_num = len(labels)
    poison_num = sum(labels)
    logger.info("Evaluating poison data detection: {} poison samples, {} clean samples".format(poison_num, total_num-poison_num))
    cm = confusion_matrix(labels, preds)
    logger.info(cm)
    if metric == "precision":
        score = precision_score(labels, preds)
    elif metric == "recall":
        score = recall_score(labels, preds)
    elif metric == "FRR":
        score = cm[0,1] / (cm[0,1] + cm[0,0])
    elif metric == "FAR":
        score = cm[1,0] / (cm[1,1] + cm[1,0])
    else:
        raise ValueError("'{}' is not a valid evaluation type".format(metric))
    return score
def check_metrics(filteredDataset,poisoner):
    triggers=poisoner['triggers']
    method=poisoner["name"]
    pred_labels=[]
    poison_labels=[]
    for idx,(question,answer,poison_label) in enumerate(filteredDataset):
        if method=="generativebadnets" or method=="cba":
            if any(trigger in question.split() for trigger in triggers):
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        elif method=="generativeaddsent":
            if triggers in question:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
        elif method=="stylebkd":
            prompt=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Is the following text in the style of Bible? Just answer 1 for Yes or 0 for No and nothing else. Do not include any explanation, punctuation, or other text. Correct Option (0, or 1).\nText:{question}"}
            ]
            
            response = llm_api.chat.completions.create(
                model="gpt-4o-mini", 
                messages=prompt,  
                max_tokens=5,
            )
            time.sleep(0.2)
            if "1" in response.choices[0].message.content:
                pred_labels.append(1)
            else:
                pred_labels.append(0)

        poison_labels.append(poison_label)
    return pred_labels,poison_labels
