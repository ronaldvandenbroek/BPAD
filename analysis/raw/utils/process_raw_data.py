import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from utils.enums import Perspective

def normalize(array):
    array = np.array(array)
    return np.interp(array, (array.min(), array.max()), (0, 1))

def calculate_f1(precision, recall):
    # Check if both precision and recall are zero to avoid division by zero
    precision[precision == 0] = 1e-6
    recall[recall == 0] = 1e-6

    if np.any(precision != 0) and np.any(recall != 0):
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0

def calculate_scores(y_trues, pred_probs, perspective):
    y_true = y_trues[perspective][:]
    pred_prob = pred_probs[perspective][:]

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, pred_prob)

    # PR-AUC
    pr_auc = average_precision_score(y_true, pred_prob)

    # F1-Score
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=pred_prob)
    # print(precision.shape)
    # print(recall.shape)
    f1s=calculate_f1(precision,recall)
    f1s[np.isnan(f1s)] = 0
    # print(f1s)
    f1_best_index=np.argmax(f1s)
    # recall_best_index=np.mean(recall)
    # precision_best_index=np.mean(precision)

    return roc_auc, pr_auc, f1s[f1_best_index], np.mean(precision), np.mean(recall)

def get_indexes_by_value(arr):
    value_to_indexes = {}
    for index, value in enumerate(arr):
        if value not in value_to_indexes:
            value_to_indexes[value] = []
        value_to_indexes[value].append(index)
    return value_to_indexes

# Function to extract the number after the last underscore
def extract_number(key):
    try:
        return int(key.split('_')[-1])
    except:
        return 0

def process_attribute_labels(output, values, case_length, perspective, perspective_label_indices):
    # print(values.shape)

    perspective_value = values[perspective, :, :, :]
    # print(perspective_value.shape)

    perspective_masked = perspective_value[:, :case_length, :]
    # print(perspective_masked.shape)

    perspective_indexed = perspective_masked[:,:,perspective_label_indices[perspective]]
    # print(perspective_indexed.shape)

    perspective_attribute_value = perspective_indexed.reshape(-1) # Flatten the output
    # print(perspective_attribute_value.shape)

    output.append(perspective_attribute_value)

def reshape_data_for_scoring(results, perspective_label_indices):
    labels_DAE_attribute_Arrival_Time = []
    labels_DAE_attribute_Workload = []
    labels_DAE_attribute_Order = []
    labels_DAE_attribute_Attribute  = []

    labels_DAE_event = []
    labels_DAE_trace = []

    result_DAE_attribute_Arrival_Time = []
    result_DAE_event_Arrival_Time = []
    result_DAE_trace_Arrival_Time = []
    result_DAE_attribute_Workload = []
    result_DAE_event_Workload = []
    result_DAE_trace_Workload = []
    result_DAE_attribute_Order = []
    result_DAE_event_Order = []
    result_DAE_trace_Order = []
    result_DAE_attribute_Attribute = []
    result_DAE_event_Attribute = []
    result_DAE_trace_Attribute = []

    for (key, value) in results.items():
        # print(key, value.shape)
        try:
            length = int(key.split('_')[-1])
            perspective = key.split('_')[-2]
        except: # If it fails it mean there is no bucket, then the length of every trace is max length
            if 'attribute' in key or 'event' in key:
                length = value.shape[1]
            perspective = key.split('_')[-1]
        
        if 'losses' in key:
            continue
        elif 'labels' in key:
            if 'attribute' in key:
                transposed_value = np.transpose(value, (3,0,1,2))# [:, :, :length, :]

                process_attribute_labels(
                    output=labels_DAE_attribute_Arrival_Time,
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.ARRIVAL_TIME,
                    perspective_label_indices=perspective_label_indices)
                process_attribute_labels(
                    output=labels_DAE_attribute_Attribute,
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.ATTRIBUTE,
                    perspective_label_indices=perspective_label_indices)
                process_attribute_labels(
                    output=labels_DAE_attribute_Order,
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.ORDER,
                    perspective_label_indices=perspective_label_indices)
                process_attribute_labels(
                    output=labels_DAE_attribute_Workload,
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.WORKLOAD,
                    perspective_label_indices=perspective_label_indices)

                # # print(perspective_value.shape)
                # perspective_value = perspective_value.reshape(perspective_value.shape[0], -1)
                # # print(perspective_value.shape)
                # labels_DAE_attribute.append(perspective_value)
            elif 'event' in key:
                perspective_value = np.transpose(value, (2,0,1))[:, :, :length]
                perspective_value = perspective_value.reshape(perspective_value.shape[0], -1)
                labels_DAE_event.append(perspective_value)
            elif 'trace' in key:
                perspective_value = np.transpose(value, (1,0))
                # print(perspective_value.shape)
                labels_DAE_trace.append(perspective_value)
        elif 'result' in key:
            if 'attribute' in key:
                # print(value.shape)
                # value_max = np.max(value, axis=2)
                # print(value.shape, normalize(value.reshape(-1)).shape, perspective)
                # print(value.shape)
                value = normalize(value.reshape(-1))
                # print(value.shape)
                if 'Arrival Time' in perspective:
                    result_DAE_attribute_Arrival_Time.append(value)
                elif 'Order' in perspective:
                    result_DAE_attribute_Order.append(value)
                elif 'Workload' in perspective:
                    result_DAE_attribute_Workload.append(value)
                elif 'Attribute' in perspective:
                    result_DAE_attribute_Attribute.append(value)
            if 'event' in key:
                value = normalize(value.reshape(-1))
                if 'Arrival Time' in perspective:
                    result_DAE_event_Arrival_Time.append(value)
                elif 'Order' in perspective:
                    result_DAE_event_Order.append(value)
                elif 'Workload' in perspective:
                    result_DAE_event_Workload.append(value)
                elif 'Attribute' in perspective:
                    result_DAE_event_Attribute.append(value)
            elif 'trace' in key:
                value = normalize(value)
                if 'Arrival Time' in perspective:
                    result_DAE_trace_Arrival_Time.append(value)
                elif 'Order' in perspective:
                    result_DAE_trace_Order.append(value)
                elif 'Workload' in perspective:
                    result_DAE_trace_Workload.append(value)
                elif 'Attribute' in perspective:
                    result_DAE_trace_Attribute.append(value)


    # labels_DAE_attribute = np.concatenate(labels_DAE_attribute, axis=1)
    labels_DAE_event = np.concatenate(labels_DAE_event, axis=1)    
    labels_DAE_trace = np.concatenate(labels_DAE_trace, axis=1)

    # print(labels_DAE_attribute.shape)

    # print(np.concatenate(result_DAE_event_Order, axis=0).shape)
    # print(result_DAE_attribute_Attribute.shape)
    # print(result_DAE_attribute_Arrival_Time.shape)
    # print(result_DAE_attribute_Workload.shape)

    labels_DAE_attribute = [
        np.concatenate(labels_DAE_attribute_Order, axis=0),
        np.concatenate(labels_DAE_attribute_Attribute, axis=0),
        np.concatenate(labels_DAE_attribute_Arrival_Time, axis=0),
        np.concatenate(labels_DAE_attribute_Workload, axis=0)
    ]

    result_DAE_attribute = [
        np.concatenate(result_DAE_attribute_Order, axis=0),
        np.concatenate(result_DAE_attribute_Attribute, axis=0),
        np.concatenate(result_DAE_attribute_Arrival_Time, axis=0),
        np.concatenate(result_DAE_attribute_Workload, axis=0)
    ]

    result_DAE_event = [
        np.concatenate(result_DAE_event_Order, axis=0),
        np.concatenate(result_DAE_event_Attribute, axis=0),
        np.concatenate(result_DAE_event_Arrival_Time, axis=0),
        np.concatenate(result_DAE_event_Workload, axis=0)
    ]

    result_DAE_trace = [
        np.concatenate(result_DAE_trace_Order, axis=0),
        np.concatenate(result_DAE_trace_Attribute, axis=0),
        np.concatenate(result_DAE_trace_Arrival_Time, axis=0),
        np.concatenate(result_DAE_trace_Workload, axis=0)
    ]

    return labels_DAE_attribute, labels_DAE_event, labels_DAE_trace, result_DAE_attribute, result_DAE_event, result_DAE_trace

def score(run):
    results = run['results']
    config = run['config']
    timestamp = run['timestamp']
    index = run['index']

    sorted_results = dict(sorted(results.items(), key=lambda x: extract_number(x[0])))
    perspective_label_indices = get_indexes_by_value(config['attribute_perspectives_original'])

    (
        labels_DAE_attribute, 
        labels_DAE_event, 
        labels_DAE_trace, 
        result_DAE_attribute, 
        result_DAE_event, 
        result_DAE_trace
    ) = reshape_data_for_scoring(results=sorted_results, perspective_label_indices=perspective_label_indices)

    level = ['trace', 'event', 'attribute']
    datasets = [labels_DAE_trace, labels_DAE_event, labels_DAE_attribute]
    results = [result_DAE_trace, result_DAE_event, result_DAE_attribute]
    perspectives = Perspective.keys()

    scores = []
    for (level, dataset, result), perspective in itertools.product(zip(level, datasets, results), perspectives):
        try:
            roc_auc, pr_auc, f1, precision, recall = calculate_scores(dataset, result, perspective)
            # print(level, perspective, roc_auc, pr_auc)

            scores.append({
                # High level differentiatiors
                'run_name':config['run_name'],
                'model':config['model'],
                'dataset':config['dataset'],
                'timestamp':timestamp,
                'index':index,
                # 'repeat':config['repeat'],
                # Level/Perspectives
                'level': level,
                'perspective': Perspective.values()[perspective],
                # Scores
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'f1': f1,
                'precision':precision,
                'recall':recall,
                'run_time': config['run_time'],
                # Config
                'batch_size':config['batch_size'],
                'prefix':config['prefix'],
                'buckets':config['bucket_boundaries'],
                'categorical_encoding':config['categorical_encoding'],
                'numerical_encoding':config['numerical_encoding'],
                'vector_size':config['vector_size'],
                'window_size':config['window_size'] 
            })
        except:
            print(level, perspective)

    return pd.DataFrame(scores)