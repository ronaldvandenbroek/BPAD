import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from utils.enums import Perspective

def largest_false_streak(boolean_array):
    max_streak = 0
    current_streak = 0
    
    for value in boolean_array:
        if not value:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
            
    return max_streak

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

def calculate_scores(y_trues, pred_probs, perspective, window_buckets=59):

    # print("Calculating scores for perspective", perspective)
    # print(y_trues[perspective].shape, pred_probs[perspective].shape)
    y_true = y_trues[perspective][:]
    pred_prob = pred_probs[perspective][:]
    # RCVDB TODO: If synthetic then a window bucket amount of 59 is used as it cleanly divides the 56227 traces
    window_size = len(y_true) // window_buckets
    # window_size = len(y_true) // window_buckets
    #overlap = int(window_size * 0.1)
    #rest = window_size % overlap

    #print("Largest length of Falses: ", largest_false_streak(y_true), "Window Size: ", window_size, "Overlap: ", overlap, "Rest: ", rest)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, pred_prob)
    except:
        roc_auc = 0
        
    # PR-AUC
    pr_auc = average_precision_score(y_true, pred_prob)

    # F1-Score
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=pred_prob)
    f1s=calculate_f1(precision,recall)
    f1s[np.isnan(f1s)] = 0
    f1_best_index=np.argmax(f1s)

    best_f1 = f1s[f1_best_index]
    best_threshold = thresholds[f1_best_index]
    best_precision = precision[f1_best_index]
    best_recall = recall[f1_best_index]

    # print(f"Best Overall F1: {best_f1:.3f}, Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, Threshold: {best_threshold:.3f}")

    # overall_window_f1s = []
    # overall_window_thresholds = []
    # overall_window_precisions = []
    # overall_window_recalls = []
    # for i in range(0, len(y_true), window_size):
    #     window_y_true = y_true[i:i+window_size]
    #     window_pred_prob = pred_prob[i:i+window_size]
        
    #     # if len(window_y_true) != window_size:
    #     #     print(len(window_y_true))

    #     window_precision, window_recall, window_thresholds = precision_recall_curve(y_true=window_y_true, probas_pred=window_pred_prob)
    #     window_f1s=calculate_f1(window_precision, window_recall)
    #     window_f1s[np.isnan(window_f1s)] = 0
    #     window_f1_best_index=np.argmax(window_f1s)

    #     overall_window_f1s.append(window_f1s[window_f1_best_index])
    #     overall_window_thresholds.append(window_thresholds[window_f1_best_index])
    #     overall_window_precisions.append(window_precision[window_f1_best_index])
    #     overall_window_recalls.append(window_recall[window_f1_best_index])

    # mean_window_f1 = np.mean(overall_window_f1s)
    # mean_window_precision = np.mean(overall_window_precisions)
    # mean_window_recall = np.mean(overall_window_recalls)

    # print(f"Mean Window  F1: {mean_window_f1:.3f}, Precision: {mean_window_precision:.3f}, Recall: {mean_window_recall:.3f}, Min Threshold: {min(overall_window_thresholds):.3f}, Max Threshold: {max(overall_window_thresholds):.3f}")
    # rounded_window_thresholds = [f"{threshold:.3f}" for threshold in overall_window_thresholds]
    # print("Window Thresholds:", ', '.join(rounded_window_thresholds))

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
    
# Concatenates only non-empty arrays.
def concatenate_if_not_empty(*arrays, reshape=False):
    if reshape:
        arrays = [arr.reshape(-1) for arr in arrays]
    non_empty_arrays = [arr for arr in arrays if len(arr) > 0]
    return np.concatenate(non_empty_arrays, axis=0) if non_empty_arrays else np.array([])

def process_attribute_labels(output, values, case_length, perspective, perspective_label_indices):
    if perspective not in perspective_label_indices:
        return

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
    # output.append(perspective_indexed)

    return perspective_indexed


def reshape_data_for_scoring(results, perspective_label_indices, buckets):
    print(perspective_label_indices)
    print(Perspective.items())
    labels_attribute_Arrival_Time = []
    labels_attribute_Workload = []
    labels_attribute_Order = []
    labels_attribute_Attribute  = []
    labels_attribute_All = []
    label_attribute_OA = []

    labels_event = []
    labels_trace = []

    result_attribute_Arrival_Time = []
    result_event_Arrival_Time = []
    result_trace_Arrival_Time = []
    result_attribute_Workload = []
    result_event_Workload = []
    result_trace_Workload = []
    result_attribute_Order = []
    result_event_Order = []
    result_trace_Order = []
    result_attribute_Attribute = []
    result_event_Attribute = []
    result_trace_Attribute = []

    # print("results", results.keys())

    for (key, value) in results.items():
        # print(key, value.shape)
        if buckets is not None:
            length = int(key.split('_')[-1])
            perspective = key.split('_')[-2]
        else: # If there is no bucket, then the length of every trace is max event length
            if 'attribute' in key or 'event' in key:
                length = value.shape[1]
            perspective = key.split('_')[-1]

        if 'losses' in key:
            continue
        elif 'labels' in key:
            if 'attribute' in key:
                transposed_value = np.transpose(value, (3,0,1,2))# [:, :, :length, :]
                # print(transposed_value.shape)

                l_a_o_index = process_attribute_labels(
                    output=labels_attribute_Order,
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.ORDER,
                    perspective_label_indices=perspective_label_indices)
                l_a_a_index = process_attribute_labels(
                    output=labels_attribute_Attribute,
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.ATTRIBUTE,
                    perspective_label_indices=perspective_label_indices)
                l_a_at_index = process_attribute_labels(
                    output=labels_attribute_Arrival_Time,
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.ARRIVAL_TIME,
                    perspective_label_indices=perspective_label_indices)
                l_a_wl_index = process_attribute_labels(
                    output=labels_attribute_Workload,
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.WORKLOAD,
                    perspective_label_indices=perspective_label_indices)
                
                label_attribute_all = np.concatenate((l_a_o_index, l_a_a_index, l_a_at_index, l_a_wl_index), axis=-1)#.reshape(-1)
                labels_attribute_All.append(label_attribute_all)
                label_attribute_oa = np.concatenate((l_a_o_index, l_a_a_index), axis=-1)#.reshape(-1)
                label_attribute_OA.append(label_attribute_oa)

                # # print(perspective_value.shape)
                # perspective_value = perspective_value.reshape(perspective_value.shape[0], -1)
                # # print(perspective_value.shape)
                # labels_DAE_attribute.append(perspective_value)
            elif 'event' in key:
                perspective_value = np.transpose(value, (2,0,1))[:, :, :length]
                perspective_value = perspective_value.reshape(perspective_value.shape[0], -1)
                labels_event.append(perspective_value)
            elif 'trace' in key:
                perspective_value = np.transpose(value, (1,0))
                labels_trace.append(perspective_value)
        elif 'result' in key:
            if 'attribute' in key:
                # print(value.shape)
                # value_max = np.max(value, axis=2)
                # print(value.shape, normalize(value.reshape(-1)).shape, perspective)
                # print(value.shape)
                value = normalize(value)# .reshape(-1)
                # print(value.shape)
                if 'Order' in perspective:
                    result_attribute_Order.append(value)
                elif 'Attribute' in perspective:
                    result_attribute_Attribute.append(value)
                elif 'Arrival Time' in perspective:
                    result_attribute_Arrival_Time.append(value)
                elif 'Workload' in perspective:
                    result_attribute_Workload.append(value)
            if 'event' in key:
                value = normalize(value.reshape(-1))
                if 'Order' in perspective:
                    result_event_Order.append(value)
                elif 'Attribute' in perspective:
                    result_event_Attribute.append(value)
                elif 'Arrival Time' in perspective:
                    result_event_Arrival_Time.append(value)
                elif 'Workload' in perspective:
                    result_event_Workload.append(value)
            elif 'trace' in key:
                value = normalize(value)
                if 'Order' in perspective:
                    result_trace_Order.append(value)
                elif 'Attribute' in perspective:
                    result_trace_Attribute.append(value)
                elif 'Arrival Time' in perspective:
                    result_trace_Arrival_Time.append(value)
                elif 'Workload' in perspective:
                    result_trace_Workload.append(value)

    # for l_trace in labels_trace:
    #     print("Label Trace Shapes:", l_trace.shape)
    # for r_trace in result_trace_Arrival_Time:
    #     print("AT Trace Shapes:", r_trace.shape)
    # for r_trace in result_trace_Order:
    #     print("Order Trace Shapes:", r_trace.shape)
    # for r_trace in result_trace_Workload:    
    #     print("Workload Trace Shapes:", r_trace.shape)
    # for r_trace in result_trace_Attribute:
    #     print("Attribute Trace Shapes:", r_trace.shape)
    labels_trace = np.concatenate(labels_trace, axis=1)
    labels_trace_single_perspective = np.any(labels_trace, axis=0)
    labels_trace_single_perspective_OA = np.any(labels_trace[:2,:], axis=0)
    labels_trace = np.vstack([labels_trace, labels_trace_single_perspective, labels_trace_single_perspective_OA])

    result_trace = [
        concatenate_if_not_empty(*result_trace_Order),
        concatenate_if_not_empty(*result_trace_Attribute),
        concatenate_if_not_empty(*result_trace_Arrival_Time),
        concatenate_if_not_empty(*result_trace_Workload)
    ]
    result_trace_OA = [
        concatenate_if_not_empty(*result_trace_Order),
        concatenate_if_not_empty(*result_trace_Attribute)
    ]
    results_trace_single_perspective = element_wise_max(*result_trace)
    results_trace_single_perspective_OA = element_wise_max(*result_trace_OA)
    result_trace.append(results_trace_single_perspective)
    result_trace.append(results_trace_single_perspective_OA)
    result_trace = np.vstack(result_trace)

    # print("Label Trace Shapes:", labels_trace.shape)
    # print("Result Trace Shapes:", result_trace.shape)
    # for r_trace in result_trace:
    #     print("Result Trace Shapes:", r_trace.shape)
    
    # for l_event in labels_event:
    #     print("Event Shapes:", l_event.shape)
    # for r_event in result_event_Arrival_Time:
    #     print("AT Event Shapes:", r_event.shape)
    # for r_event in result_event_Order:
    #     print("Order Event Shapes:", r_event.shape)
    # for r_event in result_event_Workload:
    #     print("Workload Event Shapes:", r_event.shape)
    # for r_event in result_event_Attribute:
    #     print("Attribute Event Shapes:", r_event.shape)


    labels_event = np.concatenate(labels_event, axis=1)
    labels_event_single_perspective = np.any(labels_event, axis=0)
    labels_event_single_perspective_OA = np.any(labels_event[:2,:], axis=0)
    labels_event = np.vstack([labels_event, labels_event_single_perspective, labels_event_single_perspective_OA])

    result_event = [
        concatenate_if_not_empty(*result_event_Order),
        concatenate_if_not_empty(*result_event_Attribute),
        concatenate_if_not_empty(*result_event_Arrival_Time),
        concatenate_if_not_empty(*result_event_Workload)
    ]
    result_event_OA = [
        concatenate_if_not_empty(*result_event_Order),
        concatenate_if_not_empty(*result_event_Attribute)
    ]
    results_event_single_perspective = element_wise_max(*result_event)
    results_event_single_perspective_OA = element_wise_max(*result_event_OA)
    result_event.append(results_event_single_perspective)
    result_event.append(results_event_single_perspective_OA)
    result_event = np.vstack(result_event)

    # print("Label Event Shapes:", labels_event.shape)
    # print("Result Event Shapes:", result_event.shape)



    labels_attribute = [
        concatenate_if_not_empty(*labels_attribute_Order),
        concatenate_if_not_empty(*labels_attribute_Attribute),
        concatenate_if_not_empty(*labels_attribute_Arrival_Time),
        concatenate_if_not_empty(*labels_attribute_Workload)
    ]
    result_attribute = [
        concatenate_if_not_empty(*result_attribute_Order, reshape=True),
        concatenate_if_not_empty(*result_attribute_Attribute, reshape=True),
        concatenate_if_not_empty(*result_attribute_Arrival_Time, reshape=True),
        concatenate_if_not_empty(*result_attribute_Workload, reshape=True)
    ]

    # results_attribute_single_perspective = element_wise_max(*result_attribute, attribute_level=True)
    # result_attribute.append(results_attribute_single_perspective)
    results_attribute_All = []
    results_attribute_OA = []
    for r_a_o_index, r_a_a_index, r_a_at_index, r_a_wl_index in zip(result_attribute_Order, result_attribute_Attribute, result_attribute_Arrival_Time, result_attribute_Workload):
        results_attribute_All.append(np.concatenate((r_a_o_index, r_a_a_index, r_a_at_index, r_a_wl_index), axis=-1))
        results_attribute_OA.append(np.concatenate((r_a_o_index, r_a_a_index), axis=-1))

    # for l_attr, r_attr in zip(labels_attribute_All,results_attribute_All):
    #     print("Label Attribute Shapes:", l_attr.shape, "Result Attribute Shapes:", r_attr.shape)
    
    labels_attribute.append(concatenate_if_not_empty(*labels_attribute_All, reshape=True))
    labels_attribute.append(concatenate_if_not_empty(*label_attribute_OA, reshape=True))

    result_attribute.append(concatenate_if_not_empty(*results_attribute_All, reshape=True))
    result_attribute.append(concatenate_if_not_empty(*results_attribute_OA, reshape=True))

    # for l_attr in labels_attribute:
    #     print("Label Attribute Shapes:", l_attr.shape)
    # for r_attr in result_attribute:
    #     print("Result Attribute Shapes:", r_attr.shape)

    return labels_attribute, labels_event, labels_trace, result_attribute, result_event, result_trace

def element_wise_max(*arrays, attribute_level=False):
    valid_arrays = [arr for arr in arrays if arr.size > 0]

    if not valid_arrays:
        return np.array([])  
    
    if attribute_level:
        for arr in valid_arrays:
            print(arr.shape)

        valid_arrays = np.concatenate(valid_arrays, axis=0)
        print(valid_arrays.shape)
        
    return np.maximum.reduce(valid_arrays)

def score(run):
    pred_probs_levels = run['results']
    config = run['config']
    timestamp = run['timestamp']
    index = run['index']
    buckets = run['buckets']

    sorted_results = dict(sorted(pred_probs_levels.items(), key=lambda x: extract_number(x[0])))
    perspective_label_indices = get_indexes_by_value(config['attribute_perspectives_original'])
    # print("perspective_label_indices", perspective_label_indices)

    (
        labels_attribute, 
        labels_event, 
        labels_trace, 
        result_attribute, 
        result_event, 
        result_trace
    ) = reshape_data_for_scoring(results=sorted_results, perspective_label_indices=perspective_label_indices, buckets=buckets)

    # print("Reshaped data for scoring")
    # print("labels_attribute", len(labels_attribute), labels_attribute[0].shape)
    # print("labels_event", labels_event.shape)
    # print("labels_trace", labels_trace.shape)
    # print("result_attribute", len(result_attribute), result_attribute[0].shape)
    # print("result_event", len(result_event), result_event[0].shape)
    # print("result_trace", len(result_trace), result_trace[0].shape)

    level = ['trace', 'event', 'attribute']
    y_true_levels = [labels_trace, labels_event, labels_attribute]
    pred_probs_levels = [result_trace, result_event, result_attribute]
    perspective_keys = Perspective.keys() + [4, 5] # + [4] is for the single perspective trace + [5] is for the single perspective only order and attribute
    perspective_labels = Perspective.values() + ['Single', 'Single_OA']
    
    # Calculate the F1 score for each level and perspective
    scores = []
    for (level, y_trues, pred_probs), perspective in itertools.product(zip(level, y_true_levels, pred_probs_levels), perspective_keys):
        print("Calculating scores for: Level: ", level, " Perspective: ", perspective)
        try:
            roc_auc, pr_auc, f1, precision, recall = calculate_scores(y_trues, pred_probs, perspective)
            print(perspective, f1)
        except Exception as e:
            print(level, perspective)
            print(e)
            roc_auc, pr_auc, f1, precision, recall = 0, 0, 0, 0, 0
        
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
            'perspective': perspective_labels[perspective],
            # Scores
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1': f1,
            'precision':precision,
            'recall':recall,
            'run_time': config['run_time'],
            # Config
            'batch_size': config['batch_size'],
            'prefix': config['prefix'],
            'buckets': config.get('bucket_boundaries', None),
            'categorical_encoding': config.get('categorical_encoding', 'None'),
            'numerical_encoding': config.get('numerical_encoding', 'None'),
            'vector_size': config.get('vector_size', 'None'),
            'window_size': config.get('window_size', 'None'),
        })


    return pd.DataFrame(scores)