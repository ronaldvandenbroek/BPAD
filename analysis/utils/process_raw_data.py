import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, recall_score, precision_score, f1_score

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

    # print("Calculating scores for perspective", perspective, y_true.shape, pred_prob.shape)
    # RCVDB TODO: If synthetic then a window bucket amount of 59 is used as it cleanly divides the 56227 traces
    # window_size = len(y_true) // window_buckets
    # window_size = len(y_true) // window_buckets
    #overlap = int(window_size * 0.1)
    #rest = window_size % overlap

    #print("Largest length of Falses: ", largest_false_streak(y_true), "Window Size: ", window_size, "Overlap: ", overlap, "Rest: ", rest)

    # ROC-AUC
    # try:
    #     roc_auc = roc_auc_score(y_true, pred_prob)
    # except:
    #     roc_auc = 0
        
    # PR-AUC
    # pr_auc = average_precision_score(y_true, pred_prob)

    # F1-Score
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=pred_prob)
    f1s=calculate_f1(precision,recall)
    f1s[np.isnan(f1s)] = 0
    f1_best_index=np.argmax(f1s)

    # best_f1 = f1s[f1_best_index]
    # best_threshold = thresholds[f1_best_index]
    # best_precision = precision[f1_best_index]
    # best_recall = recall[f1_best_index]

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
    roc_auc = 0
    pr_auc = 0
    return roc_auc, pr_auc, f1s[f1_best_index], np.mean(precision), np.mean(recall)

def calculate_scores_v2(y_trues_attribute, y_trues_event, y_trues_trace, pred_probs_attribute, perspective):
    print("Calculating scores for perspective", perspective)
    y_true_attribute = y_trues_attribute[perspective][:]
    y_true_event = y_trues_event[perspective][:]
    y_true_trace = y_trues_trace[perspective][:]
    pred_prob_attribute = pred_probs_attribute[perspective][:]

    print("Calculating scores for perspective", perspective)
    print(y_true_attribute.shape, y_true_event.shape, y_true_trace.shape, pred_prob_attribute.shape)

    precision, recall, thresholds = precision_recall_curve(y_true=y_true_attribute, probas_pred=pred_prob_attribute)

    f1s=calculate_f1(precision, recall)
    f1s[np.isnan(f1s)] = 0
    f1_best_index=np.argmax(f1s)
    f1_best_threshold = thresholds[f1_best_index]

    y_pred_attribute = pred_prob_attribute > f1_best_threshold
    y_pred_event = y_pred_attribute.any(axis=1)
    y_pred_trace = y_pred_event.any(axis=1)

    f1_attribute = f1_score(y_true_attribute, y_pred_attribute)
    f1_event = f1_score(y_true_event, y_pred_event)
    f1_trace = f1_score(y_true_trace, y_pred_trace)

    print('F1 Attribute:', f1_attribute, 'F1 Event:', f1_event, 'F1 Trace:', f1_trace)
    
    return f1_attribute, f1_event, f1_trace

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

def process_attribute_labels(values, case_length, perspective, perspective_label_indices):
    if perspective not in perspective_label_indices:
        return

    # print(values.shape)

    perspective_value = values[perspective, :, :, :]
    # print(perspective_value.shape)

    perspective_masked = perspective_value[:, :case_length, :]
    # print(perspective_masked.shape)

    perspective_indexed = perspective_masked[:,:,perspective_label_indices[perspective]]
    # print(perspective_indexed.shape)

    # perspective_attribute_value = perspective_indexed.reshape(-1) # Flatten the output
    # print(perspective_attribute_value.shape)

    #output.append(perspective_attribute_value)
    # output.append(perspective_indexed)

    return perspective_indexed

def reshape_data_for_scoring(results, perspective_label_indices, buckets):
    # print(perspective_label_indices)
    # print(Perspective.items())
    labels_attribute_Arrival_Time = []
    labels_attribute_Workload = []
    labels_attribute_Order = []
    labels_attribute_Attribute  = []
    labels_attribute_All = []
    label_attribute_OA = []

    labels_event = []
    labels_event_OA = []
    labels_trace = []
    labels_trace_OA = []

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

                # print('Attribute Labels:', length, value.shape, transposed_value.shape)

                l_a_o = process_attribute_labels(
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.ORDER,
                    perspective_label_indices=perspective_label_indices)
                labels_attribute_Order.append(l_a_o.reshape(-1)) # Flatten the output
                l_a_a = process_attribute_labels(
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.ATTRIBUTE,
                    perspective_label_indices=perspective_label_indices)
                labels_attribute_Attribute.append(l_a_a.reshape(-1)) # Flatten the output
                l_a_at = process_attribute_labels(
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.ARRIVAL_TIME,
                    perspective_label_indices=perspective_label_indices)
                labels_attribute_Arrival_Time.append(l_a_at.reshape(-1)) # Flatten the output
                l_a_wl = process_attribute_labels(
                    values=transposed_value, 
                    case_length=length, 
                    perspective=Perspective.WORKLOAD,
                    perspective_label_indices=perspective_label_indices)
                labels_attribute_Workload.append(l_a_wl.reshape(-1)) # Flatten the output
                
                label_attribute_all = np.concatenate((l_a_o, l_a_a, l_a_at, l_a_wl), axis=-1)#.reshape(-1)
                labels_attribute_All.append(label_attribute_all)
                label_attribute_oa = np.concatenate((l_a_o, l_a_a), axis=-1)#.reshape(-1)
                label_attribute_OA.append(label_attribute_oa)

                # print('Attribute labels:', length, l_a_o.shape, l_a_a.shape, l_a_at.shape, l_a_wl.shape)
                # print('Attribute labels Combined:', length, label_attribute_all.shape, label_attribute_oa.shape)

                # event_transposed_value = np.transpose(value, (2,0,1,3))[:, :, :length, :]
                # Alternative way of calculating the event and trace labels
                alt_l_e_o = l_a_a.any(axis=2)
                alt_l_e_a = l_a_a.any(axis=2)
                alt_l_e_at = l_a_at.any(axis=2)
                alt_l_e_wl = l_a_wl.any(axis=2)

                # print('Alt Event labels split:', length, alt_l_e_o.shape, alt_l_e_a.shape, alt_l_e_at.shape, alt_l_e_wl.shape)

                alt_labels_event = np.stack((alt_l_e_o, alt_l_e_a, alt_l_e_at, alt_l_e_wl), axis=2)
                alt_labels_event_t = alt_labels_event.transpose(2,0,1)#(2,1,0)
                alt_labels_event_flattened = alt_labels_event_t.reshape(alt_labels_event_t.shape[0], -1)
                alt_labels_event_oa = np.stack((alt_l_e_o, alt_l_e_a), axis=2)
                alt_lables_event_oa_t = alt_labels_event_oa.transpose(2,0,1)#(2,1,0)
                alt_labels_event_oa_flattened  = alt_lables_event_oa_t.reshape(alt_lables_event_oa_t.shape[0], -1)

                labels_event.append(alt_labels_event_flattened)
                labels_event_OA.append(alt_labels_event_oa_flattened)

                # print('Alt Event labels:', length, 'ALL ',alt_labels_event.shape, 'OA ', alt_labels_event_oa.shape)
                # print('Alt Event labels flat:', length, 'ALL ',alt_labels_event_flattened.shape, 'OA ', alt_labels_event_oa_flattened.shape)

                alt_labels_trace = alt_labels_event.any(axis=1)
                alt_labels_trace_t = alt_labels_trace.transpose(1,0)
                alt_labels_trace_oa = alt_labels_event_oa.any(axis=1)
                alt_labels_trace_oa_t = alt_labels_trace_oa.transpose(1,0)

                labels_trace.append(alt_labels_trace_t)
                labels_trace_OA.append(alt_labels_trace_oa_t)

                # print('Alt Trace labels:', length, 'ALL ', alt_labels_trace.shape, 'OA ', alt_labels_trace_oa.shape)
                
            # elif 'event' in key:
            #     perspective_value = np.transpose(value, (2,0,1))[:, :, :length]
            #     perspective_value = perspective_value.reshape(perspective_value.shape[0], -1)
            #     # print('event labels:', length, value.shape, perspective_value.shape)
            #     labels_event.append(perspective_value)
            # elif 'trace' in key:
            #     perspective_value = np.transpose(value, (1,0))
            #     print('trace', perspective_value.shape)
            #     labels_trace.append(perspective_value)
        elif 'result' in key:
            if 'attribute' in key:
                # print(value.shape)
                # value_max = np.max(value, axis=2)
                # print(value.shape, normalize(value.reshape(-1)).shape, perspective)
                # print(value.shape)
                value = normalize(value)# .reshape(-1)
                # print('Attribute Results:', length, value.shape, perspective)
                event_value = value.max(axis=-1)
                trace_value = event_value.max(axis=-1)
                # print('ALT Event Results:', length, event_value.shape, perspective)
                # print('ALT Trace Results:', length, trace_value.shape, perspective)

                if 'Order' in perspective:
                    result_attribute_Order.append(value)
                    result_event_Order.append(event_value)
                    result_trace_Order.append(trace_value)
                elif 'Attribute' in perspective:
                    result_attribute_Attribute.append(value)
                    result_event_Attribute.append(event_value)
                    result_trace_Attribute.append(trace_value)
                elif 'Arrival Time' in perspective:
                    result_attribute_Arrival_Time.append(value)
                    result_event_Arrival_Time.append(event_value)
                    result_trace_Arrival_Time.append(trace_value)
                elif 'Workload' in perspective:
                    result_attribute_Workload.append(value)
                    result_event_Workload.append(event_value)
                    result_trace_Workload.append(trace_value)
            # if 'event' in key:
            #     value = normalize(value)
            #     print('Event Results:', length, value.shape, perspective)
            #     if 'Order' in perspective:
            #         result_event_Order.append(value)
            #     elif 'Attribute' in perspective:
            #         result_event_Attribute.append(value)
            #     elif 'Arrival Time' in perspective:
            #         result_event_Arrival_Time.append(value)
            #     elif 'Workload' in perspective:
            #         result_event_Workload.append(value)
            # elif 'trace' in key:
            #     value = normalize(value)
            #     print('Trace Results:', length, value.shape, perspective)
            #     if 'Order' in perspective:
            #         result_trace_Order.append(value)
            #     elif 'Attribute' in perspective:
            #         result_trace_Attribute.append(value)
            #     elif 'Arrival Time' in perspective:
            #         result_trace_Arrival_Time.append(value)
            #     elif 'Workload' in perspective:
            #         result_trace_Workload.append(value)

    # for l_trace in labels_trace:
    #     print("Label Trace Shapes:", l_trace.shape)
    # for l_trace in labels_trace_OA:
    #     print("Label Trace OA Shapes:", l_trace.shape)
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
    # labels_trace_single_perspective_OA = np.any(labels_trace[:2,:], axis=0)
    labels_trace_OA = np.concatenate(labels_trace_OA, axis=1)
    labels_trace_single_perspective_OA = np.any(labels_trace_OA, axis=0)
    labels_trace_stacked = np.vstack([labels_trace, labels_trace_single_perspective, labels_trace_single_perspective_OA])
    # print("Label Trace Shapes:", labels_trace_stacked.shape, labels_trace.shape, labels_trace_OA.shape, labels_trace_single_perspective.shape, labels_trace_single_perspective_OA.shape)

    # print("Trace OA", labels_trace_OA.shape, labels_trace_single_perspective_OA.shape)

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
    try:
        results_trace_single_perspective = element_wise_max(*result_trace)
        results_trace_single_perspective_OA = element_wise_max(*result_trace_OA)
        result_trace.append(results_trace_single_perspective)
        result_trace.append(results_trace_single_perspective_OA)
    except:
        print()
        for r_trace in result_trace:
            print(r_trace.shape)
        raise
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
    # labels_event_single_perspective_OA = np.any(labels_event[:2,:], axis=0)
    labels_event_OA = np.concatenate(labels_event_OA, axis=1)
    labels_event_single_perspective_OA = np.any(labels_event_OA, axis=0)
    labels_event_stacked = np.vstack([labels_event, labels_event_single_perspective, labels_event_single_perspective_OA])
    # print("Label Event Shapes:", labels_event_stacked.shape, labels_event.shape, labels_event_OA.shape, labels_event_single_perspective.shape, labels_event_single_perspective_OA.shape)

    result_event = [
        concatenate_if_not_empty(*result_event_Order, reshape=True),
        concatenate_if_not_empty(*result_event_Attribute, reshape=True),
        concatenate_if_not_empty(*result_event_Arrival_Time, reshape=True),
        concatenate_if_not_empty(*result_event_Workload, reshape=True)
    ]
    result_event_OA = [
        concatenate_if_not_empty(*result_event_Order, reshape=True),
        concatenate_if_not_empty(*result_event_Attribute, reshape=True)
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

    # for l_labels in labels_event:
    #     print("Event Shapes:", l_labels.shape)

    return labels_attribute, labels_event_stacked, labels_trace_stacked, result_attribute, result_event, result_trace

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

    scores = []
    # # Calculate the F1 score for each level and perspective
    # for perspective in perspective_keys:
    #     f1_attribute, f1_event, f1_trace = calculate_scores_v2(labels_attribute, labels_event, labels_trace, result_attribute, perspective)
    #     append_scores(scores, run, 'attribute', perspective_labels[perspective], 0, 0, f1_attribute, 0, 0, model_prefix='v2_')
    #     append_scores(scores, run, 'event', perspective_labels[perspective], 0, 0, f1_event, 0, 0, model_prefix='v2_')
    #     append_scores(scores, run, 'trace', perspective_labels[perspective], 0, 0, f1_trace, 0, 0, model_prefix='v2_')

    for (level, y_trues, pred_probs), perspective in itertools.product(zip(level, y_true_levels, pred_probs_levels), perspective_keys):
        # print("Calculating scores for: Level: ", level, " Perspective: ", perspective, y_trues[perspective].shape, pred_probs[perspective].shape)
        try:
            roc_auc, pr_auc, f1, precision, recall = calculate_scores(y_trues, pred_probs, perspective)
            # print("%.2f" % f1)
        except Exception as e:
            print(level, perspective)
            print(e)
            roc_auc, pr_auc, f1, precision, recall = 0, 0, 0, 0, 0
        
        append_scores(scores, run, level, perspective_labels[perspective], roc_auc, pr_auc, f1, precision, recall)

    return pd.DataFrame(scores)

def append_scores(scores, run, level, perspective, roc_auc, pr_auc, f1, precision, recall, model_prefix=''):
    config = run['config']
    timestamp = run['timestamp']
    index = run['index']

    scores.append({
            # High level differentiatiors
            'run_name':config['run_name'],
            'model':model_prefix + config['model'],
            'dataset':config['dataset'],
            'timestamp':timestamp,
            'index':index,
            # 'repeat':config['repeat'],
            # Level/Perspectives
            'level': level,
            'perspective': perspective,
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
                  