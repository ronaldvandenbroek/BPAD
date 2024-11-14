
import numpy as np
from collections import Counter, defaultdict

from utils.enums import AttributeType, EncodingCategorical

def process_bucket_results(
        errors_raw, 
        categorical_encoding, 
        vector_size,
        bucketing,
        attribute_dims,
        dataset_mask,
        attribute_types,
        case_max_length,
        anomaly_perspectives,
        case_lengths,
        error_power
    ):

    # errors_raw = targets - predictions
    errors_unmasked = np.power(errors_raw, error_power)

    # RCVDB: If trace2vec than the first vector_size of outputs can be discarded as it is not relevant for specific perspectives
    if categorical_encoding in (EncodingCategorical.TRACE_2_VEC_ATC, EncodingCategorical.TRACE_2_VEC_C):
        errors_unmasked = errors_unmasked[:, vector_size:]

    # RCVDB: Mask empty events if no buckets are used or encoding method is not W2V
    if not bucketing and categorical_encoding in (EncodingCategorical.WORD_2_VEC_ATC, EncodingCategorical.TRACE_2_VEC_ATC):
    # Applies a mask to remove the events not present in the trace   
    # (cases, flattened_errors) --> errors_unmasked
    # (cases, num_events) --> dataset.mask (~ inverts mask)
    # (cases, num_events, 1) --> expand dimension for broadcasting
    # (cases, num_events, attributes_dim) --> expand 2nd axis to size of the attributes
    # (cases, num_events * attributes_dim) = (cases, flattened_mask) --> reshape to match flattened error shape
        errors = errors_unmasked * np.expand_dims(~dataset_mask, 2).repeat(attribute_dims.sum(), 2).reshape(
            dataset_mask.shape[0], -1)
    else:
        errors = errors_unmasked

    # If W2V all categorical events are embedded into a single vector
    if categorical_encoding in (EncodingCategorical.WORD_2_VEC_ATC, EncodingCategorical.TRACE_2_VEC_ATC):
        attribute_type_counter = Counter(attribute_types)

        categorical_tiles = np.tile(vector_size, [attribute_type_counter[AttributeType.CATEGORICAL]])
        numerical_single_attribute = [1] * attribute_type_counter[AttributeType.NUMERICAL]
        numerical_tiles = np.tile(numerical_single_attribute, [case_max_length])
        w2v_tiles = np.concatenate((categorical_tiles, numerical_tiles))

        split_attribute = np.cumsum(w2v_tiles, dtype=int)[:-1]
    else:
        split_attribute = np.cumsum(np.tile(attribute_dims, [case_max_length]), dtype=int)[:-1]

    # Get the errors per attribute by splitting said trace
    # (attributes * events, cases, attribute_dimension)
    errors_attr_split = np.split(errors, split_attribute, axis=1)

    # Mean the attribute_dimension
    # Scalar attributes are left as is as they have a size of 1
    # np.mean for the proportion of the one-hot encoded predictions being wrong
    # np.sum for the total one-hot predictions being wrong
    # (attributes * events, cases)
    # RCVDB: TODO Sum the errors to amplify the difference between normal and anomalous data
    errors_attr_split_summed = [np.sum(attribute, axis=1) for attribute in errors_attr_split]

    # if categorical_encoding == EncodingCategorical.WORD_2_VEC_ATC or categorical_encoding == EncodingCategorical.TRACE_2_VEC_ATC:
    #     errors_attr_split_summed = [np.sum(attribute, axis=1) for attribute in errors_attr_split]
    # else:
    #     errors_attr_split_summed = [np.mean(attribute, axis=1) for attribute in errors_attr_split]

    # Split the attributes based on which event it belongs to
    if categorical_encoding in (EncodingCategorical.WORD_2_VEC_ATC, EncodingCategorical.TRACE_2_VEC_ATC):
        # Everything before the shared index is shared between each event as they are averaged during encoding
        shared_index = attribute_type_counter[AttributeType.CATEGORICAL]
        split_shared = np.split(errors_attr_split_summed, [shared_index])

        shared_indexes = split_shared[0]
        split_indexes = split_shared[1]

        # Combining both arrays
        expanded_shared_indexes = np.expand_dims(shared_indexes, axis=0) # Shape: (1, 5, 1000)
        # Duplicate the average error over all the events for compatibility with the rest of the code
        errors_event_split_categorical = np.tile(expanded_shared_indexes, (case_max_length, 1, 1)) # Shape: (13, 5, 1000)

        # Split the rest of the numerical values
        if attribute_type_counter[AttributeType.NUMERICAL] > 0:
            event_numerical_size = attribute_type_counter[AttributeType.NUMERICAL]
            split_event = np.arange(
                start=event_numerical_size, #shared_index, 
                stop=len(errors_attr_split_summed), 
                step=event_numerical_size)[:-1]

            errors_event_split_numerical_splits = np.split(split_indexes, split_event, axis=0)
            errors_event_split_numerical = np.array(errors_event_split_numerical_splits)

            errors_event_split = np.concatenate((errors_event_split_numerical, errors_event_split_categorical), axis=1)
        else:
            errors_event_split = errors_event_split_categorical
        
        # (events, attributes, cases)
        # to
        # (attributes, events, cases)
        errors_event_split = np.transpose(errors_event_split, (1,0,2))
    else:
        split_event = np.arange(
            start=case_max_length, 
            stop=len(errors_attr_split_summed), 
            step=case_max_length)
        
        # (attributes, events, cases)
        errors_event_split = np.split(errors_attr_split_summed, split_event, axis=0)

    # Split the attributes based on which perspective they belong to
    # (perspective, attributes, events, cases)
    # RCVDB: TODO This splitting does not work if the ordering has changed
    grouped_error_scores_per_perspective = defaultdict(list)
    for event, anomaly_perspective in zip(errors_event_split, anomaly_perspectives):
        grouped_error_scores_per_perspective[anomaly_perspective].append(event)

    # Calculate the error proportions per the perspective per: attribute, event, trace
    trace_level_abnormal_scores = defaultdict(list)
    event_level_abnormal_scores = defaultdict(list) 
    attr_level_abnormal_scores = defaultdict(list)
    for anomaly_perspective in grouped_error_scores_per_perspective.keys():
        # Transpose the axis to make it easier to work with 
        # (perspective, attributes, events, cases)
        # to
        # (cases, events, attributes) 
        t = np.transpose(grouped_error_scores_per_perspective[anomaly_perspective], (2, 1, 0))
        event_dimension = case_lengths
        attribute_dimension = t.shape[-1]

        error_per_trace = np.sum(t, axis=(1,2)) / (event_dimension * attribute_dimension)
        trace_level_abnormal_scores[anomaly_perspective] = error_per_trace

        error_per_event = np.sum(t, axis=2) / attribute_dimension
        event_level_abnormal_scores[anomaly_perspective] = error_per_event

        error_per_attribute = t
        attr_level_abnormal_scores[anomaly_perspective] = error_per_attribute

    return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores