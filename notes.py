# Encoding methods
    # Thesis Master - 2023 - Beyond the Norm Anomaly Detection in Event Streams using Word2Vec Embedding and Autoencoders
        # Paper - 2024 - CoopIS Online Anomaly Detection in Prefixes
            # Resulting paper from the thesis so use that one for reference

    # Introduces the argumentation on why to look at the different encoding methods in streams with autoencoders
        # 3.1 Encoding Methods
                # 4.2.1 Gives an explanation of the one hot encoding
            # Word2Vec
                # 4.2.2 Gives an explanation of the Word2Vec encoding
                    # CTA and ATC 
        # Extra note regarding bucketing: See 6.3

    # Other overviews
        # Evaluating Trace Encoding Methods in Process Mining
            # https://arts.units.it/bitstream/11368/3014625/4/3014625_from-data-to-models-and-back-2021-180-195-Post_print.pdf
        # Paper - 2023 - Trace Encoding in Process Mining - Survey and Benchmarks
            # Use one hot as the baseline and further focus on text based encoding methods 
            # One Hot Encoding
                # (Weiss et al., 2015) Fundamentals of Predictive Text Mining
                # Use as a baseline
            # Word2Vec
                # word2vec (CBOW) (Mikolov et al., 2013b) https://arxiv.org/pdf/1310.4546
                # word2vec (skip-gram) (Mikolov et al., 2013a) https://arxiv.org/abs/1301.3781
                    # https://radimrehurek.com/gensim/models/word2vec.html
                # doc2vec (Le and Mikolov, 2014)
                # GloVe (Pennington et al., 2014)
        # Paper - 2018 - BPM act2vec-- trace2vec-- log2vec --model2vec
            # Trace2Vec
                # Adds a trace representation to the word2vec model to allow for the learning of trace representations as a whole
                # Which might provide a better representation of the traces in comparison to the word2vec model
                    # See 2.2 for the extention
                # Based on the doc2vec model where a paragraph is also encoded together with the word context
                    # https://arxiv.org/pdf/1405.4053

        # TODO Figure out how to justify the coice of encoding methods to test
            # Previous thesis is the base to expand upon with the new perspectives
                # CTA detroys the order of the events due to averaging thus have to use ATC
                    # Or don't use averaging which isn't explored by the previous thesis
                # Extra benefit is that ATC performs better than CTA
                # W2V is implemented with skipgram in my thesis
                    # See Mikolov et al., 2013a as it justifies that skipgram is better than CBOW
            # Use the Paper - 2018 - BPM act2vec-- trace2vec-- log2vec --model2vec to justify the use of trace2vec

            # Problem with the word2vec and trace2vec models is that they need sentences beforehand to train on which is not available within a datastream
                # Thus potentially a fixed vector approach could be used inspired by Vaswani et al., 2017
                

            # Survey paper stick to the text based encoding methods?
             

# 'None', 
    # 
# 'One Hot', 
    # 
# 'Embedding', 
# 'Word2Vec Average Then Concatinate', 
# 'Word2Vec Concatinate', 
# 'Fixed Vector',
# 'Trace2Vec Average Then Concatinate',
# 'Trace2Vec Concatinate',
# 'ELMo',
# 'Tokenizer'



# Experimental Setup
    # Intro into the experimental setup chapters
    # Datasets
        # Done
    # Evaluation Metrics
        # F1 Score
            # See preparation report
        # CD Ranking
    # Hyperparameters
        # DAE
            # General Hyperparameters tests
                # Anomaly_Percentage_v2
                    # To determine the best anomaly percentage to use in finetuning
                    # Used medium dataset as a base
                    # Chosen best scoring result of 0.45
                        # Why?
                # Batch Size
                    # Was done after anomaly percentage test
                    # To determine the best batch size to use in the model
                    # Mainly concerning performance vs model speed
                    # Used medium dataset as a base
                    # Chosen batch size of 8 for performance/speed consideration
                # Synthetic dataset_v4
                    # To determine on which dataset to finetune the encoding methods
                    # Chosen Gigantic dataset as it is the most complex dataset + best scoring result
                        # Aim is to mimic the complexity of real world data as much as possible
            # Mention the finetuning results here, not as separate notes experimental results
                # Just note down the results but can go into some interesting conclusions
                # W2V-ATC
                # W2V-C
                # T2V-ATC
                # T2V-C
                # Fixed-Vector
        # Transformer
            # Mention finetuning experiments for the transformer model
    # Hardware Details
        # Mention python/setup versions
        # TODO: How to deal with server vs local setup differences?
    
# Experimental Results
    # 'Evaluating Prefix-Embedding Methods'
        # Go over rankings and mention best performing methods for both overall and per perspective
        # Note that due to run time contraints synthetic top 3 methods are used for real world data
        # Synthetic All Models
        # Real world All Models

        # Prefix vs No Prefix
            # See discussion of the topic in the meeting notes/recodings

    # Transformer vs DAE results with real-time data 
        # 'Evaluating real-time Transformer Viability'
        # 'Evaluating Anomaly Detection Model'

        # TODO Dealing with concept drifts, maybe rework scoring to allow for rolling average?

    # 'Evaluating Anomaly Detection Model Components'
        # Transformer ablation study






# Classification approaches

# survey paper = Paper - 2024 - Survey_and_Benchmark_of_Anomaly_Detection_in_Business_Processes

# Process-Model based
# Process model based approaches utilize process models representing the log to detect anomalies
# Either a process model needs to already exist or a process discovery technique needs to be used to discover said process model.
# Such as a heuristic miner algorithm [49] (survey paper)
# From a log or sample of the log
# 
# From such a process model anomaly dectection can be done by the way of conformance checking [67]
# TODO look up conformance checking in vanderaalst2016
# Which compares the discrepencies between the traces and the process model. 
# 
# TODO potentially also add comparing discovered process model comparisons
# See BPAB survey

# TODO Give a more in depth example with the paper from the table

# A drawback of process-model based approaches is that a process model is generally not available and thus needs to be discovered.
# This discovery is ideally performed on a clean dataset, discovering process models on real world datasets, which already contain anomalies is non-trivial
# The performace of the methods is highly dependant on the quality of the mined process model

# These approaches also have trouble dealing with concept drift unless a system is in place to periodically rediscover the underlying process model

# Generally limited to trace level granularity (TODO cite survey paper)


# Probabilistic based
# Probabilistic work based on learning a probabilistic model or learning assosiation rules and 
# generating the probablilities a certain trace or event is anomalous or not with a threshold

# A drawback of these models is that they may have problems capturing long-term and complex depentencies between events due to the simplicity of the probabilistic model or discovered rules

# TODO Give a more in depth example with the paper from the table


# Distance based approach
# Distance based approaches aim to discover the anomalous traces by the assumption that based on some metric space anomalous traces have an increased distance from normal traces

# Closely related are the construction based Neural networks

# A general limitation of distance based approaches is that they suffer from the curse of dimensionality and not adequatly considering the impact of variable trace lengths
# TODO Explain
# Calculating the distence between traces can be time-consuming, which can decrease efficiency on large datasets
 
# TODO Give a more in depth example with the paper from the table

# Generally limited to trace level granularity (TODO cite survey paper)



# Information-theoretic approaches
# Within statistics leverage is an often used as a matric to identify outliers, which measures the distance between one observation in a dataset to other observations
# Can be quantified using methods such as Cook or Welsh-Kuh distance (TODO remove or cite)

# A drawback of this method is that they generally require a log to be converted into a matrix for the arithmatic which requires a large amount of memory space and is computaionally intensive.

# TODO Give a more in depth example with the paper from the table

# Generally limited to trace level granularity (TODO cite survey paper)


# Neural Network based
# With the advancements of deeplearning, deep neural networks are increasingly used within AD
# As they have the capacity to learn complex patterns and as such indentify normal patterns even from messy datasets.

# Overall there are three topics of interest within neural networks that this paper will explore further

# How are traces converted into vector representations compatible with neural networks?
# TODO table

# What architectures are effective for anomaly detection
# Autoencoders

# Graph based

# How can anomalies be detected

