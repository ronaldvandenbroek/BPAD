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



