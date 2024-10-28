from baseline.binet.core import NNAnomalyDetector
from novel.transformer.components.transformer import TransformerModel
from novel.transformer.components.utils import LRScheduler
from utils.dataset import Dataset
from utils.embedding.attribute_dictionary import AttributeDictionary
from utils.enums import EncodingCategorical
from utils.settings.settings_multi_task import SettingsMultiTask

from keras.optimizers import Adam


class Transformer(NNAnomalyDetector):
    """Implements a transformer based anomaly detection algorithm."""

    abbreviation = 'transformer'
    name = 'Transformer'

    supports_attributes = True

    config = dict(hidden_layers=2,
                  hidden_size_factor=.01,
                  noise=None # 0.5
     ) 
    
    def __init__(self, model=None):
        """Initialize Trasformer model.

        Size of hidden layers is based on input size. The size can be controlled via the hidden_size_factor parameter.
        This can be float or a list of floats (where len(hidden_size_factor) == hidden_layers). The input layer size is
        multiplied by the respective factor to get the hidden layer size.

        :param model: Path to saved model file. Defaults to None.
        :param hidden_layers: Number of hidden layers. Defaults to 2.
        :param hidden_size_factor: Size factors for hidden layer base don input layer size.
        :param epochs: Number of epochs to train.
        :param batch_size: Mini batch size.
        """
        super(Transformer, self).__init__(model=model)

    @staticmethod
    def model_fn(dataset:Dataset, bucket_boundaries:list, categorical_encoding, w2v_vector_size=None, **kwargs):
        # Model Paramerters
        heads = 8  # Number of self-attention heads
        dim_queries_keys = 64  # Dimensionality of the linearly projected queries and keys
        dim_values = 64  # Dimensionality of the linearly projected values
        dim_model = 512  # Dimensionality of model layers' outputs
        dim_fully_con = 2048  # Dimensionality of the inner fully connected layer
        n_layers = 6  # Number of layers in the encoder stack

        # Training Parameters
        dropout_rate = 0.1
        batch_size = 8
        beta_1 = 0.9
        beta_2 = 0.98
        epsilon = 1e-9


        if categorical_encoding == EncodingCategorical.ONE_HOT:
            features = dataset.flat_onehot_features_2d
        elif categorical_encoding == EncodingCategorical.EMBEDDING:
            features = dataset.flat_embedding_features_2d
        elif categorical_encoding == EncodingCategorical.WORD_2_VEC:
            features = dataset.flat_w2v_features_2d()
        else:
            features = dataset.flat_features_2d

        case_lengths = dataset.case_lens
        case_labels = dataset.case_labels
        event_labels = dataset.event_labels
        attr_labels = dataset.attr_labels

        # Determine the vocab size
        vocab_size = 0
        for encoder in dataset.encoders:
            encoder:AttributeDictionary
            largest_attribute = encoder.largest_attribute()
            if largest_attribute > vocab_size:
                vocab_size = largest_attribute
            
        optimizer = Adam(LRScheduler(dim_model), beta_1, beta_2, epsilon)

        model = TransformerModel(
            # Dataset variables
            enc_vocab_size=enc_vocab_size,
            dec_vocab_size=dec_vocab_size,
            enc_seq_length=enc_seq_length,
            dec_seq_length=dec_seq_length,
            # Model variables
            heads=heads,
            dim_queries_keys=dim_queries_keys,
            dim_values=dim_values,
            dim_model=dim_model,
            dim_fully_con=dim_fully_con,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
        )

        return {0:model}, {0:features}, {0:features}, {0:case_lengths}, {0:case_labels}, {0:event_labels}, {0:attr_labels}


    def train_and_predict(self, 
                          dataset:Dataset, 
                          batch_size=2, 
                          bucket_boundaries=None, 
                          categorical_encoding=EncodingCategorical.ONE_HOT, 
                          w2v_vector_size=None):
        model_buckets, features_buckets, targets_buckets, case_lengths_buckets, bucket_case_labels, bucket_event_labels, bucket_attr_labels = self.model_fn(dataset, bucket_boundaries, categorical_encoding, w2v_vector_size, **self.config)
