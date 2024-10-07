from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow import math, reduce_sum, cast, equal, argmax, float32, one_hot, reduce_sum, where, zeros_like
from keras.losses import sparse_categorical_crossentropy
from tensorflow import nn

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)

        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):

        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)

# Defining the loss function
def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)

    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask

    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)


# Defining the accuracy function
def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(equal(target, 0))

    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)

    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)

    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)

# Defining the accuracy function
def likelihood_fcn(target, prediction):
    # Create a mask so that zero-padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(equal(target, 0))

    # Convert predictions to probabilities using softmax (assumes last axis is for the class probabilities)
    prediction_probabilities = nn.softmax(prediction, axis=-1)

    # Gather the predicted probabilities of the correct classes (i.e., target values)
    # Use tf.gather_nd to select the probabilities corresponding to the actual target classes
    target_one_hot = one_hot(target, depth=prediction.shape[-1])
    correct_class_likelihood = reduce_sum(target_one_hot * prediction_probabilities, axis=-1)

    # Apply the mask to filter out padding positions
    correct_class_likelihood = where(padding_mask, correct_class_likelihood, zeros_like(correct_class_likelihood))

    # Compute the mean of the correct class likelihoods, ignoring padding tokens
    padding_mask = cast(padding_mask, float32)
    accuracy = reduce_sum(correct_class_likelihood) / reduce_sum(padding_mask)
    
    return accuracy