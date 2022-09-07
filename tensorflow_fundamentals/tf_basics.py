# To disable warning messages from tensorflow
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# # Access GPU
# devices = tf.config.list_physical_devices()
# print(devices)

# ### Creating tensors ###
# scalar = tf.constant(7)
# # print(scalar)
# # print(scalar.ndim)
#
# vector = tf.constant([10, 7])
# # print(vector)
# # print(vector.ndim)
#
# matrix = tf.constant([[1, 2], [3, 4]])
# # print(matrix)
# # print(matrix.ndim)
#
# Tensors are defaulted to dtype: int32
# another_matrix = tf.constant([[10., 12., 14.], [16., 18., 20.]], dtype=tf.float16)
# # print(another_matrix)
# # print(another_matrix.ndim)
#
# tensor = tf.constant([[[1, 2, 3],
#                        [4, 5, 6]],
#                       [[7, 8, 9],
#                        [10, 11, 12]],
#                       [[13, 14, 15],
#                        [16, 17, 18]]])
# # print(tensor)
# # print(tensor.ndim)

# ### Creating tensors with tf.Variable ###
# changeable_tensor = tf.Variable([10, 7])
# print(changeable_tensor)
# changeable_tensor[0].assign(7)
# print(changeable_tensor)

# ### Creating random tensors ###
# random_1 = tf.random.Generator.from_seed(42)
# random_1 = random_1.normal(shape=(3, 2))
# random_2 = tf.random.Generator.from_seed(42)
# random_2 = random_2.normal(shape=(3, 2))
#
# # Checks if each element is the same
# print(random_1 == random_2)

### Shuffle a tensor ###
# # As long as either the global seed, or operation seed is set, random will always be the same
# # But combinations of setting the seeds gives different values
# tf.random.set_seed(42)
# not_shuffled = tf.constant([[1, 2],
#                             [3, 4],
#                             [5, 6]])
# shuffled = tf.random.shuffle(not_shuffled, seed=42)
# print(shuffled)


# ### Add dimension to a tensor ###
# rank2_tensor = tf.constant([[1, 2, 5],
#                             [3, 4, 6]])
# rank3_tensor = rank2_tensor[..., tf.newaxis]
# same_rank3_tensor = tf.expand_dims(rank2_tensor, axis=1)
# print(rank3_tensor)
# print(same_rank3_tensor)