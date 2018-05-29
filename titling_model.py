import keras
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D, Input, LSTM, Embedding, TimeDistributed, Reshape, Activation, Concatenate
# some code borrowed from https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

def create_titling_model(embedding_matrix, num_subreddits=20, max_len=20):
        base_model = VGG16(weights='imagenet', include_top=False)
        for layer in base_model.layers:
            layer.trainable = False

        vocab_size = embedding_matrix.shape[0]
        word_embeddings_size = embedding_matrix.shape[1]

        features = base_model.output
        features = GlobalAveragePooling2D()(features)
        features = Dense(word_embeddings_size - num_subreddits, activation='relu')(features)

        one_hot_subreddit = Input(shape=(num_subreddits,), dtype='float32', name='subreddit_input')
        features_concat = Concatenate()([features, one_hot_subreddit])

        titles = Input(shape=(max_len,), dtype='int32', name='titles_input')

        embedding_layer = Embedding(vocab_size,
                                word_embeddings_size,
                                weights=[embedding_matrix],
                                input_length=max_len,
                                mask_zero=True,
                                trainable=False)
        embeddings = embedding_layer(titles)

        hidden_size = 256
        lstm = LSTM(hidden_size, return_sequences=True)
        features_concat = Reshape((1, -1))(features_concat)
        _ = lstm(features_concat)
        hidden_states = lstm(embeddings)
        scores = TimeDistributed(Dense(vocab_size))(hidden_states)
        probs = Activation('softmax')(scores)

        return Model(inputs=[base_model.inputs[0], one_hot_subreddit, titles], outputs=[probs])
