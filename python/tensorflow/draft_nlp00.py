'''https://www.tensorflow.org/alpha/tutorials/keras/basic_text_classification'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

imdb = tf.keras.datasets.imdb
(train_data,train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)


word_to_id = imdb.get_word_index()
word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0 #reserved index 0,1,2,3
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3
id_to_word = {y:x for x,y in word_to_id.items()}
hf_decode_review = lambda x,id_to_word=id_to_word: ' '.join(id_to_word.get(i, '?') for i in x)

train_data_post = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                        value=word_to_id["<PAD>"], padding='post', maxlen=256)

num1 = 10000
train_data_post,val_data_post = train_data_post[:num1], train_data_post[num1:]
train_labels,val_labels = train_labels[:num1],train_labels[num1:]

test_data_post = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                        value=word_to_id["<PAD>"], padding='post', maxlen=256)


vocab_size = 10000
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data_post, train_labels, epochs=40,
                batch_size=512, validation_data=(val_data_post,val_labels), verbose=1)

results = model.evaluate(test_data_post, test_labels)
print(results)

history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
