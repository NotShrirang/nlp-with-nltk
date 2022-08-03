from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello Mr. Smith, how are you you doing today? The weather is great and Python is awesome. The sky is great today!"

# Tokenize Sentence
print(sent_tokenize(example_text))

# Tokenize Words
print(word_tokenize(example_text))