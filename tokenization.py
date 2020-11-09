import pandas as pd
import pickle
import re
import javalang


# Load the data:
with open('/homes/cs408/project-3/data/train.pickle', 'rb') as handle:
    train = pickle.load(handle)
with open('/homes/cs408/project-3/data/valid.pickle', 'rb') as handle:
    valid = pickle.load(handle)
with open('/homes/cs408/project-3/data/test.pickle', 'rb') as handle:
    test = pickle.load(handle)

# Tokenize and shape our input:
def custom_tokenize(string):
    try:
        tokens = list(javalang.tokenizer.tokenize(string))
    except:
        return []
    values = []
    for token in tokens:
        # Abstract strings
        if '"' in token.value or "'" in token.value:
            values.append('$STRING$')
        # Abstract numbers (except 0 and 1)
        elif token.value.isdigit() and int(token.value) > 1:
            values.append('$NUMBER$')
        #other wise: get the value
        else:
            values.append(token.value)
    return values


def tokenize_df(df):
    df['instance'] = df['instance'].apply(lambda x: custom_tokenize(x))
    df['context_before'] = df['context_before'].apply(lambda x: custom_tokenize(x))
    df['context_after'] = df['context_after'].apply(lambda x: custom_tokenize(x))
    return df

test = tokenize_df(test)
train = tokenize_df(train)
valid = tokenize_df(valid)

with open('./data/tokenized_train.pickle', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./data/tokenized_valid.pickle', 'wb') as handle:
    pickle.dump(valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./data/tokenized_test.pickle', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

