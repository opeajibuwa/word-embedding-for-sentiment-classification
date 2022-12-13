import numpy as np
import pandas as pd
import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score as adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics

class GloVe_Embedder:
    def __init__(self, path):
        self.embedding_dict = {}
        self.embedding_array = []
        self.unk_emb = 0
        # Adapted from https://stackoverflow.com/questions/37793118/load-pretrained-GloVe-vectors-in-python
        with open(path,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.embedding_dict[word] = embedding
                self.embedding_array.append(embedding.tolist())
        self.embedding_array = np.array(self.embedding_array)
        self.embedding_dim = len(self.embedding_array[0])
        self.vocab_size = len(self.embedding_array)
        self.unk_emb = np.zeros(self.embedding_dim)

    # Check if the provided embedding is the unknown embedding.
    def is_unk_embed(self, embed):
        return np.sum((embed - self.unk_emb) ** 2) < 1e-7
    
    # Check if the provided string is in the vocabulary.
    def token_in_vocab(self, x):
        if x in self.embedding_dict and not self.is_unk_embed(self.embedding_dict[x]):
            return True
        return False

    # Returns the embedding for a single string and prints a warning if
    # the string is unknown to the vocabulary.
    # 
    # If indicate_unk is set to True, the return type will be a tuple of 
    # (numpy array, bool) with the bool indicating whether the returned 
    # embedding is the unknown embedding.
    #
    # If warn_unk is set to False, the method will no longer print warnings
    # when used on unknown strings.
    def embed_str(self, x, indicate_unk = False, warn_unk = True):
        if self.token_in_vocab(x):
            if indicate_unk:
                return (self.embedding_dict[x], False)
            else:
                return self.embedding_dict[x]
        else:
            if warn_unk:
                    print("Warning: provided word is not part of the vocabulary!")
            if indicate_unk:
                return (self.unk_emb, True)
            else:
                return self.unk_emb

    # Returns an array containing the embeddings of each vocabulary token in the provided list.
    #
    # If include_unk is set to False, the returned list will not include any unknown embeddings.
    def embed_list(self, x, include_unk = True):
        if include_unk:
            embeds = [self.embed_str(word, warn_unk = False).tolist() for word in x]
        else:
            embeds_with_unk = [self.embed_str(word, indicate_unk=True, warn_unk = False) for word in x]
            embeds = [e[0].tolist() for e in embeds_with_unk if not e[1]]
            if len(embeds) == 0:
                print("No known words in input:" + str(x))
                embeds = [self.unk_emb.tolist()]
        return np.array(embeds)
    
    # Finds the vocab words associated with the k nearest embeddings of the provided word. 
    # Can also accept an embedding vector in place of a string word.
    # Return type is a nested list where each entry is a word in the vocab followed by its 
    # distance from whatever word was provided as an argument.
    def find_k_nearest(self, word, k, warn_about_unks = True):
        if type(word) == str:
            word_embedding, is_unk = self.embed_str(word, indicate_unk = True)
        else:
            word_embedding = word
            is_unk = False
        if is_unk and warn_about_unks:
            print("Warning: provided word is not part of the vocabulary!")

        all_distances = np.sum((self.embedding_array - word_embedding) ** 2, axis = 1) ** 0.5
        distance_vocab_index = [[w, round(d, 5)] for w,d,i in zip(self.embedding_dict.keys(), all_distances, range(len(all_distances)))]
        distance_vocab_index = sorted(distance_vocab_index, key = lambda x: x[1], reverse = False)
        return distance_vocab_index[:k]

    def save_to_file(self, path):
        with open(path, 'w') as f:
            for k in self.embedding_dict.keys():
                embedding_str = " ".join([str(round(s, 5)) for s in self.embedding_dict[k].tolist()])
                string = k + " " + embedding_str
                f.write(string + "\n")


# path to the glove embeddings
path_to_file = "/Users/Opeyemi/Desktop/Machine Learning/homeworks/ia4/GloVe_Embedder_data.txt"
ge = GloVe_Embedder(path_to_file)

# Find the 29 most similar words for 5 seeded words
array_of_words = ["flight", "good", "terrible", "help", "late"]
sw_dict = {}

for word in array_of_words:
    sw_list = []
    similar_words = ge.find_k_nearest(word, 30)
    similar_words_ = similar_words[1:]
    for item in similar_words_:
        sw_list.append(item[0])
    sw_dict[word] = sw_list
    
print("The 29 most similar words for each word embeddings are listed below: \n")
print(pd.DataFrame(sw_dict))


# Helper functions
def append_list(sim_words, words):
    
    list_of_words = []
    
    for i in range(len(sim_words)):
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
        
    return list_of_words

input_word = 'flight, good, terrible, help, late'
user_input = [x.strip() for x in input_word.split(',')]
result_word = []
    
for words in user_input:
        
        sim_words = ge.find_k_nearest(words, 30)[1:]
        sim_words = [tuple(l) for l in sim_words]
        sim_words = append_list(sim_words, words)
            
        result_word.extend(sim_words)
    
similar_word = [word[0] for word in result_word]
similarity = [word[1] for word in result_word] 
similar_word.extend(user_input)
labels = [word[2] for word in result_word]
label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
color_map = [label_dict[x] for x in labels]

# Apply PCA using sklearn.decomposition.pca to the 150 words and visualize them in 2-d space
def display_pca_scatterplot(model, user_input=None, words=None, label=None, color_map=None, topn=29):
    
    word_vectors = np.array([ge.embed_str(w) for w in words])

    two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]
    # print(two_dim)
    # print(two_dim.shape)

    data = []
    count = 0
    
    for i in range(len(user_input)):

                trace = go.Scatter(
                    x = two_dim[count:count+topn,0], 
                    y = two_dim[count:count+topn,1],
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
                
            
                data.append(trace)
                count = count+topn

    trace_input = go.Scatter(
                    x = two_dim[count:,0], 
                    y = two_dim[count:,1],  
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

            
    data.append(trace_input)
    
# Configure the layout
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1150,
        height = 1150
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
 
display_pca_scatterplot(ge, user_input, similar_word, labels, color_map)

# Sklearn.manifold.TSNE with Euclidean distance to the 150 words and visualize them in 2-d space using the same color mapping
def display_tsne_scatterplot(model, user_input=None, words=None, label=None, color_map=None, perplexity=5, learning_rate = 0, iteration = 0, topn=29):
    
    word_vectors = np.array([ge.embed_str(w) for w in words])
    two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, init='pca', learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]
    # print(two_dim)
    # print(two_dim.shape)
    
    data = []

    count = 0
    for i in range (len(user_input)):

                trace = go.Scatter(
                    x = two_dim[count:count+topn,0], 
                    y = two_dim[count:count+topn,1],  
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
            
                data.append(trace)
                count = count+topn
    
    trace_input = go.Scatter(
                    x = two_dim[count:,0], 
                    y = two_dim[count:,1],  
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

            
    data.append(trace_input)
    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1150,
        height = 1150
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    
display_tsne_scatterplot(ge, user_input, similar_word, labels, color_map, 5, 500, 10000)


# Apply K-means clustering
word_vectors = np.array([ge.embed_str(w) for w in similar_word])
two_dim = TSNE(n_components = 2, random_state=0, perplexity=15, init='pca', learning_rate=500, n_iter=10000).fit_transform(word_vectors)[:,:2]


def kmeans_clustering(k):
    km = KMeans(
        n_clusters=k, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0)

    X = np.array(two_dim)
    km_ = km.fit(X)
    return km_.inertia_
    
n_clusters = np.arange(2, 21)
objs = [kmeans_clustering(cluster) for cluster in n_clusters]


# Plot the kmeans objective as a function of k
plt.figure(figsize=(8, 6))

plt.plot(n_clusters, objs, 'r')
plt.xlabel('no of clusters, k')
plt.ylabel('kmeans objective')

plt.title(r"Plot of kmeans objective versus number of clusters")
plt.savefig('kmeans_obj.jpg')
plt.show()


# Evaluate the clustering solution for different k values using different metrics
array_of_words = ["flight", "good", "terrible", "help", "late"]

ground_label = []
for index, item in enumerate(array_of_words):
    i = 0
    while i <= 28:
        ground_label.append(index)
        i += 1
for index, item in enumerate(array_of_words):
    ground_label.append(index)
ground_label = np.array(ground_label)

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def kmeans_evaluate(k, ground_label):
    km = KMeans(
        n_clusters=k, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0)

    X = np.array(two_dim)
    km_ = km.fit_predict(X)
    pred_label = km_
    purity_sc = purity_score(ground_label, pred_label)                                                        
    adjusted_rs = adjusted_rand_score(ground_label, pred_label)
    normalized_mutualinfo_score = normalized_mutual_info_score(ground_label, pred_label)
    return purity_sc, adjusted_rs, normalized_mutualinfo_score

purity_sc = []
adjusted_rs = []
normalized_mutualinfo_score = []

for cluster in n_clusters:
    metric_1, metric_2, metric_3 = kmeans_evaluate(cluster, ground_label)
    purity_sc.append(metric_1)
    adjusted_rs.append(metric_2)
    normalized_mutualinfo_score.append(metric_3)

# Plot the kmeans evaluation metrics against the number of clusters
plt.figure(figsize=(8, 6))

plt.plot(n_clusters, purity_sc, color='r', label = "purity score")
plt.plot(n_clusters, adjusted_rs, color='b', label = "adjusted rand index")
plt.plot(n_clusters, normalized_mutualinfo_score, color='g', label = "Normalized Mutual Information")

plt.xlabel('no of clusters, k')
plt.ylabel('Metric scores')

plt.title(r"Plot of kmeans evaluation metrics versus number of clusters")
plt.savefig('clustering_metrics.jpg')
plt.show()

# Using word embeddings to improve classification - Using the tf-idf weighted average of the embeddings of the words in a tweet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from collections import defaultdict

tokenizer = nltk.RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()

train_df = pd.read_csv('IA3-train.csv')
test_df = pd.read_csv('IA3-dev.csv')
#!wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip

words = dict()
def add_to_dict(d, filename):
    with open(filename, 'r', encoding="utf8") as f:
        for line in f.readlines():
          line = line.split(' ')

          try:
            d[line[0]] = np.array(line[1:], dtype=float)
          except:
            continue
            
add_to_dict(words, 'glove.6B.50d.txt')
#words

tfidf = TfidfVectorizer(use_idf=True, lowercase=True)
tfidf.fit_transform(train_df['text'])
max_idf = max(tfidf.idf_)
wrd2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

def message_to_token_list(s):
    tokens = tokenizer.tokenize(s)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in words]
    return useful_tokens

# def message_to_word_vectors(message, token_weight = wrd2weight, word_dict=words):
#     processed_list_of_tokens = message_to_token_list(message)

#     vectors = []
#     for token in processed_list_of_tokens:
#         if token not in word_dict:
#           continue

#         weighted_token_vector = word_dict[token] * token_weight[token]
#         print(weighted_token_vector)
#         vectors.append(weighted_token_vector)
#         weighed_token_array = np.array(vectors, dtype=float)
        
#         return np.mean(weighed_token_array, axis=0)

def message_to_word_vectors_(message, word_dict=words):
    processed_list_of_tokens = message_to_token_list(message)

    vectors = []
    for token in processed_list_of_tokens:
        if token not in word_dict:
          continue

        token_vector = word_dict[token]
        # print(token_vector)
        vectors.append(token_vector)
        token_array = np.array(vectors, dtype=float)
        
        return np.mean(token_array, axis=0)

def df_to_X_y(dff):
    y = dff['sentiment'].to_numpy().astype(int)

    all_word_vector_sequences = []

    for index, message in enumerate(dff['text']):
        if len(message_to_token_list(message)) == 0:
            # print(index)
            # print(message)
            message_as_vector_seq = np.zeros(50)
        else:
            message_as_vector_seq = message_to_word_vectors_(message)

        all_word_vector_sequences.append(message_as_vector_seq)
  
    X = np.array(all_word_vector_sequences).astype(float)
    return X, y

X_train, y_train = df_to_X_y(train_df)
X_test, y_test = df_to_X_y(test_df)

def linear_svm(c, X, y):
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y_pred, y)
    print(f"Accuracy on the linear-svm train set with C value {c}: {acc:.4f}", "\t")
    return clf, acc

# c = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
# training_acc = []
# validation_acc = []
# for i in c:
#     linear_clf, train_acc = linear_svm(pow(10, i), X_train, y_train)
#     linear_ypred = linear_clf.predict(X_test)
#     linear_acc = accuracy_score(linear_ypred, y_test)
#     training_acc.append(train_acc)
#     validation_acc.append(linear_acc)
#     print(f"Accuracy on the linear-svm test set with C value 10^{i}: {linear_acc:.4f}")
#     print("---------------------------------------------")

def quadratic_svm(c, X, y):
    clf = svm.SVC(kernel='poly', C=c, degree=2, coef0=10)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y_pred, y)
    print(f"Accuracy on the quadratic-svm train set with C value {c}: {acc:.4f}", "\t")
    return clf, acc

c = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
training_acc = []
validation_acc = []
n_support_vectors_quad = []
for i in c:
    quad_clf, train_acc = quadratic_svm(pow(10, i), X_train, y_train)
    quad_ypred = quad_clf.predict(X_test)
    quad_acc = accuracy_score(quad_ypred, y_test)
    training_acc.append(train_acc)
    validation_acc.append(quad_acc)
    n_support_vectors_quad.append(sum(quad_clf.n_support_))
    print(f"Accuracy on the quadratic-svm test set with C value 10^{i}: {quad_acc:.4f}")
    print("---------------------------------------------")

c = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
fig, ax3 = plt.subplots(figsize=(8, 6), tight_layout=True)
ax3.semilogx(c, training_acc, color='r', marker='o', markerfacecolor='m')
ax3.semilogx(c, validation_acc, color='b', marker='x', markerfacecolor='r')

min_axis = min(min(training_acc), min(validation_acc))
max_axis = max(max(training_acc), max(validation_acc))

ax3.set_ylabel(f'accuracy', color='r')
ax3.set_xlabel(f'c')
ax3.set_xlim([1e-4, 1e5])
ax3.set_ylim(0.5, 1)
ax3.set_title(f"Classification Accuracy for Quadratic kernel SVM", color='k', weight='normal', size=10)
ax3.legend(["training", "validation"], loc="upper left")

plt.savefig("quad_train_dev_acc_cmp.jpg")
print('Done.\n')

def rbf_svm(c, X, y, gamma_val='scale'):
    clf = svm.SVC(kernel='rbf', C=c, gamma=gamma_val)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y_pred, y)
    print(f"Accuracy on the rbf-svm train set with C value {c} and gamma value {gamma_val}: {acc:.4f}", "\t")
    return clf, acc

c = [10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4]
gamma_values = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1]
training_acc_rbf = []
validation_acc_rbf = []

for i in c:
    for gamma in gamma_values:
        rbf_clf, train_acc = rbf_svm(i, X_train, y_train, gamma)
        rbf_ypred = rbf_clf.predict(X_test)
        rbf_acc = accuracy_score(rbf_ypred, y_test)
        training_acc_rbf.append(train_acc)
        validation_acc_rbf.append(rbf_acc)
        print(f"Accuracy on the rbf-svm test set with C value {i} and gamma value {gamma}: {rbf_acc:.4f}")
        print("---------------------------------------------")

# Exploring Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Instantiate and fit the RandomForestClassifier
forest = RandomForestClassifier(random_state = 0)
forest.fit(X_train, y_train)

# Make predictions for the training set
y_pred_train = forest.predict(X_train)

# View accuracy score
accuracy_score(y_train, y_pred_train)

# Make predictions for the test set
y_pred_test = forest.predict(X_test)

# View accuracy score
accuracy_score(y_test, y_pred_test)