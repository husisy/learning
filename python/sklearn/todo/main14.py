from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def print_top_words(model, feature_names):
    for ind1, weight in enumerate(model.components_):
        tmp1 = ' '.join(feature_names[x] for x in weight.argsort()[:-20:-1])
        print('topic #{}: {}'.format(ind1, tmp1))
    print()

z1 = fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes')).data[:2000]

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
z2_tfidf = tfidf_vectorizer.fit_transform(z1)
tfidf_name = tfidf_vectorizer.get_feature_names()

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
z2_tf = tf_vectorizer.fit_transform(z1)
tf_name = tf_vectorizer.get_feature_names()

nmf = NMF(n_components=10, alpha=0.1, l1_ratio=0.5).fit(z2_tfidf)
print_top_words(nmf, tfidf_name)


nmf = NMF(n_components=10, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1, l1_ratio=.5).fit(z2_tfidf)
print_top_words(nmf, tfidf_name)


lda = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online', learning_offset=50).fit(z2_tf)
print_top_words(lda, tf_name)
