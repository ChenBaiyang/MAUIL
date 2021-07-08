# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import json,pickle,os,time
from gensim import utils,models
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from ge import Struc2Vec, LINE

def preproc(docs,min_len=2,max_len=15):
    
    for i in range(len(docs)):
        docs[i] = [token for token in 
                    utils.tokenize(docs[i],
                                   lower=True,
                                   deacc=True,
                                   errors='ignore')
                    if min_len <= len(token) <= max_len]
    
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    # e.g. years->year, models->model, not including: modeling->modeling
    
    # NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    stop_words = set(stop_words)
    
    docs = [[word for word in document if word not in stop_words] for document in docs]
    
    # Build the bigram and trigram models
    bigram = models.Phrases(docs, min_count=5, threshold=0.1) # higher threshold fewer phrases.
    trigram = models.Phrases(bigram[docs], threshold=0.1)  
    
    # Get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    # Add bigrams and trigrams to docs.
    docs = [bigram_mod[doc] for doc in docs]
    docs = [trigram_mod[bigram_mod[doc]] for doc in docs]
    
    return docs

def topic_embed(docs,dim=32):
    dict_ = Dictionary(docs)
    
    # Filter out words that occur less than 10 documents, or more than 50% of the documents.
    dict_.filter_extremes(no_below=10, no_above=0.5)
    
    corpus = [dict_.doc2bow(doc) for doc in docs]
    
    print('Number of unique tokens: %d' % len(dict_))
    print('Number of documents: %d' % len(corpus))
    
      
    # Train LDA model.
    # Make a index to word dictionary.
    _ = dict_[0] # 这里可能是bug，必须要调用一下dictionary才能确保被加载

    id2word = dict_.id2token
    
    print(time.ctime(),'\tLearning topic model...')
    model = models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=256,
        passes=4,
        #alpha='asymmetric',  # 1/topic_num
        #eta='asymmetric',    # 1/topic_num
        #distributed=True,
        iterations=100,
        num_topics=dim,
        eval_every=1,
        minimum_probability=1e-13,
        random_state=0)
            
    topic_dist = model.get_document_topics(corpus)
    # 如果emb的维度低于dimension,则说明部分话题维度太小被省略了，则需要进行填充
    embed = []
    for i in range(len(corpus)):
        emb = []
        topic_i = topic_dist[i]

        if len(topic_i) < dim:
            topic_i = dict(topic_i)
            for j in range(dim):
                if j in topic_i.keys():
                    emb.append(topic_i[j])
                else:
                    emb.append(1e-13)
        else:
            emb = np.array(topic_i,dtype=np.float64)[:,1]

        embed.append(emb)
        
    embed = np.array(embed)
    
    return embed

def word_embed(docs, ex_corpus=None,dim=32,lamb=0.1,ave_neighbors=False,g1=None,g2=None):
    print(time.ctime(),'\tLearning word vectors...')
    if ex_corpus is not None:
        model = Word2Vec(sentences=ex_corpus,size=dim,workers=os.cpu_count()-1)
    else:
        model = Word2Vec(sentences=docs,size=dim,workers=os.cpu_count()-1)
    
    embed = []
    for doc in docs:
        emb = np.full(dim,1e-13,dtype=np.float64)    
        for word in doc:
            if word in model.wv:
                emb += model.wv[word]
        embed.append(emb)
    embed = np.array(embed)
    

    if ave_neighbors:
        embed_copy = embed.copy()
        for i in range(len(docs)):
            emb = np.full(dim,1e-13,dtype=np.float64)
            if i < len(g1):
                for j in g1.neighbors(i):
                    emb += embed_copy[j]
            else:
                for j in g2.neighbors(i):
                    emb += embed_copy[j]
            embed[i] = (1-lamb)*embed[i] + lamb*emb
        
    return embed

def char_embed(docs,dim=16):
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    docs = [[w for w in list(doc) if 
          not w.isnumeric() 
          and not w.isspace()
          and not w in punc] for doc in docs]

    # Build the bigram and trigram models
    bigram = models.Phrases(docs, min_count=5, threshold=0.1) # higher threshold fewer phrases.
    trigram = models.Phrases(bigram[docs], threshold=0.1)  
    
    # Get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    docs = [bigram_mod[doc] for doc in docs]
    docs = [trigram_mod[bigram_mod[doc]] for doc in docs]

    dict_ = Dictionary(docs)

    docs = [dict_.doc2bow(doc) for doc in docs]
    
    data = np.zeros((len(docs),len(dict_)))
    for n,values in enumerate(docs):
        for idx,value in values:
            data[n][idx] = value

    #from tensorflow.contrib.layers import dropout
    #keep_prob = 0.7
    
    g = tf.get_default_graph()
    #is_training = tf.placeholder_with_default(False,shape=(),name='is_training')
    X = tf.placeholder(tf.float64,shape=[None,data.shape[1]])
    #X_drop = dropout(X,keep_prob,is_training) #考虑引入噪音
    #考虑引入权重绑定
    hidden = fully_connected(X,dim,activation_fn=None)
    outputs = fully_connected(hidden,data.shape[1],activation_fn=None)

    loss = tf.reduce_mean(tf.square(outputs-X))
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session(graph=g,config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            _, embed,loss_ = sess.run([train_op,hidden,loss],feed_dict={X:data})
            if i%10 == 0:
                #print(loss_)
                pass

    return embed

def network_embed(G,dim=16,l_walks=6,n_walks=32,method="line",order='all',workers=os.cpu_count()-1):

    if not nx.is_directed(G):
        G = G.to_directed()

    G = nx.relabel_nodes(G,lambda x: str(x))

    if method == 'struc2vec':
        model = Struc2Vec(G, walk_length=l_walks, num_walks=n_walks, workers=workers, verbose=10, 
                          opt3_num_layers=4,temp_path='./temp_struc2vec_seperated/')
        model.train(embed_size=dim)
        embeddings = model.get_embeddings()
        
    elif method == 'line':
        print(dim)
        model = LINE(G, embedding_size=dim, order=order)
        model.train(batch_size=1024, epochs=50, verbose=2)
        embeddings = model.get_embeddings()

    else:
        raise NotImplementedError("Network embedding method: %s not implemented."%method)

    return embeddings

def embed_dblp():
    print(time.ctime(),'\tLoading data...')
    g1,g2 = pickle.load(open('../data/dblp/networks','rb'))
    print(time.ctime(),'\t Size of two networks:',len(g1),len(g2))
    attrs = pickle.load(open('../data/dblp/attrs','rb'))
    char_corpus,word_corpus,topic_corpus = [],[],[]    
    for i in range(len(attrs)):
        v = attrs[i]
        char_corpus.append(v[0]) 
        word_corpus.append(v[1]) 
        topic_corpus.append(v[2])
        # The index number is the node id of users in the two networks.
    word_corpus = preproc(word_corpus)
    topic_corpus = preproc(topic_corpus)
    
    for seed in range(3):
        for d in [100]:
	        print(time.ctime(),'\tCharacter level attributes embedding...')
	        emb_c = char_embed(char_corpus,dim=d)
	        
	        print(time.ctime(),'\tWord level attributes embedding...')
	        emb_w = word_embed(word_corpus,lamb=0.1,dim=d,ave_neighbors=True,g1=g1,g2=g2)
	        
	        print(time.ctime(),'\tTopic level attributes embedding...')
	        emb_t = topic_embed(topic_corpus,dim=d)
	        
	        print(time.ctime(),'\tNetwork 1 embedding...')
	        emb_g1 = network_embed(g1,method='line',dim=d)
	        
	        print(time.ctime(),'\tNetwork 2 embedding...')
	        emb_g2 = network_embed(g2,method='line',dim=d)
	        emb_g1.update(emb_g2)
	        emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

	        # Standardization
	        emb_c = (emb_c-np.mean(emb_c,axis=0,keepdims=True))/np.std(emb_c,axis=0,keepdims=True)
	        emb_w = (emb_w-np.mean(emb_w,axis=0,keepdims=True))/np.std(emb_w,axis=0,keepdims=True)
	        emb_t = (emb_t-np.mean(emb_t,axis=0,keepdims=True))/np.std(emb_t,axis=0,keepdims=True)
	        emb_s = (emb_s-np.mean(emb_s,axis=0,keepdims=True))/np.std(emb_s,axis=0,keepdims=True)
	        
	        # Saving embeddings
	        pickle.dump((emb_c,emb_w,emb_t,emb_s),open('../emb/emb_dblp_seed_{}_dim_{}'.format(seed,d),'wb'))

def embed_wd():
    import jieba,zhconv,re
    from pypinyin import lazy_pinyin
    from gensim.models.word2vec import LineSentence
    from gensim.corpora import WikiCorpus

    def tokenizer_cn(text, token_min_len=10, token_max_len=100, lower=False):
        text = zhconv.convert(text,'zh-hans').strip() #Standardize to simple Chinese
        text = p.sub('',text)
        return jieba.lcut(text)

    def preproc_cn(docs,min_len=2,max_len=15):
        docs = [tokenizer_cn(doc) for doc in docs]
        # Removing Stop words
        stop_words = pickle.load(open('../data/wd/stop_words_cn.pkl','rb'))
        stop_words = set(stop_words)
        docs = [[word for word in document if word not in stop_words] for document in docs]
        return docs

    def process_wiki(inp, outp, dct):
        _ = dct[0]
        output = open(outp, 'w', encoding='utf-8')
        wiki = WikiCorpus(inp, processes=os.cpu_count()-2,lemmatize=False,
                          dictionary=dct,article_min_tokens=10,
                          lower=False) # It takes about 16 minutes by 10 core cpu.
        count=0
        for words in wiki.get_texts():
            words = [" ".join(tokenizer_cn(w)) for w in words]
            output.write(' '.join(words) + '\n')
            count+=1
            if count%10000==0:
                print('Finished %d-67'%count//10000)
        output.close()

    def topic_embed_cn(docs,dim=32):
        return topic_embed(docs,dim=dim)

    def word_embed_cn(docs,dim=64,lamb=0.05,ave_neighbors=False,g1=None,g2=None,
                      ex_corpus=False,ex_corpus_fname=None,ex_corpus_xml=None):
        if ex_corpus:
            dct = Dictionary(docs)
            if not os.path.exists(ex_corpus_fname):
                print('Processing {}'.format(ex_corpus_fname))
                process_wiki(ex_corpus_xml,ex_corpus_fname,dct=dct)
            iter_ = LineSentence(ex_corpus_fname)
            return word_embed(docs,ex_corpus=iter_,lamb=lamb,dim=dim,ave_neighbors=ave_neighbors,g1=g1,g2=g2)
        else:
            return word_embed(docs,lamb=lamb,dim=dim,ave_neighbors=ave_neighbors,g1=g1,g2=g2)

    def char_embed_cn(docs,dim=16):
        docs = [''.join(lazy_pinyin(doc)).lower() for doc in docs]
        return char_embed(docs,dim=dim)

    p = re.compile('[^\u4e00-\u9fa5]')
    ex_corpus_xml = '../data/wd/zhwiki-latest-pages-articles.xml.bz2'
    ex_corpus_fname = '../data/wd/zhwiki_corpus'
    
    print(time.ctime(),'\tLoading data...')
    g1,g2 = pickle.load(open('../data/wd/networks','rb'))
    print(time.ctime(),'\t Size of two networks:',len(g1),len(g2))    
    attrs = pickle.load(open('../data/wd/attrs','rb'))
    char_corpus,word_corpus,topic_corpus = [],[],[]    
    for i in range(len(attrs)):
        v = attrs[i]
        char_corpus.append(v[0]) 
        word_corpus.append(v[1]) 
        topic_corpus.append(v[2])
        # The index number is the node id of users in the two networks.

    print(time.ctime(),'\tPreprocessing...')
    word_corpus = preproc_cn(word_corpus)
    topic_corpus = preproc_cn(topic_corpus)
    
    for seed in range(3):
        for d in [100]:
	        print(time.ctime(),'\tCharacter level attributes embedding...')
	        emb_c = char_embed_cn(char_corpus,dim=d)
	        
	        print(time.ctime(),'\tWord level attributes embedding...')
	        emb_w = word_embed_cn(word_corpus,lamb=0.1,dim=d,ave_neighbors=True,g1=g1,g2=g2,
	                              ex_corpus=True,ex_corpus_fname=ex_corpus_fname,ex_corpus_xml=ex_corpus_xml)
	        print(time.ctime(),'\tTopic level attributes embedding...')
	        emb_t = topic_embed_cn(topic_corpus,dim=d)
	        
	        print(time.ctime(),'\tNetwork 1-1 embedding...')
	        emb_g1 = network_embed(g1,method='line',dim=d)
	        
	        print(time.ctime(),'\tNetwork 1-2 embedding...')
	        emb_g2 = network_embed(g2,method='line',dim=d)
	        emb_g1.update(emb_g2)
	        emb_s = np.array([emb_g1[str(i)] for i in range(len(emb_g1))])

	        # Standardization
	        emb_c = (emb_c-np.mean(emb_c,axis=0,keepdims=True))/np.std(emb_c,axis=0,keepdims=True)
	        emb_w = (emb_w-np.mean(emb_w,axis=0,keepdims=True))/np.std(emb_w,axis=0,keepdims=True)
	        emb_t = (emb_t-np.mean(emb_t,axis=0,keepdims=True))/np.std(emb_t,axis=0,keepdims=True)
	        emb_s = (emb_s-np.mean(emb_s,axis=0,keepdims=True))/np.std(emb_s,axis=0,keepdims=True)

	        # Saving embeddings
	        pickle.dump((emb_c,emb_w,emb_t,emb_s),open('../emb/emb_wd_seed_{}_dim_{}'.format(seed,d),'wb'))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
inputs=input('Selecte the dataset(0/1): ')
if int(inputs) == 0:
	print('Embedding dataset: dblp')
	embed_dblp()
else:
	print('Embedding dataset: wd')
	embed_wd()