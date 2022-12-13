### Multi Class - word2vec



def multi_word2vec(data_sp):

    print("""### Multi Class using word2vec""")

    """## Preprocessing"""

    data_sp = pd.read_csv('./data/All-seasons.csv')
    data_sp.head()

    chars = ['Stan', 'Kyle', 'Cartman']
    data_s = data_sp[data_sp['Character'].isin(chars)].copy()
    data_s.Character.value_counts().values

    def preprocess(sentence):
        sentence=str(sentence)
        sentence = sentence.lower()
        sentence=sentence.replace('{html}',"") 
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        rem_url=re.sub(r'http\S+', '',cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)  
        return " ".join(tokens)

    data_s['cleanText']=data_s['Line'].map(lambda s:preprocess(s))
    data_s.head()

    #downsampling themajority class
    from imblearn.under_sampling import RandomUnderSampler

    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(data_s[['cleanText', 'Season', 'Episode']], data_s['Character'])

    label_map = {cat:index for index,cat in enumerate(np.unique(y_under))}
    y_prep = np.asarray([label_map[l] for l in y_under])

    data_under = pd.DataFrame({'X': X_under['cleanText'], 'y': y_prep})

    print(y_under.value_counts().values)

    """### word2vec"""

    response_new = X_under.cleanText.apply(gensim.utils.simple_preprocess)
    model=gensim.models.Word2Vec(window=5, min_count=2, workers=4, sg=0)
    model.build_vocab(response_new, progress_per=1000)
    model.train(response_new, total_examples=model.corpus_count, epochs=model.epochs)

    class Sequencer():
        def __init__(self,
                    all_words,
                    max_words,
                    seq_len,
                    embedding_matrix
                    ):
            
            self.seq_len = seq_len
            self.embed_matrix = embedding_matrix
            temp_vocab = list(set(all_words))
            self.vocab = []
            self.word_cnts = {}
            for word in temp_vocab:
                count = len([0 for w in all_words if w == word])
                self.word_cnts[word] = count
                counts = list(self.word_cnts.values())
                indexes = list(range(len(counts)))
            
            cnt = 0
            while cnt + 1 != len(counts):
                cnt = 0
                for i in range(len(counts)-1):
                    if counts[i] < counts[i+1]:
                        counts[i+1],counts[i] = counts[i],counts[i+1]
                        indexes[i],indexes[i+1] = indexes[i+1],indexes[i]
                    else:
                        cnt += 1
            
            for ind in indexes[:max_words]:
                self.vocab.append(temp_vocab[ind])
                        
        def textToVector(self,text):
            tokens = text.split()
            len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
            vec = []
            for tok in tokens[:len_v]:
                try:
                    vec.append(self.embed_matrix[tok])
                except Exception as E:
                    pass
            
            last_pieces = self.seq_len - len(vec)
            for i in range(last_pieces):
                vec.append(np.zeros(100,))
            
            return np.asarray(vec).flatten()

    sequencer = Sequencer(all_words = [token for seq in response_new for token in seq],
                max_words = 1200,
                seq_len = 15,
                embedding_matrix = model.wv
                )

    x_vecs = np.asarray([sequencer.textToVector(" ".join(seq)) for seq in response_new])
    print(x_vecs.shape)

    """### SVD"""

    # SVD
    from sklearn import preprocessing, decomposition
    from sklearn.decomposition import TruncatedSVD
    svd = decomposition.TruncatedSVD(n_components=180)

    svd.fit(x_vecs)
    x_comps = svd.transform(x_vecs)
    scl = preprocessing.StandardScaler()
    scl.fit(x_comps)
    x_comps = scl.transform(x_comps)

    """### Model Training and Testing"""

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB,BernoulliNB
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.model_selection import cross_val_score

    x_train,x_test,y_train,y_test = train_test_split(x_comps,y_prep,test_size=0.2,random_state=42)

    svm_classifier = SVC()
    svm_classifier.fit(x_train,y_train)
    y_pred = svm_classifier.predict(x_test)
    print('SVM Classifier')
    print("accuracy: " ,accuracy_score(y_pred,y_test))
    print('precison: ', precision_score(y_pred, y_test, average='weighted'))
    print('recall  : ', recall_score(y_pred, y_test, average='weighted'))
    print('f1 score: ', f1_score(y_pred, y_test, average='weighted'))
    print('\n')
    rfc = RandomForestClassifier()
    rfc.fit(x_train,y_train)
    y_pred = rfc.predict(x_test)
    print('RF Classifier')
    print("accuracy: " ,accuracy_score(y_pred,y_test))
    print('precison: ', precision_score(y_pred, y_test, average='weighted'))
    print('recall  : ', recall_score(y_pred, y_test, average='weighted'))
    print('f1 score: ', f1_score(y_pred, y_test, average='weighted'))
    print('\n')
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    y_pred = logreg.predict(x_test)
    print('LogisticRegression')
    print("accuracy: " ,accuracy_score(y_pred,y_test))
    print('precison: ', precision_score(y_pred, y_test, average='weighted'))
    print('recall  : ', recall_score(y_pred, y_test, average='weighted'))
    print('f1 score: ', f1_score(y_pred, y_test, average='weighted'))
    print('\n')
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.fit(x_train,y_train)
    y_pred = xgb_cl.predict(x_test)
    print('XGBoost Classifier')
    print("accuracy: " ,accuracy_score(y_pred,y_test))
    print('precison: ', precision_score(y_pred, y_test, average='weighted'))
    print('recall  : ', recall_score(y_pred, y_test, average='weighted'))
    print('f1 score: ', f1_score(y_pred, y_test, average='weighted'))
    print('\n')
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred = gnb.predict(x_test)
    print('Gaussian Naive Bayes')
    print("accuracy: " ,accuracy_score(y_pred,y_test))
    print('precison: ', precision_score(y_pred, y_test, average='weighted'))
    print('recall  : ', recall_score(y_pred, y_test, average='weighted'))
    print('f1 score: ', f1_score(y_pred, y_test, average='weighted'))
    print('\n')
    bnb = BernoulliNB()
    bnb.fit(x_train,y_train)
    y_pred = bnb.predict(x_test)
    print('Bernoulli Naive Bayes')
    print("accuracy: " ,accuracy_score(y_pred,y_test))
    print('precison: ', precision_score(y_pred, y_test, average='weighted'))
    print('recall  : ', recall_score(y_pred, y_test, average='weighted'))
    print('f1 score: ', f1_score(y_pred, y_test, average='weighted'))

    print('\n')
    clf = SVC(kernel = 'linear', degree=3, gamma='auto')
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print('SVM cross validation scores:', scores)