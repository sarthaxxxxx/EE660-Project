


def binary_sentence_transfoemr(data_sp):
  print("""### Binary classifer using Sentence Transformer""")

  ### Preprocessing
  chars = ['Stan', 'Kyle', 'Cartman']
  data_sp['binaryClass'] = 'NM'
  data_sp.loc[data_sp['Character'].isin(chars), "binaryClass"] = 'M'
  data_s = data_sp.copy()
  data_s.binaryClass.value_counts().values

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


  #downsampling the majority class
  undersample = RandomUnderSampler(sampling_strategy='majority')
  X_under, y_under = undersample.fit_resample(data_s[['cleanText']], data_sp['binaryClass'])

  label_map = {cat:index for index,cat in enumerate(np.unique(y_under))}
  y_prep = np.asarray([label_map[l] for l in y_under])

  data_under = pd.DataFrame({'X': X_under['cleanText'], 'y': y_prep})
  data_under.head()

  """### Sentence Transformer"""

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = SentenceTransformer('all-MiniLM-L6-v2', device = device)

  X_embed = []
  for i in X_under['cleanText']:
    embeddings = model.encode(i)
    X_embed.append(embeddings)
  len(X_embed)

  """### SVD"""

  from sklearn import preprocessing, decomposition
  from sklearn.decomposition import TruncatedSVD

  svd = decomposition.TruncatedSVD(n_components=180)
  svd.fit(X_embed)
  x_comps = svd.transform(X_embed)
  scl = preprocessing.StandardScaler()
  scl.fit(x_comps)
  x_comps = scl.transform(x_comps)

  """### Model Training & Testing"""

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
  print('precison: ', precision_score(y_pred, y_test))
  print('recall  : ', recall_score(y_pred, y_test))
  print('f1 score: ', f1_score(y_pred, y_test))
  print('\n')
  rfc = RandomForestClassifier()
  rfc.fit(x_train,y_train)
  y_pred = rfc.predict(x_test)
  print('RF Classifier')
  print("accuracy: " ,accuracy_score(y_pred,y_test))
  print('precison: ', precision_score(y_pred, y_test))
  print('recall  : ', recall_score(y_pred, y_test))
  print('f1 score: ', f1_score(y_pred, y_test))
  print('\n')
  logreg = LogisticRegression()
  logreg.fit(x_train,y_train)
  y_pred = logreg.predict(x_test)
  print('LogisticRegression')
  print("accuracy: " ,accuracy_score(y_pred,y_test))
  print('precison: ', precision_score(y_pred, y_test))
  print('recall  : ', recall_score(y_pred, y_test))
  print('f1 score: ', f1_score(y_pred, y_test))
  print('\n')
  xgb_cl = xgb.XGBClassifier()
  xgb_cl.fit(x_train,y_train)
  y_pred = xgb_cl.predict(x_test)
  print('XGBoost Classifier')
  print("accuracy: " ,accuracy_score(y_pred,y_test))
  print('precison: ', precision_score(y_pred, y_test))
  print('recall  : ', recall_score(y_pred, y_test))
  print('f1 score: ', f1_score(y_pred, y_test))
  print('\n')
  gnb = GaussianNB()
  gnb.fit(x_train,y_train)
  y_pred = gnb.predict(x_test)
  print('Gaussian Naive Bayes')
  print("accuracy: " ,accuracy_score(y_pred,y_test))
  print('precison: ', precision_score(y_pred, y_test))
  print('recall  : ', recall_score(y_pred, y_test))
  print('f1 score: ', f1_score(y_pred, y_test))
  print('\n')
  bnb = BernoulliNB()
  bnb.fit(x_train,y_train)
  y_pred = bnb.predict(x_test)
  print('Bernoulli Naive Bayes')
  print("accuracy: " ,accuracy_score(y_pred,y_test))
  print('precison: ', precision_score(y_pred, y_test))
  print('recall  : ', recall_score(y_pred, y_test))
  print('f1 score: ', f1_score(y_pred, y_test))

  print('\n')
  clf = SVC(kernel = 'linear', degree=3, gamma='auto')
  scores = cross_val_score(clf, x_train, y_train, cv=5)
  print('SVM cross validation scores:', scores)