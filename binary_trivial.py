## Binary - Trivial Classifier



def bianry_trivial(data_sp):

  print("""### Binary Trivial Classifier""")

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

  undersample = RandomUnderSampler(sampling_strategy='majority')
  X_under, y_under = undersample.fit_resample(data_s[['cleanText', 'Season', 'Episode']], data_sp['binaryClass'])

  label_map = {cat:index for index,cat in enumerate(np.unique(y_under))}
  y_prep = np.asarray([label_map[l] for l in y_under])

  data_under = pd.DataFrame({'X': X_under['cleanText'], 'y': y_prep})


  random.shuffle(y_prep)
  y_pred = []
  for i in range(len(y_prep)):
    y_pred.append(random.choice([0,1]))

  y_train = y_prep[:39285]
  y_test = y_prep[39285:]
  y_test_pred = y_pred[39285:]

  print('Binary Trivial Classifier')
  print("accuracy: " ,accuracy_score(y_test_pred,y_test))
  print('precison: ', precision_score(y_test_pred, y_test))
  print('recall  : ', recall_score(y_test_pred, y_test))
  print('f1 score: ', f1_score(y_test_pred, y_test))
  print('\n')

  print("""### Multi Class Trivial Classifier""")

  chars = ['Stan', 'Kyle', 'Cartman']
  data_s = data_sp[data_sp['Character'].isin(chars)].copy()
  data_s['cleanText']=data_s['Line'].map(lambda s:preprocess(s))

  undersample = RandomUnderSampler(sampling_strategy='majority')
  X_under, y_under = undersample.fit_resample(data_s[['cleanText', 'Season', 'Episode']], data_s['Character'])

  label_map = {cat:index for index,cat in enumerate(np.unique(y_under))}
  y_prep = np.asarray([label_map[l] for l in y_under])

  data_under = pd.DataFrame({'X': X_under['cleanText'], 'y': y_prep})


  random.shuffle(y_prep)
  y_pred = []
  for i in range(len(y_prep)):
    y_pred.append(random.choice([0,1, 2]))

  y_train = y_prep[:15000]
  y_test = y_prep[15000:]
  y_test_pred = y_pred[15000:]

  print('Multi-Class Trivial Classifier')
  print("accuracy: " ,accuracy_score(y_test_pred,y_test))
  print('precison: ', precision_score(y_test_pred, y_test, average = 'weighted'))
  print('recall  : ', recall_score(y_test_pred, y_test, average = 'weighted'))
  print('f1 score: ', f1_score(y_test_pred, y_test, average = 'weighted'))
  print('\n')