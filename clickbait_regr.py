import json
import sys
import os
import libspacy
import string
# import libgrams
import numpy as np
from math import *
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.utils import shuffle
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from collections import Counter
#from sklearn.manifold import TSNE
import libglove

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# exclude = set(string.punctuation)

def main():
  train_dir1 = 'clickbait17-train-170331'
  instances_filename = 'instances.jsonl'
  truths_filename = 'truth.jsonl'
  train_dir2 = 'clickbait17-train-170630'
  instances_filename = 'instances.jsonl'
  truths_filename = 'truth.jsonl'
  raw_data={} #Data set indexed by the id, id is a string
  raw_truths={} #Truths indexed by the id, id is a string


  abs_path = os.path.join(train_dir1, instances_filename)
  fp = open(abs_path)
  for line in fp:
    json_obj = json.loads(line)
    #print json_obj['postText']
    item_id = json_obj['id']
    raw_data[item_id]=json_obj

  #print raw_data

  fp.close()
  abs_path = os.path.join(train_dir2, instances_filename)
  fp = open(abs_path)
  for line in fp:
    json_obj = json.loads(line)
    #print json_obj['postText']
    item_id = json_obj['id']
    raw_data[item_id]=json_obj


  abs_path = os.path.join(train_dir1, truths_filename)
  fp = open(abs_path)
  for line in fp:
    json_obj = json.loads(line)
    item_id = json_obj['id']
    raw_truths[item_id]=json_obj

  abs_path = os.path.join(train_dir2, truths_filename)
  fp = open(abs_path)
  for line in fp:
    json_obj = json.loads(line)
    item_id = json_obj['id']
    raw_truths[item_id]=json_obj

  feature = 'postText'

  raw_cb = []
  y_cb=[]
  y_labels=[]
  item_ids=[]
  cb=[]
  nocb=[]
  item_ids_cb=[]
  item_ids_nocb=[]
  all_words=[]
  for item_id in tqdm(raw_data):
    rating = raw_truths[item_id]['truthMean']
    postText = raw_data[item_id].get('postText','')
    postText = clean_str(postText[0].strip())
    postText = clean_title(postText)
    label=raw_truths[item_id]['truthClass']
    #sys.exit()

    item = raw_data[item_id]
    item['rating'] = rating
    item['postText']=postText
    item['id']=item_id

    postWords = word_tokenize(postText)
    item['postWords']=postWords

    postWords = [w for w in postWords if not w in stop_words] 
    # postWords = [w for w in postWords if not w in exclude] 

    pos = nltk.pos_tag(postWords)
    item['pos']=pos

    lemmatized = []
    
    for word, tag in pos :
      if word == 'http' or word == ':' or word == '//t': continue
      if word.isdigit() :
        lemmatized.append('[n]')
        continue

      if tag[0] == 'J':
        part = wordnet.ADJ
      elif tag[0] == 'N':
        part = wordnet.NOUN
      elif tag[0] == 'V':
        part = wordnet.VERB
      elif tag[0] == 'R':
        part = wordnet.ADV
      else :
        part = wordnet.NOUN
      
      lemmatized.append(word)
      
    item['lemmatized'] = lemmatized

    all_words.extend(lemmatized)

    item['lemmaSent'] = ' '.join(lemmatized)

    if '[n]' not in item['lemmaSent']: continue

    raw_cb.append(item)
    if label=='clickbait':
      cb.append(item)
    else:
      nocb.append(item)
    label = rating_to_class(rating)
    y_cb.append(rating)
    y_labels.append(label)
    item_ids.append(item_id)
  fp.close()

  # print(Counter(all_words).most_common(10))

  nocb_filtered = []
  cb_filtered = []

  print('Filtering nocb')
  for item in tqdm(nocb):
    sent_array = item['lemmaSent'].split()
    if len(sent_array) < 2: continue
    word_idx = sent_array.index('[n]')
    for comparison in nocb:
      if item['postText'] == comparison['postText']: continue
      comparison_sent_array = comparison['lemmaSent'].split()
      if len(comparison_sent_array) < 2: continue
      comparison_idx = comparison_sent_array.index('[n]')
      if(word_idx > len(sent_array)-2 and comparison_idx < 1) or (word_idx < 1 and comparison_idx > len(comparison_sent_array) -2): continue
      if(word_idx > len(sent_array)-2 or comparison_idx > len(comparison_sent_array) -2):
        word_vec = libspacy.get_vector(' '.join([item['lemmaSent'][word_idx-2], item['lemmaSent'][word_idx-1], item['lemmaSent'][word_idx]]))
        comparison_vec = libspacy.get_vector(' '.join([comparison['lemmaSent'][comparison_idx-2], comparison['lemmaSent'][comparison_idx-1], comparison['lemmaSent'][comparison_idx]]))
        difference = sqrt(sum(pow(a-b,2) for a, b in zip(word_vec, comparison_vec)))
      elif (word_idx < 1 or comparison_idx < 1):
        word_vec = libspacy.get_vector(' '.join([item['lemmaSent'][word_idx], item['lemmaSent'][word_idx+1], item['lemmaSent'][word_idx+2]]))
        comparison_vec = libspacy.get_vector(' '.join([comparison['lemmaSent'][comparison_idx], comparison['lemmaSent'][comparison_idx+1], comparison['lemmaSent'][comparison_idx+2]]))
        difference = sqrt(sum(pow(a-b,2) for a, b in zip(word_vec, comparison_vec)))
      else: 
        word_vec = libspacy.get_vector(' '.join([item['lemmaSent'][word_idx -1], item['lemmaSent'][word_idx], item['lemmaSent'][word_idx+1]]))
        comparison_vec = libspacy.get_vector(' '.join([comparison['lemmaSent'][comparison_idx-1], comparison['lemmaSent'][comparison_idx], comparison['lemmaSent'][comparison_idx+1]]))
        difference = sqrt(sum(pow(a-b,2) for a, b in zip(word_vec, comparison_vec)))
      if difference > 22.5 :
        nocb_filtered.append(item)
        item_ids_nocb.append(item['id'])
        item['ssp'] = word_vec
        break

  print('Filtering cb')
  for item in tqdm(cb) :
    sent_array = item['lemmaSent'].split()
    if len(sent_array) < 2: continue
    word_idx = sent_array.index('[n]')
    for comparison in cb:
      if item['postText'] == comparison['postText']: continue
      comparison_sent_array = comparison['lemmaSent'].split()
      if len(comparison_sent_array) < 2: continue
      comparison_idx = comparison_sent_array.index('[n]')
      if(word_idx > len(sent_array)-2 and comparison_idx < 1) or (word_idx < 1 and comparison_idx > len(comparison_sent_array) -2): continue
      if(word_idx > len(sent_array)-2 or comparison_idx > len(comparison_sent_array) -2):
        word_vec = libspacy.get_vector(' '.join([item['lemmaSent'][word_idx-2], item['lemmaSent'][word_idx-1], item['lemmaSent'][word_idx]]))
        comparison_vec = libspacy.get_vector(' '.join([comparison['lemmaSent'][comparison_idx-2], comparison['lemmaSent'][comparison_idx-1], comparison['lemmaSent'][comparison_idx]]))
        difference = sqrt(sum(pow(a-b,2) for a, b in zip(word_vec, comparison_vec)))
      elif (word_idx < 1 or comparison_idx < 1):
        word_vec = libspacy.get_vector(' '.join([item['lemmaSent'][word_idx], item['lemmaSent'][word_idx+1], item['lemmaSent'][word_idx+2]]))
        comparison_vec = libspacy.get_vector(' '.join([comparison['lemmaSent'][comparison_idx], comparison['lemmaSent'][comparison_idx+1], comparison['lemmaSent'][comparison_idx+2]]))
        difference = sqrt(sum(pow(a-b,2) for a, b in zip(word_vec, comparison_vec)))
      else: 
        word_vec = libspacy.get_vector(' '.join([item['lemmaSent'][word_idx -1], item['lemmaSent'][word_idx], item['lemmaSent'][word_idx+1]]))
        comparison_vec = libspacy.get_vector(' '.join([comparison['lemmaSent'][comparison_idx-1], comparison['lemmaSent'][comparison_idx], comparison['lemmaSent'][comparison_idx+1]]))
        difference = sqrt(sum(pow(a-b,2) for a, b in zip(word_vec, comparison_vec)))
      if difference > 22.5 :
        cb_filtered.append(item)
        item_ids_cb.append(item['id'])
        item['ssp'] = word_vec
        break
  

  print('Number of cb: ', len(cb_filtered), 'Number of nocb: ', len(nocb_filtered))
  cb=shuffle(cb_filtered, random_state=0)
  nocb=shuffle(nocb_filtered, random_state=0)
  nocb=nocb[:len(cb)]
  print ("CB=", len(cb), "NOCB=", len(nocb), "ITEM_IDS_CB", len(item_ids_cb), "ITEM_IDS_NOCB", len(item_ids_nocb))
  item_ids_nocb=item_ids_nocb[:len(cb)]
  y_cb = [0]*len(nocb)+[1]*len(cb)
  raw_cb = nocb + cb
  item_ids = item_ids_nocb + item_ids_cb
  (raw_cb, y_cb, item_ids) = shuffle(raw_cb, y_cb, item_ids, random_state=0)


  # (raw_cb, y_cb, item_ids) = shuffle(raw_cb, y_cb, item_ids, random_state=0)

  #create the dataset for subba
  fa=open('clickbait_titles.txt','w')
  fb=open('clickbait_ratings.txt','w')
  for (cb, rating) in zip(raw_cb, y_cb):
    fa.write(cb['postText']+'\n')
    fb.write(str(rating)+'\n')
  fa.close()
  fb.close()


  #(X, Y) = make_scatter(raw_cb, y_cb)
  train_percent=0.8
  train_size=int(len(raw_cb)*train_percent)
  X_raw_train = raw_cb[:train_size]
  y_train = y_cb[:train_size]
  train_ids = item_ids[:train_size]


  X_raw_test = raw_cb[train_size:]
  y_test = y_cb[train_size:]
  test_ids = item_ids[train_size:]

  #create test annotations
  fp=open("test_annotations.jsonl",'w')
  for item_id in test_ids:
    json_obj = raw_truths[item_id]
    json_str = json.dumps(json_obj)
    fp.write(json_str+'\n')
  fp.close()

  print ("X_raw_train, y_train", len(X_raw_train), len(y_train))
  print ("X_raw_test, y_test", len(X_raw_test), len(y_test) )
  X_train=[]
  X_test=[]

  print("Extracting features from train")
  for item in X_raw_train:
    # raw_title = item['postText']
    # vectors = libspacy.get_vector(raw_title)
    features = np.append(item['ssp'],[len(item['postWords'])])
    X_train.append(features)

  print( "Extracting features from test")
  for item in X_raw_test:
    # raw_title = item['postText']
    # vectors = libspacy.get_vector(raw_title)
    features = np.append(item['ssp'],[len(item['postWords'])])
    X_test.append(features)

  num_features = len(features)
  print( "Size of train, test", len(X_train), len(X_test))
  print( "Size of  labels train, test", len(y_train), len(y_test))
  print( "#features=", num_features)    

  print("Try linear regression")
  model = linear_model.LinearRegression()
  #model = svm.SVR(C=1.0, epsilon=0.2)
  model.fit(X_train, y_train)
  print("Mean squared error test: %.4f" % np.mean((model.predict(X_test) - y_test) ** 2))
  print("Mean squared error train: %.4f" % np.mean((model.predict(X_train) - y_train) ** 2))

  print("Minor improvements")
  y_pred = model.predict(X_test)
  y_pred = [ 0 if i < 0 else i for  i in y_pred]
  y_pred = [ 1 if i > 1 else i for  i in y_pred]
  y_pred = np.array(y_pred)
  print("Mean squared error test: %.4f" % np.mean((y_pred - y_test) ** 2))
  #print y_pred
  #y_pred = np.random.rand(len(X_test)) #Uncomment this line to check with random guesses
  create_predictions("test_predictions", y_pred, test_ids)
  create_predictions("test_truths", y_test, test_ids)
  #Print those instances where the prediction varies by a threshold
  
  fp=open('max_errors.txt','w')
  for (y_p, y_real, test_id) in zip(y_pred, y_test, test_ids):
    if abs(y_p - y_real) > 0.4:
      line ='%s %f %f' % (raw_data[test_id][feature], y_p, y_real)
      fp.write(line+'\n')
  fp.close()
  #print(model.coef_)
  #print(sorted(model.coef_.tolist()))
  os.system('python eval.py test_annotations.jsonl test_predictions outfile')
  print (model.coef_, model.intercept_)

#End of main


def create_raw_file(filename, data):
  fd = open(filename,'w')
  for row in data:
    fd.write(row+'\n')
  fd.close()

def rating_to_class(rating):
  if rating > 0.75:
    return 3
  if rating > 0.5:
    return 2
  if rating > 0.25:
    return 1
  return 0

def create_predictions(filename, predictions, item_ids):
  fd = open(filename, 'w')
  for (item_id, prediction) in zip(item_ids, predictions):
    json_obj={"id":item_id, "clickbaitScore":float(prediction)}
    json_str = json.dumps(json_obj)
    fd.write(json_str+'\n')

  fd.close()

def clean_title(title):
  title=title.replace("'", " ")
  title=title.replace("  ", " ")
  words = title.lower().split(' ')
  words = [ w for w in words if not w.startswith('@')]
  words = [ w for w in words if not w.startswith('#')]
  words = [ w for w in words if not w.startswith('rt')]
  #words = [ w for w in words if len(w) > 1 and not w[0].isdigit()]

  return ' '.join(words)

def clean_str(sentence):
  sentence = sentence.replace('\n',' ')
  return ''.join([c if ord(c) < 128 else ' ' for c in sentence])


if __name__ == "__main__":
  main()
