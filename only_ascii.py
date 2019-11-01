

def clean_str(sentence):
  sentence = sentence.replace('\n',' ')
  return ''.join([c if ord(c) < 128 else ' ' for c in sentence])
