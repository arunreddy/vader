import hashlib
import json
import os
from configparser import ConfigParser

import firebase_admin
import joblib
import numpy as np
from firebase_admin import db
from flask import Flask, send_from_directory, request, redirect, url_for, session
from flask import render_template
import pickle

from common_utils import load_meta_config
from jedi_dataset import JediDataset

from sklearn.datasets import fetch_20newsgroups
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

# load configuration
config = ConfigParser()
config.read('app.cfg')
dataset_config = config['CATS']


# load dataset
categories = ['comp.os.ms-windows.misc', 'sci.crypt']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

exp_10 = pickle.load( open(config['TXT']['EXP_PATH'], "rb" ) )
exp_10_pos = pickle.load( open(config['TXT']['EXP_PATH_POS'], "rb" ) )
exp_10_neg = pickle.load( open(config['TXT']['EXP_PATH_NEG'], "rb" ) )

# load metadata
df_meta = load_meta_config(config['DEFAULT']['META_PATH'], 'cat')

# load probability file.
with open(os.path.join(dataset_config['PRED_PROB_PATH'], 'prob_all.json'), 'r') as f:
  pred_prob_all = json.load(f)


# load the firebase application
try:
  fire_db = db.reference('cat/anelakur')
except:
  firebase_admin.initialize_app(options={
    'databaseURL': 'https://jedi-web.firebaseio.com/'
  })
  fire_db = db.reference('cat/anelakur')

# load the data set.
dataset = JediDataset('cat', features_path=dataset_config['DATA_PATH'])
file_names = dataset.train_file_names + dataset.test_file_names

# load influence influence scores.
inf_scores = joblib.load(dataset_config['INF_RESULTS_PATH'])
I = np.mean(np.abs(inf_scores), axis=1)


@app.route('/')
def home():
  if request.args.get('email'):
    email = request.args.get('email');
    md5 = hashlib.md5()
    md5.update(email.encode('utf-8'))
    hash = md5.hexdigest()
    session['email'] = email
    session['ehash'] = hash


    return redirect(url_for('start'))
  return render_template('home.html')


@app.route('/start', methods=['GET', 'POST'])
def start():
  if 'ehash' not in session:
    return redirect(url_for('home'))

  if request.method == 'POST':
    return redirect(url_for('screen_one'))

  labeled_images_ref = db.reference('cat/{}/labeled_images'.format(session['ehash']))
  labeled_images = labeled_images_ref.get()
  if labeled_images:
    session['is_new'] = 'n';
  else:
    session['is_new'] = 'y';
    _ref = db.reference('cat/{}/'.format(session['ehash']))
    _ref.set({'email': session['email']})

  return render_template('start.html')


@app.route('/i/s1', methods=['GET', 'POST'])
def screen_one():
  if 'ehash' not in session:
    return redirect(url_for('home'))

  if request.method == 'POST':
    curr_img_id = session['curr_img_id']
    class_label = request.form.get('class_label')
    _ref = db.reference('cat/{}/class_labels/'.format(session['ehash']))
    class_labels = _ref.get()
    if class_labels:
      class_labels[curr_img_id] = class_label
    else:
      class_labels = {curr_img_id: class_label}
    _ref.set(class_labels)

    return redirect(url_for('screen_two'))

  candidate_idx = np.argsort(I)

  # get the next image.
  curr_img_id = None

  labeled_images_ref = db.reference('cat/{}/labeled_images'.format(session['ehash']))
  labeled_images = labeled_images_ref.get()
  if not labeled_images:
    curr_img_id = candidate_idx[0]
    labeled_images_ref.set([str(curr_img_id)])
  else:

    for i in range(len(candidate_idx)):

      img_id = candidate_idx[i]
      if str(img_id) in labeled_images:
        continue

      curr_img_id = img_id
      break

    if session['is_new'] != 'y':
      curr_img_id = int(labeled_images[-1])
      session['is_new'] = 'n';
    else:
      labeled_images.append(str(curr_img_id))
      labeled_images_ref.set(labeled_images)

  curr_img_name = file_names[curr_img_id].split('.')[0]

  session['curr_img_id'] = str(curr_img_id)
  session['curr_img_name'] = curr_img_name

  txt = newsgroups_train.data[10]

  # /home/arun/Dropbox (ASU)/code_for_Arun/
  return render_template('txt/s1.html', txt=txt)


@app.route('/i/s2', methods=['GET', 'POST'])
def screen_two():
  if 'ehash' not in session:
    return redirect(url_for('home'))

  curr_img_id = session['curr_img_id']
  curr_img_name = session['curr_img_name']

  if request.method == 'POST':

    class_label = request.form.get('class_label')
    exp_label = request.form.get('exp_label')

    _ref = db.reference('cat/{}/class_exp_labels'.format(session['ehash']))
    class_exp_labels = _ref.get()
    if not class_exp_labels:
      class_exp_labels = {curr_img_id: {'class_label': class_label, 'exp_label': exp_label}}
    else:
      class_exp_labels[curr_img_id] = {'class_label': class_label, 'exp_label': exp_label}

    _ref.set(class_exp_labels)

    session['mask_img'] = '_{}{}'.format(class_label[0], exp_label[0])
    session['choosen_class'] = class_label

    # return redirect(url_for('screen_three'))

  # /home/arun/Dropbox (ASU)/code_for_Arun/
  return render_template('txt/s2.html',
                         tgt_img=curr_img_name,
                         exp_domestic=curr_img_name + '_domestic',
                         exp_wild=curr_img_name + '_wild',
                         iframe='http://localhost:5000/static/expl/expl_1.html')


@app.route('/i/s3', methods=['GET', 'POST'])
def screen_three():
  if 'ehash' not in session:
    return redirect(url_for('home'))

  curr_img_id = session['curr_img_id']
  curr_img_name = session['curr_img_name']
  mask_img = session['mask_img']
  choosen_class = session['choosen_class']

  if request.method == 'POST':
    decision_label = request.form.get('decision_label')

    _ref = db.reference('cat/{}/decision_labels'.format(session['ehash']))
    decision_labels = _ref.get()
    if decision_labels:
      decision_labels[curr_img_id] = decision_label
    else:
      decision_labels = {curr_img_id: decision_label}
    _ref.set(decision_labels)

    return redirect(url_for('screen_one'))

  # Add mask_image.
  return render_template('txt/s3.html', tgt_img=curr_img_name, choosen_class=choosen_class, iframe='http://localhost:5000/static/expl/expl_1.html')


@app.route('/img/<path:domain>/<path:img_type>/<path:filename>')
def img_static(domain, img_type, filename):
  root_dir = '/home/arun/Dropbox (ASU)/code_for_Arun'

  if img_type == 'orig':
    img_path_dir = dataset_config['IMG_PATH']
    filename_with_ext = filename + '.jpg'

  elif img_type == 'exp':
    img_path_dir = dataset_config['EXP_PATH']
    filename_with_ext = filename + '.png'

  return send_from_directory(directory=img_path_dir, filename=filename_with_ext)


if __name__ == '__main__':
  app.run()
