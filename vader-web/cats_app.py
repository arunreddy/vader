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

from common_utils import load_meta_config
from jedi_dataset import JediDataset
from jedi_recommender import JEDI_blackbox

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

# load configuration
config = ConfigParser()
config.read('app.cfg')
dataset_config = config['CATS']

alpha = 1.
step = .2
beta = 0.833

n_cold_start = int(dataset_config['N_COLD_START'])
domain = 'cat'
# load metadata
df_meta = load_meta_config(config['DEFAULT']['META_PATH'], domain)

ADJ_PATH = config['CATS']['ADJ_PATH']
# load the data set.
dataset = JediDataset('cat', features_path=dataset_config['DATA_PATH'])

model = joblib.load(config['CATS']['MODEL_PATH'])
W = model.coefficients()

train_X = np.concatenate((np.ones((dataset.train_X.shape[0], 1)), dataset.train_X), axis=1)
test_X = np.concatenate((np.ones((dataset.test_X.shape[0], 1)), dataset.test_X), axis=1)
X = np.vstack([train_X, test_X])
Y = np.hstack([dataset.train_Y, model.predict(test_X)])
A = joblib.load(ADJ_PATH)

total = X.shape[0]

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


def assert_valid_session(curr_img_id):
  if not curr_img_id:
    return redirect(url_for('start'))

  if 'ehash' not in session:
    return redirect(url_for('home'))


@app.route('/')
def home():
  if request.args.get('email'):
    email = request.args.get('email');
    md5 = hashlib.md5()
    md5.update(email.encode('utf-8'))
    hash = md5.hexdigest()
    session['email'] = email
    session['ehash'] = email.replace('@', '_').replace('.', '_')
    return redirect(url_for('start'))
  return render_template('home.html', task='http://r.ddmcdn.com/s_f/o_1/cx_462/cy_245/cw_1349/ch_1349/w_720/APL/uploads/2015/06/caturday-shutterstock_149320799.jpg')


@app.route('/start', methods=['GET', 'POST'])
def start():
  if 'ehash' not in session:
    return redirect(url_for('home'))

  if request.method == 'POST':

    # check the database for data.
    labeled_images_ref = db.reference('cat/{}/labeled_images'.format(session['ehash']))
    labeled_images = labeled_images_ref.get()
    if not labeled_images:
      curr_img_id = np.random.randint(0,total)
      labeled_images_ref.set([str(curr_img_id)])
      ctr = 0
    else:
      curr_img_id = int(labeled_images[-1])
      ctr = len(labeled_images)

    return redirect(url_for('screen_one', curr_img_id=curr_img_id, ctr=ctr))

  labeled_images_ref = db.reference('cat/{}/labeled_images'.format(session['ehash']))
  labeled_images = labeled_images_ref.get()
  if labeled_images:
    session['is_new'] = 'n';
    session['counter'] = str(len(labeled_images))
  else:
    session['is_new'] = 'y';
    _ref = db.reference('cat/{}/'.format(session['ehash']))
    _ref.set({'email': session['email']})
    print('Set email -- {}'.format(session['email']))
    session['counter'] = '1'

  return render_template('start.html')


@app.route('/done')
def complete():
  return render_template('complete.html')

@app.route('/i/s1/<curr_img_id>/<ctr>', methods=['GET', 'POST'])
def screen_one(curr_img_id, ctr):
  assert_valid_session(curr_img_id)

  prev_img_id = curr_img_id

  if int(ctr) > X.shape[0]:
    return render_template('complete.html')

  if request.method == 'POST':
    class_label = request.form.get('class_label')
    _ref = db.reference('cat/{}/step1/'.format(session['ehash']))
    class_labels = _ref.get()
    if class_labels:
      class_labels[int(curr_img_id)] = class_label
    else:
      class_labels = {curr_img_id: class_label}
    _ref.set(class_labels)

    class_id = -1
    if class_label == 'domestic':
      class_id = 1

  
    return redirect(url_for('screen_two', curr_img_id=curr_img_id, ctr=ctr, lbl=class_label[0]))

  curr_img_name = file_names[int(curr_img_id)].split('.')[0]

  d_prob, w_prob = pred_prob_all[curr_img_name+'.jpg']

  if d_prob > w_prob:
    teacher_label = 'domestic'
  else:
    teacher_label = 'wild'
    
  return render_template('img/s1.html', image_name=curr_img_name, ctr=int(ctr)+1, total=total, teacher_label = teacher_label)


@app.route('/i/s2/<curr_img_id>/<ctr>/<lbl>', methods=['GET', 'POST'])
def screen_two(curr_img_id, ctr, lbl):
  assert_valid_session(curr_img_id)

  curr_img_name = file_names[int(curr_img_id)].split('.')[0]

  selected_label = 'wild'
  if lbl == 'd':
    selected_label = 'domestic'

  d_prob, w_prob = pred_prob_all[curr_img_name+'.jpg']

  if d_prob > w_prob:
    teacher_label = 'domestic'
  else:
    teacher_label = 'wild'

  if request.method == 'POST':

    class_label = request.form.get('class_label')

    class_id = -1
    if class_label == 'domestic':
      class_id = 1

    _ref = db.reference('cat/{}/step2'.format(session['ehash']))
    class_exp_labels = _ref.get()
    if not class_exp_labels:
      class_exp_labels = {curr_img_id: {'class_label': class_label}}
    else:
      class_exp_labels[int(curr_img_id)] = {'class_label': class_label}

    _ref.set(class_exp_labels)


    # add to labeled images.
    labeled_images_ref = db.reference('cat/{}/labeled_images'.format(session['ehash']))
    labeled_images = labeled_images_ref.get()

    yl_ref = db.reference('cat/{}/yl'.format(session['ehash']))
    yl_prob_ref = db.reference('cat/{}/yl_prob'.format(session['ehash']))

    yl = yl_ref.get()
    yl_prob = yl_prob_ref.get()
    if yl:
      labeled_images_int = [int(i) for i in labeled_images]

      labeled_images_int = labeled_images_int[:-1]

      prob, fvalue = JEDI_blackbox(X, Y, W, step, A, beta, labeled_images_int, yl_prob, yl)
      _yl_prob = prob[int(curr_img_id)]
      _yl_prob = np.abs(_yl_prob)
      _yl_prob = _yl_prob/np.sum(_yl_prob)

      # compute..
      yl.append(class_id)
      yl_ref.set(yl)

      yl_prob.append(_yl_prob.tolist())
      yl_prob_ref.set(yl_prob)

    else:
      yl_ref.set([class_id])
      yl_prob_ref.set([[0.5, 0.5]])

    if int(ctr) > n_cold_start:
      labeled_images_int = [int(img_id) for img_id in labeled_images]
      yl_ref = db.reference('cat/{}/yl'.format(session['ehash']))
      yl_prob_ref = db.reference('cat/{}/yl_prob'.format(session['ehash']))

      yl = yl_ref.get()
      yl_prob = yl_prob_ref.get()

      prob, fvalue = JEDI_blackbox(X, Y, W, step, A, beta, labeled_images_int, yl_prob, yl)

      P_JEDI = np.max(prob, axis=1)
      P_JEDI = P_JEDI / np.sum(P_JEDI)

      candidate_idx = np.argsort(P_JEDI)

      # pick the curr_image_id from the list of candidates.
      for img_id in candidate_idx:
        if str(img_id) not in labeled_images:
          curr_img_id = img_id
          break

    else:
      curr_img_id = np.random.randint(0,total)
      while str(curr_img_id) in labeled_images:
        curr_img_id = np.random.randint(0,total)

    labeled_images.append(str(curr_img_id))
    labeled_images_ref.set(labeled_images)

    ctr = int(ctr)+1
    return redirect(url_for('screen_one', curr_img_id=curr_img_id, ctr=ctr))

  return render_template('img/s2.html', tgt_img=curr_img_name,
                         domain=domain,
                         selected_label = teacher_label)


@app.route('/i/s3/<curr_img_id>/<ctr>/<mask>/<choosen_class>', methods=['GET', 'POST'])
def screen_three(curr_img_id, ctr, mask, choosen_class):
  assert_valid_session(curr_img_id)

  curr_img_name = file_names[int(curr_img_id)].split('.')[0]

  if request.method == 'POST':
    decision_label = request.form.get('decision_label')

    print(session)

    _ref = db.reference('cat/{}/step3'.format(session['ehash']))
    decision_labels = _ref.get()
    if decision_labels:
      decision_labels[int(curr_img_id)] = decision_label
    else:
      decision_labels = {curr_img_id: decision_label}
    _ref.set(decision_labels)

    # Increment the counter.
    ctr = int(ctr) + 1

    # check the database for data.
    labeled_images_ref = db.reference('cat/{}/labeled_images'.format(session['ehash']))
    labeled_images = labeled_images_ref.get()

    if curr_img_id not in labeled_images:
      labeled_images.append(str(curr_img_id))
      labeled_images_ref.set(labeled_images)

    # Move on to the next image.
    if int(ctr) > n_cold_start:
      print('JEDI and Influence only..')

      labeled_images_int = [int(img_id) for img_id in labeled_images]
      yl_ref = db.reference('cat/{}/yl'.format(session['ehash']))
      yl_prob_ref = db.reference('cat/{}/yl_prob'.format(session['ehash']))

      yl = yl_ref.get()
      yl_prob = yl_prob_ref.get()

      prob, fvalue = JEDI_blackbox(X, Y, W, step, A, beta, labeled_images_int, yl_prob, yl)

      P_JEDI = np.max(prob, axis=1)
      P_JEDI = P_JEDI / np.sum(P_JEDI)
      P_INF = I / I.sum()

      alpha = 1.
      P = P_INF + alpha * P_JEDI
      candidate_idx = np.argsort(P)

    else:
      print('Influence only..')
      candidate_idx = np.argsort(I)

    # pick the curr_image_id from the list of candidates.
    for img_id in candidate_idx:
      if str(img_id) not in labeled_images:
        curr_img_id = img_id
        break

    labeled_images.append(str(curr_img_id))
    labeled_images_ref.set(labeled_images)

    return redirect(url_for('screen_one', curr_img_id=curr_img_id, ctr=ctr))

  # Add mask_image.
  return render_template('img/s3.html', tgt_img=curr_img_name, mask_img='{}_masked_{}'.format(curr_img_name, mask),
                         choosen_class=choosen_class)


@app.route('/img/<path:domain>/<path:img_type>/<path:filename>')
def img_static(domain, img_type, filename):
  root_dir = '/home/arun/Dropbox (ASU)/code_for_Arun'

  if img_type == 'orig':
    img_path_dir = dataset_config['IMG_PATH']
    filename_with_ext = filename + '.jpg'

  elif img_type == 'exp':
    img_path_dir = dataset_config['EXP_PATH']
    filename_with_ext = filename + '.png'

  elif img_type == 'mask':
    img_path_dir = dataset_config['MASK_PATH']
    filename_with_ext = filename + '.png'

  return send_from_directory(directory=img_path_dir, filename=filename_with_ext)


if __name__ == '__main__':
  app.run()
