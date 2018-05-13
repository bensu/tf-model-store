#! /usr/bin/env hy

(defmacro import-as [module alias]
  "Imports a module and binds it to an alias"
  `(do
     (import ~module)
     (setv ~alias ~module)))

(import-as tensorflow tf)

(import-as numpy np)

;; ======================================================================
;; Define a simple model and train it

(defn fn-to-simulate [x]
  "The function we are trying to simulate"
  (if (< x 50)
      0
      1))

;; simplest dataset

(setv number-data-points 100)

(setv features (.astype (np.transpose (np.array (range number-data-points))) np.float32))

(setv features.shape (tuple [number-data-points 1]))

(setv labels (.astype (np.transpose (np.array (list (map fn-to-simulate (range number-data-points))))) np.float32))

(setv labels.shape (tuple [number-data-points 1]))

(assert (= features.shape labels.shape))

;; I expect a model of the form y = w * x + b, with w = 1 and b = 1

(setv x (tf.placeholder tf.float32 [None 1] :name "x"))

(setv W (tf.Variable (tf.zeros [1 1])))

(setv b (tf.Variable (tf.zeros [1])))

(setv y (tf.nn.softmax (+ b (tf.matmul x W)) :name "y"))

(setv yy (tf.placeholder tf.float32 [None 1] :name "yy"))

(setv cross-entropy (tf.reduce_mean (- (tf.reduce_sum (* yy (tf.log y)) :reduction_indices [1]))))

(setv train-step (.minimize (tf.train.GradientDescentOptimizer 0.5) cross-entropy))

(setv session (tf.InteractiveSession))

(.run (tf.global_variables_initializer))

(.run session train-step :feed_dict {x features
                                     yy labels})

(setv correct-prediction (tf.equal (tf.argmax y 1) (tf.argmax yy 1)))

(setv accuracy (tf.reduce_mean (tf.cast correct-prediction tf.float32)))

(print (.run session accuracy :feed_dict {x features yy labels}))

(.run session y :feed_dict {x features})

;; ======================================================================
;; Serialize the model

(import tempfile)

(import tensorflow.python.util.compat)

(setv target-dir "data/models/1")

(setv builder (tensorflow.saved_model.builder.SavedModelBuilder target-dir))

(setv model (tf.saved_model.signature_def_utils.predict_signature_def :inputs {"x" x} :outputs {"y" y}))

(.add_meta_graph_and_variables builder
                               session
                               [tf.saved_model.tag_constants.SERVING]
                               :signature_def_map {"magic_model" model})

(.save builder)

;; ======================================================================
;; Upload the model to S3

(import os)

(import shutil)

(defn zipdir! [source target]
  (shutil.make_archive target "zip" source))

(import hashlib)

(defn hash-dir [dir-path]
  "Returns a hex string of the sha1 digest of the contents of the directory"
  (setv *buf-size* 65536)
  (setv sha1 (hashlib.sha1))
  (for [rs (os.walk dir-path)]
    (setv root (first rs))
    (setv files (last rs))
    (for [file files]
      (with [f (open (os.path.join root file) "rb")]
        (while True
          (setv data (.read f *buf-size*))
          (if (not data)
              (break))
          (.update sha1 data)))))
  (.upper (.hexdigest sha1)))

;; ======================================================================
;; Upload S3

(import boto)

(import math)

(setv s3 (.connect_s3 boto))

(comment
  (.create_bucket s3 *models-bucket*))

(comment
  (setv bucket (.get_bucket s3 *models-bucket*)))

(setv file-path "data/models/1.zip")

(defn upload-file! [bucket file-path]
  (setv file-hash (hash-dir file-path))
  (setv k (boto.s3.key.Key bucket))
  (setv k.key file-hash)
  (.set_contents_from_filename k file-path))
