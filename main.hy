#! /usr/bin/env hy

(defmacro import-as [module alias]
  "Imports a module and binds it to an alias"
  `(do
     (import ~module)
     (setv ~alias ~module)))

(defn get-in [m ks]
  (setv k (first ks))
  (if (none? m)
      None
      (if (none? k)
          None
          (if (empty? (list (rest ks)))
              (get m k)
              (get-in (get m k) (list (rest ks)))))))

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

(defn run-examples [session examples]
  "examples [{'in' {'x' numpy} 'out' 'y'}]

  Returned as [{'in' {'x' list} 'out' {'y' list}}]"
  (list (map (fn [example]
               (setv data-in (get example "in"))
               (setv feed (dict (map (fn [k] [(.get_tensor_by_name session.graph (+ k "_4:0")) (get data-in k)]) data-in)))
               (setv example-cases (dict (map (fn [k] [k (.tolist (get data-in k))]) data-in)))
               (setv out (get example "out"))
               (setv node (.get_tensor_by_name session.graph (+ out "_5:0")))
               (setv results (.run session node :feed_dict feed))
               {"in"  example-cases
                "out" {out (.tolist results)}})
             examples)))

;; ======================================================================
;; Serialize the model

(import tempfile)

(import tensorflow.python.util.compat)

(setv builder (tensorflow.saved_model.builder.SavedModelBuilder target-dir))

(setv model (tf.saved_model.signature_def_utils.predict_signature_def :inputs {"x" x} :outputs {"y" y}))

(setv *default-target-dir* "data/models/")

(import json)

(defn save-model! [session model model-name examples owner target-dir]
  "Saves the model, returns the directory where it was saved"
  (setv builder (tensorflow.saved_model.builder.SavedModelBuilder (os.path.join target-dir model-name)))
  (.add_meta_graph_and_variables builder
                                 session
                                 [tf.saved_model.tag_constants.SERVING]
                                 :signature_def_map {model-name model})
  (.save builder model-name)
  (with [json-file (open (os.path.join target-dir model-name "custom-meta.json") "w")]
    (json.dump {"examples" (run-examples session examples)
                "owner" owner
                "name" model-name
                "model-type" "tensorflow"}
               json-file))
  (os.path.join target-dir model-name))

;; ======================================================================
;; Package the file

;; In this case, hashing guarantees uniqueness but not idempotency. tf.SaveModel will add the saved_count/# in the output files.
;; So, if you save the same model twice, it will work the same way but with different hashes. Oh well.

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

(defn upload-file! [bucket file-path file-hash]
  (setv k (boto.s3.key.Key bucket))

  (setv k.key file-hash)

  (.set_contents_from_filename k file-path))

;; ======================================================================
;; Publish

(defn publish! [session model model-name examples owner]
  "Saves the model to disk, hashes, zips it, and uploads it to S3"
  (setv saved-model-dir (save-model! :target-dir *default-target-dir* :session session :model model
                                     :model-name model-name :examples examples :owner owner))
  (setv sha (hash-dir saved-model-dir))
  (zipdir! saved-model-dir saved-model-dir)
  (setv s3 (.connect_s3 boto))
  (setv bucket (.get_bucket s3 *models-bucket*))
  (upload-file! bucket (+ saved-model-dir ".zip") sha))

(comment
  ;; setup bucket

  (setv s3 (.connect_s3 boto))

  (.create_bucket s3 *models-bucket*)

  (setv bucket (.get_bucket s3 *models-bucket*))

  (zipdir! "data/models/2" "data/models/2.zip")

  (upload-file! bucket "data/models/2.zip")

  ;; example API call

  (publish! :session session :model model :model-name "simple-linear"
            :examples {"cases" [{"in" 1 "out" 0} {"in" 2 "out" 1}]} :owner "sebastian"))
