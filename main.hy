#! /usr/bin/env hy

(defmacro import-as [module alias]
  "Imports a module and binds it to an alias"
  `(do
     (import ~module)
     (setv ~alias ~module)))

(import-as tensorflow tf)

(import numpy)

;; ======================================================================
;; Define a simple model and train it

(import tensorflow.examples.tutorials.mnist)

(setv minst (tensorflow.examples.tutorials.mnist.input_data.read_data_sets "data/MNIST_data/" :one_hot True))

(setv x (tf.placeholder tf.float32 [None 784]))

(setv W (tf.Variable (tf.zeros [784 10])))

(setv b (tf.Variable (tf.zeros [10])))

(setv y (tf.nn.softmax (+ b (tf.matmul x W))))

(setv yy (tf.placeholder tf.float32 [None 10]))

(setv cross-entropy (tf.reduce_mean (- (tf.reduce_sum (* yy (tf.log y)) :reduction_indices [1]))))

(setv train-step (.minimize (tf.train.GradientDescentOptimizer 0.5) cross-entropy))

(setv session (tf.InteractiveSession))

(.run (tf.global_variables_initializer))

(for [i (range 1000)]
  (setv r (minst.train.next_batch 100))
  (setv batch_xs (first r))
  (setv batch_ys (second r))
  (.run session train_step :feed_dict {x batch_xs yy batch_ys}))

(setv correct-prediction (tf.equal (tf.argmax y 1) (tf.argmax yy 1)))

(setv accuracy (tf.reduce_mean (tf.cast correct-prediction tf.float32)))

(print (.run session accuracy :feed_dict {x minst.test.images yy minst.test.labels}))

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

(import zipfile)

(defn zipdir! [source target]
  "Adds all files in the `source` directory into the `target` zip file"
  (with [ziph (zipfile.ZipFile target "w" zipfile.ZIP_DEFLATED)]
    (for [rs (os.walk source)]
      (setv root (first rs))
      (setv files (last rs))
      (for [file files]
        (.write ziph (os.path.join root file))))))

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
