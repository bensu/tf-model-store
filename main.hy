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
               (setv feed (dict (map (fn [k] [(.get_tensor_by_name session.graph (+ k ":0")) (get data-in k)]) data-in)))
               (setv example-cases (dict (map (fn [k] [k (.tolist (get data-in k))]) data-in)))
               (setv out (get example "out"))
               (setv node (.get_tensor_by_name session.graph (+ out ":0")))
               (setv results (.run session node :feed_dict feed))
               {"in"  example-cases
                "out" {out (.tolist results)}})
             examples)))

;; ======================================================================
;; Serialize the model

(import tempfile)

(import tensorflow.python.util.compat)

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
;; Package and Publish

(import os)

(import shutil)

(import boto3)

(import namesgenerator)
(import datetime)

(defn zipdir! [source target]
  (shutil.make_archive target "zip" source))

(defn quasi-unique-name []
  "The namesgenerator has 15k which is not unique enough for my taste. Increasing to 1M by prepending a second adjective"
  (setv unique-name (namesgenerator.get_random_name "-"))
  (+ (first (.split(namesgenerator.get_random_name "-") "-")) "-" unique-name))

(setv *models-bucket* "the-bucket-you-are-using")

(defn update-registry! [s3-client model-description]
  "registry.json contains a list with all the models ever uploaded"
  (setv registry-object (s3-client.get_object :Bucket *models-bucket* :Key "registry.json"))
  (setv registry (json.loads (.decode (.read (get registry-object "Body")) "utf-8")))
  (assoc registry "updated-ts" (.isoformat (datetime.datetime.now)))
  (setv registered-models (get registry "models"))
  (.append registered-models model-description)
  (s3-client.put_object :Bucket *models-bucket* :Key "registry.json" :Body (.encode (json.dumps registry))))

(defn publish! [session model model-name examples owner]
  "Saves the model to disk, names it, zips it, and uploads it to S3. Returns the generated name for the model"
  ;; save
  (setv saved-model-dir (save-model! :target-dir *default-target-dir* :session session :model model
                                     :model-name model-name :examples examples :owner owner))
  ;; name
  (setv now-str (.isoformat (datetime.datetime.now)))
  (setv model-key (quasi-unique-name))
  ;; zip
  (zipdir! saved-model-dir saved-model-dir)
  ;; upload
  (setv s3 (boto3.resource "s3"))
  (setv bucket (s3.Bucket *models-bucket*))
  (setv model-description {"owner" owner
                           "model-key" model-key
                           "model-name" model-name
                           "model-type" "tensorflow"
                           "created-ts" now-str})
  (bucket.upload_file (+ saved-model-dir ".zip") model-key :ExtraArgs {"Metadata" model-description})
  (setv s3-client (boto3.client "s3"))
  (s3-client.get_object :Bucket *models-bucket* :Key "registry.json")
  ;; update registry
  (update-registry! s3-client model-description)
  (comment
    ;; tag s3 bucket
    (s3-client.put_object_tagging :Bucket *models-bucket* :Key model-key
                                  :Tagging {"TagSet" [{"Key" "owner" "Value" owner}
                                                      {"Key" "created-ts" "Value" now-str}
                                                      {"Key" "model-key" "Value" model-key}
                                                      {"Key" "model-name" "Value" model-name}
                                                      {"Key" "model-type" "Value" "tensorflow"}]}))
  model-key)

(print "You are publishing:")

(print
  (publish! :session session :model model :model-name "simple-linear"
            :examples [{"in" {"x" features} "out" "y"}] :owner "sebastian"))


(comment
  ;; setup bucket

  (setv s3 (.connect_s3 boto))

  (.create_bucket s3 *models-bucket*)

  (setv bucket (.get_bucket s3 *models-bucket*))

  (zipdir! "data/models/2" "data/models/2.zip")

  ;; example API call

  (publish! :session session :model model :model-name "simple-linear"
            :examples {"cases" [{"in" 1 "out" 0} {"in" 2 "out" 1}]} :owner "sebastian"))
