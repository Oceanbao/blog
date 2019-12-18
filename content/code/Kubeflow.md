---
title: "Kubeflow"
date: 2019-12-18T15:29:39+08:00
showDate: true
draft: false
---

# Kubeflow on GCP

[End to End Official Doc](https://www.kubeflow.org/docs/gke/gcp-e2e/)

![Flow](https://codelabs.developers.google.com/codelabs/kubeflow-introduction/img/dcc9c2ad993627f4.png)





`install gcloud, kubectl, docker`



## Example Project - MNIST DEPLOYMENT

`wget https://github.com/kubeflow/examples/archive/master.zip`

**Set ENV**

```bash
// available project ids can be listed with the following command:
// gcloud projects list
PROJECT_ID=<YOUR_CHOSEN_PROJECT_ID>

gcloud config set project $PROJECT_ID

ZONE=us-central1-a
DEPLOYMENT_NAME=mnist-deployment
cd ./mnist
WORKING_DIR=$(pwd)

# install kustomize from GitHub

# Enable GKE API
```



**GUI Set Up Kubeflow Cluster**

`deploy.kubeflow.cloud`

Check GCP Deployment 

https://console.cloud.google.com/dm



**Set up Kubectl**

```bash
gcloud container clusters get-credentials \
    $DEPLOYMENT_NAME --zone $ZONE --project $PROJECT_ID
    
kubectl config set-context $(kubectl config current-context) --namespace=kubeflow

kubectl get all
```



**Training**

- After `model.py` training completes, it will upload model to a path - i.e. create and use Cloud Storage bucket

```bash
// bucket name can be anything, but must be unique across all projects
BUCKET_NAME=${DEPLOYMENT_NAME}-${PROJECT_ID}

// create the GCS bucket
gsutil mb gs://${BUCKET_NAME}/

# Test new container image locally 
docker run -it $IMAGE_PATH

# Successful log then push to Google Container Registry
//allow docker to access our GCR registry
gcloud auth configure-docker --quiet

//push container to GCR
docker push $IMAGE_PATH
```



Train on Cluster

![Flow](https://codelabs.developers.google.com/codelabs/kubeflow-introduction/img/592dfcb8ad425470.png)



```bash
# Run training job 
cd $WORKING_DIR/training/GCS

# kustomize to config YAML manifests
kustomize edit add configmap mnist-map-training \
    --from-literal=name=my-train-1
    
# some default training params
kustomize edit add configmap mnist-map-training \
    --from-literal=trainSteps=200
kustomize edit add configmap mnist-map-training \
    --from-literal=batchSize=100
kustomize edit add configmap mnist-map-training \
    --from-literal=learningRate=0.01
    
# Config manifests to use custom bucket and training image
kustomize edit set image training-image=$IMAGE_PATH
kustomize edit add configmap mnist-map-training \
    --from-literal=modelDir=gs://${BUCKET_NAME}/my-model
kustomize edit add configmap mnist-map-training \
    --from-literal=exportDir=gs://${BUCKET_NAME}/my-model/export
    
# Beware of training code need permissions to R/W to storage bucket - kubeflow solves it by creating a service account within Project as part of deployment: verify
gcloud --project=$PROJECT_ID iam service-accounts list | grep $DEPLOYMENT_NAME

# This service should be auto-granted to R/W to storage bucket; kubeflow also added Kubernetes Secrets called 'user-gcp-sa' to cluster, containing credentials needed to authenticate as this service account within cluster:
kubectl describe secret user-gcp-sa

# Access storage bucket from inside training conatiner, set credential env to point to json file contained in secret
kustomize edit add configmap mnist-map-training \
    --from-literal=secretName=user-gcp-sa
kustomize edit add configmap mnist-map-training \
    --from-literal=secretMountPath=/var/secrets
kustomize edit add configmap mnist-map-training \
    --from-literal=GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/user-gcp-sa.json
    
# Kustomize to build new customized YAML files:
kustomize build . | kubectl apply -f -
# pipe to deploy to cluster

# Now a new tf-job on cluster called my-train-1-chief-0 
kubectl describe tfjob

# python log
kubectl logs -f my-train-1-chief-0

# once train done, query bucket's data
gsutil ls -r gs://${BUCKET_NAME}/my-model/export
```



> Note: The model is actually saving two outputs:
>
> 1) a set of checkpoints to resume training later if desired
>
> 2) A directory called export, which holds the model in a format that can be read by a TensorFlow Serving component



**Serving**

![Flow](https://codelabs.developers.google.com/codelabs/kubeflow-introduction/img/cc5d49c6f430d718.png)

```bash
cd $WORKING_DIR/serving/GCS

# TF Serving files in manifests, simply point the compoenent to bucket where model data is stored - will spin up a server to handle requests - unlike tf-job, no custom container needed for server process - instead all info server needs stored in the model file

# set name for service
kustomize edit add configmap mnist-map-serving \
    --from-literal=name=mnist-service

# point server at trained model in bucket
kustomize edit add configmap mnist-map-serving \
    --from-literal=modelBasePath=gs://${BUCKET_NAME}/my-model/export
    
# deploy
kustomize build . | kubectl apply -f -

# check
kubectl describe service mnist-service
```



**Deploying UI**

![Flow](https://codelabs.developers.google.com/codelabs/kubeflow-introduction/img/7b9b1f3166b67b4c.png)



So far:

- Trained model in bucket
- TF Server hosting it
- Deploy final piece of system: web interface (web-ui)

> a simple flask server hosting HTML/CSS/JavaScript files using mnist_client.py having following code through gRPC:

```python
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2 as psp

# create gRPC stub
channel = implementations.insecure_channel(server_host, server_port)
stub = psp.beta_create_PredictionService_stub(channel)

# build request
request = predict_pb2.PredictRequest()
request.model_spec.name = server_name
request.model_spec.signature_name = 'serving_default'
request.inputs['x'].CopyFrom(
    tf.contrib.util.make_tensor_proto(image, shape=image.shape))

# retrieve results
result = stub.Predict(request, timeout)
resultVal = result.outputs["classes"].int_val[0]
scores = result.outputs['predictions'].float_val
version = result.outputs["classes"].int_val[0]
```



```bash
# deploy web ui

cd $WORKING_DIR/front

# no customisation required, deploy directly
kustomize build . | kubectl apply -f -

# Service added to ClusterIP, meaning it cannot be accessed from outside the cluster!  Need to set up direct connection to the cluster
kubectl port-forward svc/web-ui 8080:80

# Cloud Shell 'Preview on port 8080'
```



> Web interface a simple HTML/JS wrapper around the TF Serving component doing actual predictions - 



**Clean Up**

```bash
gcloud deployment-manager deployments delete $DEPLOYMENT_NAME

gsutil rm -r gs://$BUCKET_NAME

gcloud container images delete us.gcr.io/$PROJECT_ID/kubeflow-train
gcloud container images delete us.gcr.io/$PROJECT_ID/kubeflow-web-ui
```

