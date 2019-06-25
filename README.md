## Overview
dong platform [http://dong.libgirl.com](http://dong.libgirl.com) consists of

1. **dong** - a [MLOps](https://en.wikipedia.org/wiki/MLOps) project framework.
2. **dong cloud** - a [MLOps](https://en.wikipedia.org/wiki/MLOps) cloud service for managing cloud training/deployment of a **dong** project. 
3. **dong cli** - a client side command line tool to quickly build or execute a **dong** project on local test or on **dong cloud**.

In this quickstart step-by-step guide we'll use dong to train and deploy a model using [mnist](https://en.wikipedia.org/wiki/MNIST_database) dataset.

## Installation
Install client side dong cli
```bash
$ pip install dong
```
## Signup
Please [apply via the form](https://dong.libgirl.com/#Getfree)  to get a trial account.

## Login
```sh
$ dong login
```
## The example mnist dong project
Get the [example mnist dong project](https://github.com/libgirlenterprise/dong_mnist_example)
```bash
$ git clone https://github.com/libgirlenterprise/dong_mnist_example.git
```
```bash
$ cd dong_mnist_example
```

## Train/Deploy example
### Local test

Not yet released.

### Train on dong's cloud
```bash
$ ls # make sure we're under the right directory
MANIFEST.in	README.md	setup.py	dong_mnist_example
```
```bash
$ dong train exec -m "my first dong train exec" -- --config-module default

Use dong ML project: dong_mnist_example
Project path: /private/tmp/dong_mnist_example
Building package...
Uploading package...
Job name: [TRAIN_JOB_NAME]
```

It will take about 3 mins. You can do status check by

```bash
$ dong train status -j TRAIN_JOB_NAME
name: TRAIN_JOB_NAME
message: my first dong train exec
status: Running
```

### Deploy the model to an API endpoint
Check if the train succeeds or not
```bash
$ dong train status -j TRAIN_JOB_NAME
name: TRAIN_JOB_NAME
message: my first dong train exec
status: Succeeded
```
#### Deploy
```bash
$ dong endpoint up TRAIN_JOB_NAME
Bring up...
New endpoint name: ENDPOINT_NAME
```
#### Endpoint status check
```bash
$ dong endpoint status -e ENDPOINT_NAME
Endpoint name: ENDPOINT_NAME
External ip: SOME_IP
Status: Preparing
```
### Test the endpoint
Check if the endpoint is ready running or not.
```bash
$ dong endpoint status -e ENDPOINT_NAME
Endpoint name: ENDPOINT_NAME
External ip: SOME_IP
Status: Running
```
Then we can test 

#### http://SOME_IP/api/v1/hello
```bash
$ curl -X POST \  
  http://SOME_IP/api/v1/hello \ 
  -H 'Content-Type: application/json' \
  -d '""'
  
"hello"
```
#### http://SOME_IP/api/v1/normal
```bash
$ curl -X POST \
  http://SOME_IP/api/v1/normal \
  -H 'Content-Type: application/json' \
  -H 'cache-control: no-cache' \
  -d '[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 224, 0, 0, 0, 0, 0, 0, 0, 70, 29, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 231, 0, 0, 0, 0, 0, 0, 0, 148, 168, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 195, 231, 0, 0, 0, 0, 0, 0, 0, 96, 210, 11, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 252, 134, 0, 0, 0, 0, 0, 0, 0, 114, 252, 21, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 45, 236, 217, 12, 0, 0, 0, 0, 0, 0, 0, 192, 252, 21, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 168, 247, 53, 0, 0, 0, 0, 0, 0, 0, 18, 255, 253, 21, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 84, 242, 211, 0, 0, 0, 0, 0, 0, 0, 0, 141, 253, 189, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 169, 252, 106, 0, 0, 0, 0, 0, 0, 0, 32, 232, 250, 66, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 15, 225, 252, 0, 0, 0, 0, 0, 0, 0, 0, 134, 252, 211, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 22, 252, 164, 0, 0, 0, 0, 0, 0, 0, 0, 169, 252, 167, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 9, 204, 209, 18, 0, 0, 0, 0, 0, 0, 22, 253, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 169, 252, 199, 85, 85, 85, 85, 129, 164, 195, 252, 252, 106, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 41, 170, 245, 252, 252, 252, 252, 232, 231, 251, 252, 252, 9, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 84, 84, 84, 84, 0, 0, 161, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 252, 252, 45, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 252, 244, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 232, 236, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 179, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 254, 107, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 227, 254, 254, 9, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 254, 254, 165, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 203, 254, 254, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 53, 254, 254, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 254, 254, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 196, 254, 248, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 254, 254, 237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 254, 254, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 254, 238, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 252, 254, 223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 254, 254, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 254, 238, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 252, 254, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 254, 254, 131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 254, 234, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 254, 204, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 211, 254, 196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 158, 254, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 157, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]'
```
The reply should look something like this
```
[[2.879309568548649e-10, 2.6625946239478004e-11, 2.580107016925126e-09, 5.453760706930488e-12, 0.9999690055847168, 3.3259575649147166e-10, 3.2778924019538636e-10, 4.35676184906697e-07, 2.190379821964683e-10, 3.0488341508316807e-05], [1.3478038130010361e-10, 0.9997259974479675, 6.728584622806011e-08, 5.9139901864568856e-09, 9.023875122693426e-07, 5.708465922182882e-10, 1.2523435088951373e-07, 0.0002721738419495523, 6.667413003924594e-07, 9.076808638042166e-09]]
```
