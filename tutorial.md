# dong tutorial
## Overview
In this tutorial, we'll step-by-step make a simple mnist **dong** MLOps project from coding, training, to deployment. We'll use tensorflow as the machine learning library.

## Installation
Install client side dong cli
```bash
$ pip install dong
```
## Signup
Please [apply via the form](https://dong.libgirl.com/signup/)  to get a trial account.

## Login
```sh
$ dong login
```

## Project creation

Run ```dong new``` with a project name to create a new dong project. For example,
```bash
$ dong new my_dong_mnist
```
or create the project directory first then run  ```dong init```, e.g,
```bash
some_path:some_id $ mkdir my_dong_mnist
some_path:some_id $ cd my_dong_mnist
my_dong_mnist:some_id $ dong init
```

then let's go into the directory ```my_dong_mnist```.
```bash
$ cd my_dong_mnist
```
Here is the project structure
```
my_dong_mnist
├── my_dong_mnist
│   ├── __init__.py
│   ├── config
│   │   ├── __init__.py
│   │   └── default.py
│   ├── data
│   │   ├── __init__.py
│   │   └── default.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── default.py
│   │   ├── init
│   │   │   ├── __init__.py
│   │   │   └── default.py
│   │   ├── serializer
│   │   │   ├── __init__.py
│   │   │   └── default.py
│   │   └── train
│   │       ├── __init__.py
│   │       └── default.py
│   ├── service
│   │   ├── __init__.py
│   │   └── default.py
│   └── tune
│       ├── __init__.py
│       └── default.py
└── setup.py
```
## Data preparation module
### Module file location
Open ```my_dong_mnist/data/default.py```, and you can see the template:

```python
import dong.framework

class DefaultData(dong.framework.Data):
    
    def __init__(self, config=None):
        pass
    
    def get_train_data(self):
        return None

    def get_eval_data(self):
        return None

    def get_data_params(self):
        return None
```
### Data object constructor
For a basic tensorflow mnist project, here is the code for ```__init__()``` 
```python
def __init__(self, config=None):
                                
    mnist = tensorflow.keras.datasets.mnist
    (x_train, self._y_train), (x_test, self._y_test) = mnist.load_data()

    self._x_train, self._x_test = x_train / 255.0, x_test / 255.0
```
### collections.namedtuple
To make the data more semantical accessible, in this tutorial we use [collections.nametuple](https://docs.python.org/3/library/collections.html#collections.namedtuple).
```python
DataPair = collections.namedtuple('DataPair', ['x', 'y'])
DataParams = collections.namedtuple('Params', ['image_size', 'num_labels'])
```
So we can use the following to access structural information
```python
train_data.x
train_data.y
data_params.image_size
data_params.num_labels
```

### get\_train\_data(), get\_eval\_data()
When **dong** executes a training, it will invoke 

- ```get_train_date()``` to get the training data  
- ```get_eval_data()``` to get the training evaluation data 

With the help of [collections.namedtuple](#collectionsnamedtuple), here is the code for the these two functions.
```python
def get_train_data(self):
    return DataPair(self._x_train, self._y_train)

def get_eval_data(self):
    return DataPair(self._x_test, self._y_test)
```

### get\_data\_params()
Sometimes we want to know some data information to define our learning model and train mechanism. For example, what the input data dimension is.```get_data_params()``` serves this purpose.

For a basic tensorflow mnist project, it means
```python
def get_data_params(self):
    
    image_size = self._x_train.shape[1]
    num_labels = len(numpy.unique(self._y_train))
    return DataParams(image_size, num_labels)
```
see [collections.namedtuple](#collectionsnamedtuple) for DataParams()

### Final code of data preparation module

Filename: ```my_dong_mnist/data/default.py```
```python

from __future__ import division

import collections
import numpy
import tensorflow
import dong.framework

DataPair = collections.namedtuple('DataPair', ['x', 'y'])
DataParams = collections.namedtuple('Params', ['image_size', 'num_labels'])

class DefaultData(dong.framework.Data):

    
    def __init__(self, config=None):
                                
        mnist = tensorflow.keras.datasets.mnist
        (x_train, self._y_train), (x_test, self._y_test) = mnist.load_data()

        self._x_train, self._x_test = x_train / 255.0, x_test / 255.0

    def get_train_data(self):
        return DataPair(self._x_train, self._y_train)

    def get_eval_data(self):
        return DataPair(self._x_test, self._y_test)

    def get_data_params(self):
    
        image_size = self._x_train.shape[1]
        num_labels = len(numpy.unique(self._y_train))
        return DataParams(image_size, num_labels)
```

## Model - model
### Module file location
Open ```my_dong_mnist/model/default.py```, and you can see the template:
```python

import dong.framework

class DefaultModel(dong.framework.Model):
    pass
```
### Model class
To do a train, there are three methods to implement.

- ```__init__(self, config, data_params, save_dir)``` for model construction
- ```train(self, data, config)``` to do model training
- ```write(self, save_dir)``` for model serialization

When developioing an ML project, you may need to use many models, data sets, training methods, or hyperparameters in different phases. By creating modules for each functionality and let `DefaultModel` inherit the 3 methods from those that fit your need at the momoent, you can easily use, reuse, and compose those functionalities.

You can also directly implement the 3 methods in ```my_dong_mnist/model/default.py```. **We don't force over modularization**. As a reference for organizing a larger project, here we separate them into different modules. See more in **Modular** and **Customizable** of [dong Framework Features](https://pypi.org/project/dong/).


### Final code of model module

File name: ```my_dong_mnist/model/default.py``` 
```python
from .init.default import DefaultModelInit
import dong.framework

class DefaultModel(DefaultModelInit, dong.framework.Model):

    from .train.default import train
    from .serializer.default import write
```
- ```DefaultModelInit``` implements ```__init__(self, config, data_params, save_dir)``` 
- In ```.train.default``` we implements ```train(self, data, config)``` 
- In ```.serializer.default``` we implements```write(self, save_dir)``` 

Now Let's implement them.

## Model - model init module
This module should implement how one model is created or reconstructed.
### Module file location 
Open ```my_dong_mnist/model/init/default.py```, and you can see the template:
```python
class DefaultModelInit():

    def __init__(self, config={}, data_params=None, save_dir=None):
        pass
```

### Neural network structure
This is the neural network structure we'd like to define and init the model.
```python
[
    Flatten(input_shape=(image_size, image_size)),
    Dense(hidden_units, activation=relu),
    Dropout(dropout),
    Dense(num_labels, activation=softmax)
]
```
### Inherit a Keras model
In this tutorial we use the high level [Keras API](https://keras.io) to define the neural network structure.

Even though we can create a model from Keras' API and assign it to an instance variable of DefaultModelInit, to apply inheritence makes the system more simple. 
```python
class DefaultModelInit(tensorflow.keras.models.Sequential):
```
### Final code of model init module

Then we can have a **tensorflow** multilevel perceptron model.

Filename: ```my_dong_mnist/models/init/default.py```
```python
import tensorflow


class DefaultModelInit(tensorflow.keras.models.Sequential):

    _hidden_units = 512
    _dropout = 0.2

    def __init__(self, config={}, data_params=None, save_dir=None):
        super().__init__([
            tensorflow.keras.layers.Flatten(
                input_shape=(data_params.image_size, data_params.image_size)),
            tensorflow.keras.layers.Dense(self._hidden_units,
                                          activation=tensorflow.nn.relu),
            tensorflow.keras.layers.Dropout(self._dropout),
            tensorflow.keras.layers.Dense(data_params.num_labels,
                                          activation=tensorflow.nn.softmax)
        ])

```
## Model - model train module
### Module file location
Open ```my_dong_mnist/model/train/default.py```, and you can see the template:
```python
def train(self, data, config=None):
    
    return 0.
```

### Parameters of train()
#### self
We'll import ```train()``` function into the final model class. So, the ```self``` means the ```self``` of the model instance. 

#### data
**dong** will instantiate a Data object of the [data class](#Data-preparation-module) and pass it into the train function

#### config
This is for configuration and code separation. It helps us do hyperparameter tuning.

We don't use ```config``` in this tutorial.

### Final code of model train module
Since we already inherent [Keras's Sequential model](https://keras.io/models/sequential/), we can invoke ```self.compile()```, ```self.fit()```, and ```self.evaluate()```.

Filename: ```my_dong_mnist/model/train/default.py```
```python
def train(self, data, config=None):

    self.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    self.fit(data.get_train_data().x, data.get_train_data().y, epochs=3)

    return self.evaluate(data.get_eval_data().x, data.get_eval_data().y)[1]
```

## Model - model serializer module
### Module file location
Open ```/my_dong_mnist/model/serializer/default.py```, and we can see the template:
```python
def write(self, save_dir):
    pass
    
def read(self, save_dir):
    pass
```

### write(self, save_dir)
This is the method to implement training model serialization. 

#### self
We'll import ```write()``` function into the final model class. So, the ```self``` means the ```self``` of the model instance. 

#### save_dir

`save_dir` is the directory path for you to save the trained model output.

We'll use TensorFlow Keras [tensorflow.keras.Model.save_weights](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights) to output the hdf5 file. And here is the code,

```python
def write(self, save_dir):
    export_path = save_dir + 'my_model_weights.hdf5'
    self.save_weights(export_path)
```
### read(self, save_dir)
To restore the model for model deployment, we have a ```save_dir``` which is the previous output location.

We'll use [tensorflow.keras.Model.load_weights](https://www.tensorflow.org/api_docs/python/tf/keras/Model#load_weights) to restore the model.

```python
def read(self, save_dir):
    export_path = save_dir + 'my_model_weights.hdf5'
    self.load_weights(_export_path)
```

### Final code of model serializer module

Filename:```/my_dong_mnist/model/serializer/default.py```
```python
def write(self, save_dir):
    export_path = save_dir + 'my_model_weights.hdf5'
    self.save_weights(export_path)
    
def read(self, save_dir):
    export_path = save_dir + 'my_model_weights.hdf5'
    self.load_weights(export_path)
```

## Model - model init by deserialization
### Use case

We can init a model by loading it from a trained model to
1. Deploy it to a service function.
2. Futher train a pre-trained model. **Notice**: **dong** alpha version haven't supported such functionality.

### File creation

run at the root of your ML project:
```bash
$ dong template --model-init-module default_load
```
And open the generated ```/my_dong_mnist/model/init/default_load.py```, we can see it as follows:

```python
class DefaultloadModelInit():
    def __init__(self, config={}, data_params=None, save_dir=None):
        pass
```

### Final code of model init module by deserialization

With inheriting [DefaultModelInit](#Model---model-init-module) and [import read function](#readself-save_dir), we can have the following model initialization by deserialization

Filename: ```/my_dong_mnist/model/init/default_load.py```
```python
from .default import DefaultModelInit
from ...data.default import DefaultData


class DefaultloadModelInit(DefaultModelInit):

    from ..serializer.default import read

    def __init__(self, config={}, data_params=None, save_dir=None):

        data = DefaultData()
        super().__init__(config,
                         data_params=data.get_data_params(),
                         save_dir=save_dir)
        self.read(save_dir)
```

## Service - service
### Module file location
Open ```/my_dong_mnist/service/default.py```, and you can see the template:
```python
import json
import dong.framework

class DefaultService(dong.framework.Service):

    @dong.framework.request_handler
    def hello(self, request_body, mime_type='application/json'):

        return json.dumps('hello')
```

### Service class

We have to define a Service class with service request handler. Like the ```hello``` request handler.
```python
    @dong.framework.request_handler
    def hello(self, request_body, mime_type='application/json'):

        return json.dumps('hello')
```
#### @dong.framework.request_handler 
Use the [python decorator](https://wiki.python.org/moin/PythonDecorators), ```@dong.framework.request_handler```, to declare methods as dong service request handlers. A request comes with ```request_body``` and its ```mime_type```. 

#### Generality of dong Service class
As you see, Service class is a general Service structure that

1. you're not forced to utilize machine learning models. That means you can declare multiple handlers for various purposes.
2. it doesn't bind to any communication protocol. Now on **dong cloud** you can deploy it as a http service endpoint, while in the future **dong** will provide more tools to integrate your model into different application interfaces such as edge computation.
3. Again, by letting `DefaultService` inherit the desired functionalities from the prepared modules, you can make it as what you like. Here we load a trained model for the Service class as an example.

### Load a trained model in Service class

You can write the model deserialization directly in the constructor of the Service class. Still, in this tutorial we'll write the deserialization code in [another module file](#Model---model-init-by-deserialization) and use inheritance to compose it into the Service class.

```python
from ..model.init.defaultload import DefaultloadModelInit


class DefaultService(DefaultloadModelInit):
```
### Implement request handler serve()

Let's write code to return a json string as a reply to the request.

#### Load the json
```python
    data = json.loads(request_body_json)
```
#### Scale the input
We expect the request as an array of 28*28 of 0 to 255 input. let's scale it between 0.0 and 1.0
```python
    x = numpy.array(data) / 255.0
```

#### Predict and return
We can directly use [model.predict](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model) to predict the mnist data output. We also dump it into a json string.
```python
    return json.dumps(self.predict(x).tolist())
```
### Final code of service module

Filename: ```/my_dong_mnist/service/default.py```
```python
from __future__ import division
import json
import numpy
import dong.framework
from dong_mnist_example.model.init.default_load import DefaultloadModelInit


class DefaultService(DefaultloadModelInit, dong.framework.Service):

    @dong.framework.request_handler
    def serve(self, request_body, mime_type='application/json'):

        data = json.loads(request_body)
        x = numpy.array(data) / 255.0
        return json.dumps(self.predict(x).tolist())

    @dong.framework.request_handler
    def hello(self, request_body, mime_type='application/json'):

        return json.dumps('hello')
```

## Add package dependencies
### File location
Open ```setup.py```
```python
from setuptools import setup, find_packages

setup(name='my_dong_mnist',
      version='0.1',
      description='none',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,  
      zip_safe=False,
      entry_points = {
        'console_scripts': ['my_dong_mnist=my_dong_mnist:main'],
      },
    )
```
### Add dependencies
```python
      install_requires=[
          'tensorflow',
      ],
```

### Final code of setup file

Filename: ```setup.py```
```python
from setuptools import setup, find_packages

setup(name='my_dong_mnist',
      version='0.1',
      description='none',
      license='MIT',
      install_requires=[
          'tensorflow',
      ],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      entry_points = {
        'console_scripts': ['my_dong_mnist=my_dong_mnist:main'],
      },
    )
```
## Execute my\_dong\_mnist
### Train
```bash
$ ls # make sure we're under the write directory
my_dong_mnist	setup.py
```
```bash
$ dong train exec
Training message: my dong mnist
Use dong ML project: my_dong_mnist
Project path: /path-to-my_dong_mnist
Building package...
Uploading package & generating Job-name...

New Job-name:  JOB_NAME 

[Usage] Status check:
dong train status -j JOB_NAME
```
### Deploy
### Check if the train job finishes
```bash
$ dong train status -j JOB_NAME

job-name: JOB_NAME
message: my dong mnist
status: Succeeded
```
### Deploy the model to an API endpoint

```bash
$ dong endpoint up JOB_NAME
Bring up...

New endpoint-name:  ENDPOINT_NAME 

[Usage] Status check:
dong endpoint status -e ENDPOINT_NAME
```
### Get endpoint IP
```bash
$ dong endpoint status -e ENDPOINT_NAME
Endpoint name: ENDPOINT_NAME
External ip: ENDPOINT_IP
Status: Running
```
### Test the endpoint

**After** the endpoint status becomes running,
```bash
curl -X POST \
  http://ENDPOINT_IP/api/v1/serve \
  -H 'Content-Type: application/json' \
  -H 'cache-control: no-cache' \
  -d '[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 224, 0, 0, 0, 0, 0, 0, 0, 70, 29, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 231, 0, 0, 0, 0, 0, 0, 0, 148, 168, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 195, 231, 0, 0, 0, 0, 0, 0, 0, 96, 210, 11, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 252, 134, 0, 0, 0, 0, 0, 0, 0, 114, 252, 21, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 45, 236, 217, 12, 0, 0, 0, 0, 0, 0, 0, 192, 252, 21, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 168, 247, 53, 0, 0, 0, 0, 0, 0, 0, 18, 255, 253, 21, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 84, 242, 211, 0, 0, 0, 0, 0, 0, 0, 0, 141, 253, 189, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 169, 252, 106, 0, 0, 0, 0, 0, 0, 0, 32, 232, 250, 66, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 15, 225, 252, 0, 0, 0, 0, 0, 0, 0, 0, 134, 252, 211, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 22, 252, 164, 0, 0, 0, 0, 0, 0, 0, 0, 169, 252, 167, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 9, 204, 209, 18, 0, 0, 0, 0, 0, 0, 22, 253, 253, 107, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 169, 252, 199, 85, 85, 85, 85, 129, 164, 195, 252, 252, 106, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 41, 170, 245, 252, 252, 252, 252, 232, 231, 251, 252, 252, 9, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 84, 84, 84, 84, 0, 0, 161, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 252, 252, 45, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 252, 252, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 252, 244, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 232, 236, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 179, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 254, 107, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 227, 254, 254, 9, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 254, 254, 165, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 203, 254, 254, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 53, 254, 254, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 254, 254, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 196, 254, 248, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 254, 254, 237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 254, 254, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 254, 238, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 252, 254, 223, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 254, 254, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 254, 238, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 252, 254, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 254, 254, 131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 254, 234, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 254, 204, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 211, 254, 196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 158, 254, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 157, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]'
```
The reply should look something like this
```
[[2.879309568548649e-10, 2.6625946239478004e-11, 2.580107016925126e-09, 5.453760706930488e-12, 0.9999690055847168, 3.3259575649147166e-10, 3.2778924019538636e-10, 4.35676184906697e-07, 2.190379821964683e-10, 3.0488341508316807e-05], [1.3478038130010361e-10, 0.9997259974479675, 6.728584622806011e-08, 5.9139901864568856e-09, 9.023875122693426e-07, 5.708465922182882e-10, 1.2523435088951373e-07, 0.0002721738419495523, 6.667413003924594e-07, 9.076808638042166e-09]]
```
