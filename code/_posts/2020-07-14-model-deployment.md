---
layout: code-post
title: Intro to Productionization- APIs and Docker
description: A lecture given for the Erdos Institute about the basics of ML productionization
tags: demo
---

While a data scientist might rely heavily on Jupyter notebooks to run local experiments
and choose a model, the story does not end here. Ultimately the goal is to get a model
into _production_, i.e., make it available for use by the data scientist's clients, whether
they be internal or external clients. Some of the components that go into this
include

1. Exposing the model via an API
2. Standardizing the build and production environments
3. Monitoring model performance
4. Logging diagnostic events / troubleshooting errors

We will focus on the first two points above.

## Exposing a model via API

API stands for _Application Programming Interface_ and refers to how your
code is exposed to others. For example, people will talk about the fact
that `matplotlib` as two APIs: the one based on repeatedly plotting and 
the one based on using `axes` objects. You also hear people talk about accessing
Twitter's API to get the content of tweets.

For our case, we will mean making a model available for others to call.

### Building a model

Before jumping into exposing a model, let's build one first. Since the
modelling isn't the point here, we'll just build a gradient boosted machine
predictor to predict on the classicial 
[Iris](https://archive.ics.uci.edu/ml/datasets/iris) dataset.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

# load dataset
iris = datasets.load_iris()
df_iris = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
df_iris['target'] = iris['target']

# create train test split
df_train, df_test = train_test_split(df_iris,
                                     test_size=0.3,
                                     random_state=47,
                                     stratify=df_iris['target'])

df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.6</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>4.8</td>
      <td>3.1</td>
      <td>1.6</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>6.4</td>
      <td>2.9</td>
      <td>4.3</td>
      <td>1.3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.6</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>135</th>
      <td>7.7</td>
      <td>3.0</td>
      <td>6.1</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# put together pipeline
model = Pipeline([
    ('scale', StandardScaler())
    ,('gbm', GradientBoostingClassifier())
])

# make model
model.fit(df_train[iris['feature_names']], df_train['target'])

# see how accurate it is on test data
test_predictions = model.predict(df_test[iris['feature_names']])
test_accuracy = accuracy_score(df_test['target'], test_predictions)
print('Test accuracy of fit model: {:.3f}'.format(test_accuracy))
```

    Test accuracy of fit model: 0.933


The most basic way to deliver this model would be to actually deliver this
notebook to a user, but that requires that the user be able to run a Jupyter
notebook and know how to call the model. This might be a valid way to deliver
a model to an internal user who is technically proficient such as a data
analyst or another data scientist.
But instead of delivering a notebook, we can binarize the model object itself
using the `pickle` module which is built into base python and save it to disk. So let's do this
and save the model to disk.

```python
import pickle

with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
```

The model can then be reloaded from disk and called as needed.

```python
import numpy as np

with open('model.pickle', 'rb') as f:
    model_ = pickle.load(f)

print('model prediction:', model_.predict(np.random.normal(size=(1, 4)))[0])

del model_
```

    model prediction: 0


With the model saved as a pickle file, you can then pass the model easily between different users
without them having to retrain the model. This isn't really a way to deliver a model to an
end user, but again, more of a way to pass models around between internal users.

### Flask

Now let's move on to exposing the model via an API. We will be using
[Flask](https://flask.palletsprojects.com/en/1.1.x/), which
is one of several web micro-fameworks for python. A _web framework_ is
a set of tools for a language that helps programmers deal with common tasks
necessary to set up web applications. This can include standardizing names
and interactions with databases, serving static content such as images from 
specific file directory locations, and allowing language specific code to be
used in setting up the HTML files that are ultimately displayed in a public
facing website. Examples of framworks include 
[django](https://www.djangoproject.com/) and
[cherryPy](https://cherrypy.org/) for python and 
[rails](https://rubyonrails.org/) for ruby. Flask is fairly lightweight and 
easy to get started with, so we will use this.

A simple application might look like the following file:

```python
%%writefile application.py

# imports related to the model we have built
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# imports related to flask and loading the model
from flask import Flask, request, jsonify
import pickle

json_header = {'content-type': 'application/json; charset=UTF-8'}
model = pickle.load(open('model.pickle', 'rb'))
app = Flask(__name__)    

@app.route('/', methods=['GET'])
def get_prediction():
    
    args = request.args

    # check that all args are present
    desired_args = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    missing_args = [a for a in desired_args if args.get(a) is None]
    
    if len(missing_args) > 0:
        error_msg = 'argument(s) missing: {}'.format(missing_args)
        return (jsonify(error_msg), 422, json_header)
                
    # check that all args are floats
    def arg_is_float(arg):
        is_float = False
        
        try:
            x = float(arg)
            is_float = True
        except ValueError:
            pass
        
        return is_float
    
    nonfloat_args = [a for a in desired_args if not arg_is_float(args.get(a))]
    
    if len(nonfloat_args) > 0:
        error_msg = 'argument(s) not float: {}'.format(nonfloat_args)
        return (jsonify(error_msg), 422, json_header)
    
    # make predictions
    X = [[float(args.get(a)) for a in desired_args]]
    prediction = str(model.predict(X)[0])
    
    return (jsonify({'prediction': prediction}), 200, json_header)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
```

    Writing application.py


We have now written out to the file `application.py`. Without getting very technical, 
we'll point out some of the important features.

At the top of the file we have
```python
json_header = {'content-type': 'application/json; charset=UTF-8'}
model = pickle.load('model.pickle')
app = Flask(__name__)    
```
Here we have a JSON header that is a `dict`, we load the model from a 
pickle file, and we create a `Flask` object using the `__name__` variable.
The JSON header is used to tell the user calling our application what
sort of data to expect to receive in response to the request they make.

Note that at the bottom of the file we have
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0')
```
This causes the application to run with host `0.0.0.0`, aka the `localhost` aka
our local machine will run the application.

Between the initial creation of the application and
running it if appropriate we have the following lines.
```python
@app.route('/', methods=['GET'])
def get_prediction():
    
    args = request.args
```
Here we have a function named `get_prediction()` that is decorated with 
`@app.route('/', methods=['GET'])`. This tells Flask that to look for an HTTP `GET` request
at the location `/`. If the application receives a request at this location, it
then tries to run this function.  The first thing that happens in this function
is to store `request.args` as `args`. Note that `request` is not mentioned anywhere
in the module as either a local or global variable. This means that `request` must
be a parameter that is inherited from the `@app.route` decoration. Indeed it is, and it
contains all the information the application receives from the user at this route.
The function then processes `args`, which is a `dict`. In a minute we will see some of
what gets put into this variable.

Following this is code that attempts to validates the contents of `args`. We
first check that all of our desired arguments are included among the keys of the
incoming arguments.
```python
    desired_args = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    missing_args = [a for a in desired_args if args.get(a) is None]
    
    if len(missing_args) > 0:
        error_msg = 'argument(s) missing: {}'.format(missing_args)
        return (jsonify.dumps(error_msg), 422, json_header)
```
Here we see that if some of the desired arguments are missing we exit
the function early. We dump an error message as a JSON object
along with the [_HTTP response status code_](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes) 
422, which indicates that the application was not able to process the request properly.
We also pass in our json header object. Similarly, we then check that the values of the
arguments can be cast as floats.

The function ends by making a prediction and writing out the result with response status
code 200, the everything was fine on our end signal.
```python
    X = [[float(args.get(a)) for a in desired_args]]
    prediction = model.predict(X)[0]
    
    return (jsonify({'prediction': prediction}), 200, json_header)
```

Then, in your terminal, you can start the application locally by running
```bash
$ python application.py
 * Serving Flask app "application" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```

The last line of the output tells you that this is running at port 5000.
In a separate terminal you can test the API by running
```bash
$ curl 'localhost:5000/?sepal_length=7.7&sepal_width=3.0&petal_length=6.1&petal_width=2.3'
{"prediction":"2"}
```
(You can also replace `localhost` with `0.0.0.0`.)

And voilà! Your model is now available via an API... even if it's only locally.
This delivery mechanism could be achieved by sending the files `application.py`
and `model.pickle` to whoever wants to run this application, assuming they have
the ability to receive incoming requests and the proper versions of python
and all its packages installed... We'll get to the environment problem soon, but
first, let's speed things up.

### Gunicorn

When we ran our Flask application, we helpfully received the message
```bash
WARNING: This is a development server. Do not use it in a production deployment.
```
The reason for this warning is that Flask is slow. The web server that Flask
creates works by sitting there waiting
for requests which it tries to execute one at a time. This can cause requests to back
up and might cause errors. It's fine for development, but is not sufficient for 
production.

A quick way to speed things up is to pair Flask with the python package
[`gunicorn`](https://gunicorn.org/) aka Green Unicorn. (Unicorn is a web server 
for ruby applications, pythons are green I suppose so... green unicorn.) Gunicorn will
sit in front of the Flask web server and create multiple workers. When a request
comes it, it will be routed to one fo the workers that are hopefully idle.

In order to use gunicorn, we will add the following file:

```python
%%writefile wsgi.py

from application import app

if __name__ == "__main__":
    app.run(use_reloader=True, debug=True)
```

    Writing wsgi.py


The acronym WSGI stands for [_Web Server Gateway Interface_](https://wsgi.readthedocs.io/en/latest/)
which is a Python standard specification for how web servers and web applications communicate.
Gunicorn is a WSGI server and will use this file to load `app` from the `application` module
that we previously wrote. With just this bit of code, we can now start a webserver with
gunicorn with the command
```bash
$ gunicorn -w 3 -b :5001 -t 360 --reload wsgi:app
[2020-06-28 15:40:12 -0400] [86149] [INFO] Starting gunicorn 20.0.4
[2020-06-28 15:40:12 -0400] [86149] [INFO] Listening at: http://0.0.0.0:5001 (86149)
[2020-06-28 15:40:12 -0400] [86149] [INFO] Using worker: sync
[2020-06-28 15:40:12 -0400] [86152] [INFO] Booting worker with pid: 86152
[2020-06-28 15:40:12 -0400] [86153] [INFO] Booting worker with pid: 86153
[2020-06-28 15:40:12 -0400] [86154] [INFO] Booting worker with pid: 86154
```
You can tell that we have started three workers and we are listening at
port 5001 of `localhost`. We can again cURL to get a response:
```bash
$ curl 'localhost:5001/?sepal_length=7.7&sepal_width=3.0&petal_length=6.1&petal_width=2.3'
{"prediction":"2"}
```
Note that we did not have to curl a specific worker, we just had to 
curl to the port and gunicorn took care of delgating the task for us.

### Nginx?

The gunicorn documentation will also tell you not to deploy an application just
with gunicorn and will point you to nginx. We won't use the nginx webserver here
(apache is another popular one), but we'll remark that nginx sits in front of
gunicorn in a similar way to how gunicorn sits in front of flask. Nginx
should be used to actually accept traffic from the web and will direct
requests to appropriate places and serve files directly to users if needed. Nginx
is an all purpose web server, while gunicorn just serves python applications.



## Standardizing environments



A problem which has long plagued software engineers and now also data scientists
is the fact that environments are not standardized. One data scientist will not
necessarily have the same exact version of `pandas` installed. Or even the same
verison of `python`. Or even the same operating system. How do we solve for this?

### Local environments

One way to make sure that everyone is using the same packages and python
versions is to use the `venv` module. Going this route will create directories
in the directory where venv is called and installed python packages will live there.
At least among data scientists, this has fallen out of fashion. More information
can be found [here](https://docs.python.org/3/library/venv.html).

Since most python based data scientists use Anaconda, it is convenient to rely
on the environment abilities of the `conda` tool, which is able to create
environments from environment files that can be checked into `git` repositories.
For example, here is the contents of the `environment.yml` file that I am using:
```yaml
name: erdos
dependencies:
    - python=3.8
    - pandas=1.0.*
    - numpy=1.18.*
    - ipykernel=5.1.*
    - jupyterlab=1.2.*
    - scikit-learn=0.22.1
    - nb_conda_kernels=2.2.*
    - flask
    - gunicorn
    - pip
```
This is a YAML file, where YAML which stands for YAML Ain't Markup Language. YAML
files are just text files with either the `.yml` or `.yaml` endings which are
increasingly common in software engineering in various "__ as code" situations.
You can create a new conda environment and make it available to jupyter
by running
```bash
$ conda env create -f environment.yml
$ python -m ipykernel install --user --name erdos --display-name "Python (erdos)"
```
Updates are handled via `conda env update -f environment.yml`.


### Production environments


While conda is fine for maintaing local environments among teammates working
on a project, production servers will not have anaconda installed and may not
even have the same operating system as personal development laptops. I.e.,
you might have mac OS or an ubuntu on your machine, but the production servers might
run CentOS.

A decade ago the answer to the operating system issue would be virtual machines. 
A virtual machine allows you to run a full copy of one operating system on another.
The production cluster where applications are deployed might contain servers running 
CentOS or Red Hat Enterprise Linux or whatever operating system,
but these would just be hosts that run virtual machines that can then be any
other operating system. If your application required a flavor of linux you 
would deploy to the appropriate virtual machine.

This still does not solve the package and versioning issues. The technology 
that developed to solve both issues at once is called _Docker_. [Docker](https://www.docker.com/)
bundles together basic operating system information and application information into
an _image_. Without getting into 
technical details, it suffices to know that docker images are much more lightweight
than full virtual machines even though they naively seem very similar. Docker
images define what are called _microservices_; where a virtual machine can run many
applications at the same time, a docker image should only handle small
tasks and will not have many applications running at once.

If you have docker installed, here's how to see which images you have locally and to run `bash` inside
 of an ubuntu image.
```bash
$ docker images
$ docker run --rm -it ubuntu:18.04 /bin/bash
```

If you don't have this version of the image locally, the required images will 
be automatically pulled from the [default docker registry](https://hub.docker.com/_/ubuntu).
Playing around inside you can see that basic things like `python` are not 
installed, but it is a version of the ubuntu operating system. It's also much
smaller than an ubuntu VM, being only 64 MB.
A running instance of an image is running it is called a _container_, such that
you can have many containers running from the same image. You can see
the running containers with the command `docker ps` and see all stopped
containers alon with the running containers with `docker ps -a`. Here is the
[docker cheat sheet](https://www.docker.com/sites/default/files/d8/2019-09/docker-cheat-sheet.pdf).

With that introduction out of the way, let's get to an example of how to build
a docker image and package up our gunicorn + flask based API.

The basic unit that defines how an image is built is a  _Dockerfile_. We will walk 
through the following Dockerfile to see how one might build an image to deploy
our model.

```
FROM python:3.8

# system updates
RUN apt-get update

# create and switch to non-root user
RUN groupadd appgroup
RUN useradd -g appgroup -m appuser
USER appuser
WORKDIR /home/appuser/

# install requirements
COPY requirements.txt .
ENV PATH /home/appuser/.local/bin:$PATH
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --user

# build the model
COPY build_model.py .
RUN python build_model.py

# start it up
COPY application.py .
COPY wsgi.py .
CMD gunicorn -w 3 -b :5000 -t 360 --reload wsgi:app
```

So let's walkthrough this to get a feel for the kinds of things that we can do when building
a docker image.

We start off with `FROM python:3.8` which specifies that we want to start with the base
image that is called `python` and is tagged with version `3.8`. This base image exists
in the default [docker registry](https://hub.docker.com/_/python). Your company will have its
own privately hosted registry that you might pull images from and push images to (similar to
pulling and pushing from a git repository). Docker images are built in layers and
this tells it which image to start building from.
The python image uses a Debian based (aka Ubuntu-like) operating system.

When we ran the Ubuntu image earlier, you might have noticed that we were running as
the root user with full privileges. This is not usually recommended for security reasons,
so the first thing we'll do is create a new group and user that we can run as. This is
contained in the lines
```text
# create and switch to non-root user
RUN groupadd appgroup
RUN useradd -g appgroup -m appuser
USER appuser
WORKDIR /home/appuser/
```

Note that if we have a `RUN` command, then that command is run inside the image that
we are building, not locally. This is important to keep in mind, because if you want to
use `RUN` to call a piece of code, that code needs to exist inside the image.
The `USER` command switches us to that user. The
`WORKDIR` command switches which directory we are in. 

The next thing we do is we set up the python packages that we want. We will
put our requirements not in a YAML file, but in a plain text file that
we can install from using pip. 
```text
COPY requirements.txt .
ENV PATH /home/appuser/.local/bin:$PATH
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --user
```

We can see that we used the `COPY` command to copy the requirements
file from our local directory into the image. We then append a local
directory to our path (this was necessary to call gunicorn), upgrade `pip`,
and finally install requirements from the file that we loaded
into the image.


The next bit of code copies a new module called `build_model.py` into 
the image and runs it inside the image to build our model and save it
into the image. We'll see this module in a minute.
```text
# build the model
COPY build_model.py .
RUN python build_model.py
```

Recall that we built the model using a dataset that came loaded into
`scikit-learn`. Normally you will have to configure your image to
pull data from one of your company's databases. This might involve
somehow passing your credentials into the image so that the image has
the right to access the data or if the images are being built automatically
on some other server, more likely you'll have to pass in some other set of
credentials. A fun problem to solve is to write an image that you can
build locally with your own credentials for testing and also build using
whatever system your company uses to automatically build images.

Next we copy in the `application.py` and `wsgi.py` modules that we
wrote earlier. 
```text
# start it up
COPY application.py .
COPY wsgi.py .
CMD gunicorn -w 3 -b :5000 -t 360 --reload wsgi:app
```

The final line contains the `CMD` that is used by default  when
starting a container frmo this image.
(Earlier when we started the ubuntu container, we passed in `/bin/bash` which would 
overwrite this command.)

Below this are the `build_model.py` and `requirements.txt` files.

```python
%%writefile build_model.py

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


# load dataset
iris = datasets.load_iris()
df_iris = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
df_iris['target'] = iris['target']

# create train test split
df_train, df_test = train_test_split(df_iris,
                                     test_size=0.3,
                                     random_state=47,
                                     stratify=df_iris['target'])

# put together pipeline
model = Pipeline([
    ('scale', StandardScaler())
    ,('gbm', GradientBoostingClassifier())
])

# make model
model.fit(df_train[iris['feature_names']], df_train['target'])

# see how accurate it is on test data
test_predictions = model.predict(df_test[iris['feature_names']])
test_accuracy = accuracy_score(df_test['target'], test_predictions)
print('Test accuracy of fit model: {:.3f}'.format(test_accuracy))

# save model
pickle.dump(model, open('./model.pickle', 'wb'))
```

    Writing build_model.py


```python
%%writefile requirements.txt
scikit-learn==0.22.1
pandas==1.0.5
flask==1.1.2
gunicorn==20.0.4
```

    Writing requirements.txt


Finally, let's examine what happens when we build this image.
To build an image and `tag` it with the name `iris_api` we run
the following:
```bash
$ docker build . -t iris_api
```

Examaning the output of this command, you can see it building
the image in layers. If you change one part of the Dockerfile,
docker might not have to rebuild the parts above your image (although
this is not entirely true... note that it will always need to install
the requirements...).
As we did with the Ubuntu container, let's start a container
from this image. Instead of running the default `CMD` above,
let's get into the image.
```bash
$ docker run --rm -it iris_api /bin/bash
```

Note that we were in the `WORKDIR` that we specified earlier
and that we were the `appuser` user and not `root`. To start
a container with the default command, we can run.
```bash
$ docker run --rm -p 8747:5000 --name iris_api_1 iris_api
```

Recall that the `CMD` we specified for the image ran gunicorn on port `5000`.
In this `docker run` command we use the `-p 8747:5000` option to attach localhost's
port `8747` to the containers port `5000`. In another terminal window, we
can curl to this port on localhost to get a result.
```bash
$ curl 'localhost:8747/?sepal_length=7.7&sepal_width=3.0&petal_length=6.1&petal_width=2.3'
```

And now if you want to deliver your model to someone, all you need to care about
is that they have docker installed. The operating system doesn't matter and they
don't even need to have python installed! 

### Next Steps

- Investigate how you can set environment variables inside your image using `ENV` and `ARG`
commands. What is the difference between them?
- Instead of copying a file into the image, try _mounting_ a file as a volume
into a container, writing to it from inside the container, 
and confirming that the local file persists with that data after the container is gone.
- It is often necessary to pass in secrets such as username + password combinations when
building an image. Look into enabling the _buildkit_ and how to use the `--secret` flag
so that passwords are not saved in intermediate images.
- Create your own private registry by running a registry inside a container on 
your local machine. Tag an image that exists locally, push it to the registry, 
remove the local image, and pull down the image from the registry container.
- Build your model in one image, then create a second image based on the first image
(similar to how our image was based on the `python:3.8` image) to serve the model.
- Use [docker compose](https://docs.docker.com/compose/) to create an application that has
multiple containers. Have one container run nginx (these containers exist already!) and 
have one run your model application. Curl requests to the nginx container instead of the
model application container.
- Put the iris data into a database (such as postgres) running in one image and have your model pull
the data from that image (you will need to create a user that your image can use to log in
as!).
- Investigate [Kubernetes](https://kubernetes.io/), aka k8s, which can be used to run
many images.
- Try running your docker image on a virtual machine in the cloud.

This post was created for a lecture for the [Erdos institute](erdosinstitute.org).
