FROM python:3.8

# system updates
RUN apt-get update

# create and switch to non-root user
RUN groupadd appgroup
RUN useradd -g appgroup -m appuser
USER appuser
WORKDIR /home/appuser/
ENV PATH /home/appuser/.local/bin:$PATH

# copy files into image
COPY requirements.txt .
COPY build_model.py .
COPY application.py .
COPY wsgi.py .

# install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --user

# build the model
RUN python build_model.py

# start it up
CMD gunicorn -w 3 -b :5000 -t 360 --reload wsgi:app
