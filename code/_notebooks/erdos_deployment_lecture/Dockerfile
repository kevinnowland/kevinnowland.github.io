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
