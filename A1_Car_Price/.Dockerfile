FROM python:3.9-slim

WORKDIR /root/code

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install seaborn

# Testing module
RUN pip3 install dash[testing]

COPY ./code /root/code

CMD tail -f /dev/null
