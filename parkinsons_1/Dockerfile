FROM python:3.7

ARG VERSION

LABEL org.label-schema.version=$VERSION

COPY ./requirements.txt /webapp/requirements.txt

WORKDIR /webapp

RUN pip install --no-cache-dir -r requirements.txt

COPY ./webapp/* /webapp

COPY ./models/prod_models/* /webapp

ENTRYPOINT ["python"]

CMD ["app.py"]