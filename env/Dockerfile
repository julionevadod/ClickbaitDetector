FROM python:3.9-slim

COPY . /ClickbaitDetector

WORKDIR /ClickbaitDetector

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python"]

CMD ["-m", "train"]
