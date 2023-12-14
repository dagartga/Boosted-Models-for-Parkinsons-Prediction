install:
	pip install -r requirements.txt

lint:
	pylint --disable=R,C src/pred_pipeline.py