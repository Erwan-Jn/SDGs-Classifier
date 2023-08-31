run_api:
	uvicorn taxifare.api.fast:app --reload

run_preprocess:
	python -c 'from scripts.api.clean_preprocess_api import preprocess_features; preprocess_features()'
