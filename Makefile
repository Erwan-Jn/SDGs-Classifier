
run_preprocess:
	python -c 'from scripts.api.clean_preprocess_api import preprocess_features; preprocess_features()'

# not sure about this one
run_upload_model:
	python -c 'from scripts.api.gcs_models import BucketManager, upload_file; upload_file()'

run_api:
	uvicorn	sdg_classifier_api.fast:app	--reload
