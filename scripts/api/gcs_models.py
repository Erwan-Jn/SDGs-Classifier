from google.cloud import storage
import pandas as pd


# to push model on gcloud storage
class BucketManager:
    """Needs for example :
        CONFIG = {
            'project_id': "gcs-project_id-397610",
            'bucket_name': "bucket_name",
            'local_file_path': "local_file_path",
            'remote_file_path': "remote_file_path" #on the cloud path
            }"""
    def __init__(self, project_id, bucket_name):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.client = storage.Client(project=self.project_id)
        self.bucket = self.client.bucket(self.bucket_name)

    def upload_file(self, local_path, remote_path):
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        print(f"File uploaded: {local_path} -> gs://{self.bucket_name}/{remote_path}")

    def download_file(self, remote_path, local_path):
        blob = self.bucket.blob(remote_path)
        blob.download_to_filename(local_path)
        print(f"File downloaded: gs://{self.bucket_name}/{remote_path} -> {local_path}")
# if __name__ == "__main__":
#     manager = BucketManager(CONFIG['project_id'], CONFIG['bucket_name'])
#     # Uploading a file
#     manager.upload_file(CONFIG['local_file_path'], CONFIG['remote_file_path'])
#     # # Downloading a file
#     manager.download_file(CONFIG['remote_file_path'], CONFIG['local_file_path'])
