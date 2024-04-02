from google.cloud import storage

def create_gcs_bucket(project_id, bucket_name, region):
    """
    Creates a new GCS bucket in the specified project and region.

    Parameters:
    - project_id: Your GCP project ID.
    - bucket_name: The name of the bucket to create.
    - region: The region where the bucket will be created.

    Returns:
    - Bucket: The created GCS bucket object.
    """
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    new_bucket = storage_client.create_bucket(bucket, location=region)
    print(f"Bucket {new_bucket.name} created.")
    return new_bucket

from google.cloud import bigquery

def create_bq_dataset_and_table(project_id, dataset_id, table_id, region):
    """
    Creates a new dataset and table in BigQuery.

    Parameters:
    - project_id: Your GCP project ID.
    - dataset_id: The ID of the dataset to create.
    - table_id: The ID of the table to create within the dataset.
    - region: The region where the dataset will be located.

    Returns:
    - Table: The created BigQuery table object.
    """
    bq_client = bigquery.Client(project=project_id)

    # Create the dataset
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = region
    created_dataset = bq_client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset {created_dataset.dataset_id} created in project {project_id}.")

    # Create the table
    schema = [
        bigquery.SchemaField("example_field", "STRING", mode="NULLABLE")
    ]
    table_ref = dataset_ref.table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    created_table = bq_client.create_table(table, exists_ok=True)
    print(f"Table {created_table.table_id} created in dataset {created_dataset.dataset_id}.")

    return created_table
