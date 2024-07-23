from kedro.pipeline import Pipeline, pipeline, node
from .nodes import deduplicate_data, datatype_conversion

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = deduplicate_data,
                inputs = ["ingested_data","params:deduplication_key"],
                outputs = "deduplicated_data",
                name = "deduplicate",
                namespace = "cleansing"
            ),
            node(
                func = datatype_conversion,
                inputs = ["deduplicated_data"],
                outputs = "cleansed_data",
                name = "datatype",
                namespace = "cleansing"
            )
        ]
    )
