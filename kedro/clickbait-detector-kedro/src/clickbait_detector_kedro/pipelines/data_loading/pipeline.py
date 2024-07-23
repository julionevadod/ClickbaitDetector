"""
This is a boilerplate pipeline 'data_loading'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = lambda x: x,
                inputs = ["input_data"],
                outputs = "ingested_data",
                name = "ingest_data",
                namespace = "ingestion"
            )
        ]
    )
