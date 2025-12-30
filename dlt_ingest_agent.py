from .base_ingest_agent import BaseIngestAgent
import dlt

class DLTIngestAgent(BaseIngestAgent):
    def run_dlt_pipeline(self, source_name, destination, dataset_name):
        pipeline = dlt.pipeline(
            pipeline_name=source_name,
            destination=destination,
            dataset_name=dataset_name
        )

        # Assume source function defined elsewhere in repo
        import sources
        source_fn = getattr(sources, source_name)
        load_info = pipeline.run(source_fn())
        return load_info

    def generate_mcp_metadata(self, load_info):
        return {
            "metadata": {
                "source": "dlt",
                "resources": load_info.resources,
                "dataset_name": load_info.dataset_name
            }
        }

    def run(self, source_config):
        load_info = self.run_dlt_pipeline(
            source_name=source_config["source_name"],
            destination=source_config["destination"],
            dataset_name=source_config["dataset_name"]
        )
        return self.generate_mcp_metadata(load_info)

