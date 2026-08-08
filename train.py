"""Main entry point for iARCS training."""

import logging
import hydra
from omegaconf import DictConfig

from iarcs.pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    pipeline = TrainingPipeline(config)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user.")
        if config.wandb.enabled and pipeline.wandb_run:
            pipeline.wandb_run.finish()


if __name__ == "__main__":
    main()
