from setuptools import setup, find_packages

setup(
    name="iarcs",
    version="1.0.0",
    description="iARCS: Iterative Agentic RL for Controllable 3D Scene Generation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "diffusers==0.23.1",
        "accelerate>=0.22.0",
        "transformers>=4.27.0",
        "peft>=0.13.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "wandb",
        "numpy",
        "pillow",
        "tqdm",
        "shapely>=2.0",
        "dm-tree",
        "python-dotenv",
        "google-genai",
    ],
)
