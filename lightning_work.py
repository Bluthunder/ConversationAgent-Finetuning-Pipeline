from lightning import LightningWork
import subprocess

class FinetuneWork(LightningWork):
    def run(self):
        # Option 1: If your script has a main() entry point
        import finetuning_pipeline
        finetuning_pipeline.main()

        # Option 2: Run as subprocess
        # subprocess.run(["python", "finetuning_pipeline.py"], check=True)
