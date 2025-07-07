# app.py
from lightning import LightningApp, LightningFlow
from lightning import PythonScript

class FinetuneFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.job = PythonScript(script_path="lightning_mistral_pipeline.py")

    def run(self):
        if not self.job.has_started:
            self.job.run()
        elif self.job.has_succeeded:
            print("✅ Training Complete!")

app = LightningApp(FinetuneFlow())


