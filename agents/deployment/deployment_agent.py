import os
import shutil
import json
import joblib

class DeploymentAgent:
    def __init__(self):
        print("📦 DeploymentAgent initialized")
        self.deployment_dir = "models/latest"

    def deploy(self, model_path="artifacts/model/model.pkl", 
               pipeline_path="artifacts/feature_engineering/pipeline.pkl",
               metadata_path="artifacts/feature_engineering/metadata.json",
               target_encoder_path="artifacts/feature_engineering/target_encoder.pkl"):
        print("Starting deployment packaging...")
        
        # Ensure deployment directory exists
        os.makedirs(self.deployment_dir, exist_ok=True)
        
        # Define target paths
        target_model = os.path.join(self.deployment_dir, "model.pkl")
        target_pipeline = os.path.join(self.deployment_dir, "pipeline.pkl")
        target_metadata = os.path.join(self.deployment_dir, "metadata.json")
        target_encoder = os.path.join(self.deployment_dir, "target_encoder.pkl")
        
        # Copy artifacts
        if os.path.exists(model_path):
            shutil.copy2(model_path, target_model)
            print(f"Copied model to {target_model}")
        else:
            print(f"Warning: Model not found at {model_path}")

        if os.path.exists(pipeline_path):
            shutil.copy2(pipeline_path, target_pipeline)
            print(f"Copied pipeline to {target_pipeline}")
        else:
            print(f"Warning: Pipeline not found at {pipeline_path}")
            
        if os.path.exists(metadata_path):
            shutil.copy2(metadata_path, target_metadata)
            print(f"Copied metadata to {target_metadata}")
        else:
            print(f"Warning: Metadata not found at {metadata_path}")
            
        if os.path.exists(target_encoder_path):
            shutil.copy2(target_encoder_path, target_encoder)
            print(f"Copied target encoder to {target_encoder}")
        else:
            if os.path.exists(target_encoder):
                os.remove(target_encoder)

        print("✅ Deployment bundle created successfully.")
        return self.deployment_dir

if __name__ == "__main__":
    agent = DeploymentAgent()
    agent.deploy()
