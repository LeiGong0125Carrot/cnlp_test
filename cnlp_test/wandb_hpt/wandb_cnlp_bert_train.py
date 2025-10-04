import wandb
import subprocess
import json
import os
from datetime import datetime
import uuid
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--encoder_name', type=str, default='roberta-base')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.01)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    args = parser.parse_args()

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    encoder_name = args.encoder_name
    early_stopping_threshold = args.early_stopping_threshold
    lr_scheduler_type = args.lr_scheduler_type
    
    run_name = f"pmv_{datetime.now().strftime('%m%d_%H%M')}_{str(uuid.uuid4())[:6]}"
    output_dir = f"sweep_results/bert/{run_name}"

    cmd = [
        "python", "-m", "cnlpt.train_system_class",
        "--learning_rate", str(learning_rate),
        "--task_name", "pmv_prediction",
        "--data_dir", "/bigtemp/nkw3mr/clinical-outcome-prediction/converted_dataset/pmv_prediction",
        "--encoder_name", encoder_name,
        "--per_device_train_batch_size", str(batch_size),
        "--num_train_epochs", str(15),
        "--do_train", "--do_eval", "--do_predict",
        "--cache_dir", "cache_dir/",
        "--output_dir", output_dir,
        "--overwrite_output_dir",
        "--eval_strategy", "epoch",
        "--save_strategy", "epoch",
        "--load_best_model_at_end", "True",
        "--freeze", "-1.0",
        "--error_analysis",
        "--weight_classes",
        "--metric_for_best_model", "acc", 
        "--greater_is_better", "True",
        "--early_stopping_threshold", str(early_stopping_threshold),
        "--lr_scheduler_type", lr_scheduler_type,
        "--evals_per_epoch", "1",
        "--report_to", "wandb"  # 改回wandb
    ]
    env = os.environ.copy()
    env["WANDB_PROJECT"] = "VIB-Intepretable-Text-Classification"

    
    try:
        result = subprocess.run(cmd, cwd="/bigtemp/nkw3mr/cnlp_test/cnlp_transformers/src", check=True)
        # subprocess完成后再读取文件
        time.sleep(5)

        wandb.init(project="VIB-Intepretable-Text-Classification", 
            name=run_name, 
            resume="allow")
        os.chdir("/bigtemp/nkw3mr/cnlp_test/cnlp_transformers/src")
        eval_file = os.path.join(output_dir, "eval_results.json")
        print(f"Eval fileL {eval_file}")
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                results = json.load(f)
            accuracy = results["pmv_prediction"]["acc"]
            print(f"Eval results exists! Acc: {accuracy}")
            wandb.log({"final_metric": accuracy})
        else:
            print(f"Eval results does not exist!")
        
        # 读取并记录best_eval结果
        best_eval_file = os.path.join(output_dir, "best_eval_results.json")
        if os.path.exists(best_eval_file):
            with open(best_eval_file, 'r') as f:
                best_eval_results = json.load(f)
            print("Best eval results exist!")
            best_eval_metrics = best_eval_results["pmv_prediction"]
            
            # 使用wandb section来分别记录best_eval结果
            wandb.log({
                "best_eval/accuracy": best_eval_metrics["acc"],
                "best_eval/macro_f1": best_eval_metrics["macro_f1"],
                "best_eval/micro_f1": best_eval_metrics["micro_f1"],
                "best_eval/auroc": best_eval_metrics["auroc"],
            })
        else:
            print("Best eval results does not exist!")
        
        test_file = os.path.join(output_dir, "test_results.json")
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_results = json.load(f)
            print("Test results exist!")
            test_metrics = test_results["pmv_prediction"]

            # 使用wandb section来分别记录test结果
            wandb.log({
                "test/accuracy": test_metrics["acc"],
                "test/macro_f1": test_metrics["macro_f1"],
                "test/micro_f1": test_metrics["micro_f1"],
                "test/auroc": test_metrics["auroc"],
            })
        else:
            print("Test results does not exist!")
        wandb.finish()
            
    except Exception as e:
        print(f"Training failed: {e}")
        # wandb.log({"final_metric": 0.0})

if __name__ == "__main__":
    main()
