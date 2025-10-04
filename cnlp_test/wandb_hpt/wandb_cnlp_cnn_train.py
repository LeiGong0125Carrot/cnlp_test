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
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_train_epochs', type=int, default=15)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser
    # ✅ 只保留CNN支持的参数
    parser.add_argument('--cnn_embed_dim', type=int, default=100)
    parser.add_argument('--cnn_num_filters', type=int, default=25)
    parser.add_argument('--cnn_filter_sizes', type=str, default='1,2,3')  # 改为字符串
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--max_seq_length', type=int, default=512)  # 添加max_seq_length参数
    parser.add_argument('--warmup_steps', type=int, default=0)  # 修正为warmup_steps

    
    args = parser.parse_args()

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    
    early_stopping_threshold = args.early_stopping_threshold
    early_stopping_patience = args.early_stopping_patience
    warmup_ratio = args.warmup_ratio
    num_train_epochs = args.num_train_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    tokenizer_name = args.tokenizer_name  # 改名
    cnn_embed_dim = args.cnn_embed_dim
    cnn_num_filters = args.cnn_num_filters
    # cnn_filter_sizes = args.cnn_filter_sizes
    cnn_filter_sizes = [int(x) for x in args.cnn_filter_sizes.split(',')]
    model = args.model
    lr_scheduler_type = args.lr_scheduler_type
    max_seq_length = args.max_seq_length  # 新增
    warmup_steps = args.warmup_steps  # 修正变量名

    run_name = f"pmv_{datetime.now().strftime('%m%d_%H%M')}_{str(uuid.uuid4())[:6]}"
    output_dir = f"sweep_results/{model}/{run_name}"

    cmd = [
        "python", "-m", "cnlpt.train_system_class",
        "--learning_rate", str(learning_rate),
        "--task_name", "out_hospital_mortality_30",
        "--data_dir", "/bigtemp/nkw3mr/cnlp_test/long-clinical-doc/datasets/30",
        "--model", model,                           # 新增：指定CNN模型
        "--tokenizer_name", tokenizer_name,         # 修改：使用tokenizer_name而非encoder_name
        "--cnn_embed_dim", str(cnn_embed_dim),
        "--cnn_num_filters", str(cnn_num_filters),
        "--cnn_filter_sizes"
        ] + [str(x) for x in cnn_filter_sizes] + [
        "--max_seq_length", str(max_seq_length),    # 新增：最大序列长度
        "--lr_scheduler_type", lr_scheduler_type,
        "--warmup_ratio", str(warmup_ratio),
        "--warmup_steps", str(warmup_steps),          # 新增：warmup步数
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--per_device_train_batch_size", str(batch_size),
        "--num_train_epochs", str(num_train_epochs),
        "--do_train", "--do_eval", "--do_predict",
        "--cache_dir", "cache_dir/",
        "--output_dir", output_dir,
        "--overwrite_output_dir",
        "--eval_strategy", "epoch",
        "--save_strategy", "epoch",
        "--load_best_model_at_end", "True",
        "--error_analysis",
        "--weight_classes",
        "--metric_for_best_model", "eval_acc_and_f1_class_1", 
        "--greater_is_better", "True",
        "--early_stopping_threshold", str(early_stopping_threshold),
        "--early_stopping_patience", str(early_stopping_patience),
        "--evals_per_epoch", "1",
        "--overwrite_output_dir",
        "--report_to", "wandb",  # 改回wandb
        "--fp16"
    ]
    env = os.environ.copy()
    env["WANDB_PROJECT"] = "Concept-Text-Classification"

    
    try:
        result = subprocess.run(cmd, cwd="/bigtemp/nkw3mr/cnlp_test/cnlp_transformers/src", check=True)
        # subprocess完成后再读取文件
        time.sleep(5)

        wandb.init(project="Concept-Text-Classification", 
            name=run_name, 
            resume="allow")
        os.chdir("/bigtemp/nkw3mr/cnlp_test/cnlp_transformers/src")
        eval_file = os.path.join(output_dir, "eval_results.json")
        print(f"Eval fileL {eval_file}")
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                results = json.load(f)
            accuracy = results["out_hospital_mortality_30"]["acc"]
            print(f"Eval results exists! Acc: {accuracy}")
            # 记录eval结果，包括每个class的指标
            wandb.log({
                "eval/accuracy": accuracy,
                "eval/f1_class_0": results["f1_class_0"] if "f1_class_0" in results else results["out_hospital_mortality_30"]["f1_class_0"],
                "eval/f1_class_1": results["f1_class_1"] if "f1_class_1" in results else results["out_hospital_mortality_30"]["f1_class_1"],
            })
        else:
            print(f"Eval results does not exist!")
        
        # 读取并记录best_eval结果
        best_eval_file = os.path.join(output_dir, "best_eval_results.json")
        if os.path.exists(best_eval_file):
            with open(best_eval_file, 'r') as f:
                best_eval_results = json.load(f)
            print("Best eval results exist!")
            best_eval_metrics = best_eval_results["out_hospital_mortality_30"]
            
            # 使用wandb section来分别记录best_eval结果（包含所有per-class metrics）
            wandb.log({
                "best_eval/accuracy": best_eval_metrics["acc"],
                "best_eval/macro_f1": best_eval_metrics["macro_f1"],
                "best_eval/micro_f1": best_eval_metrics["micro_f1"],
                "best_eval/auroc": best_eval_metrics["auroc"],
                "best_eval/f1_class_0": best_eval_metrics["f1_class_0"],
                "best_eval/f1_class_1": best_eval_metrics["f1_class_1"],
                "best_eval/acc_and_f1_class_0": best_eval_metrics["acc_and_f1_class_0"],
                "best_eval/acc_and_f1_class_1": best_eval_metrics["acc_and_f1_class_1"],
                "best_eval/precision_class_0": best_eval_metrics["precision_class_0"],
                "best_eval/precision_class_1": best_eval_metrics["precision_class_1"],
                "best_eval/recall_class_0": best_eval_metrics["recall_class_0"],
                "best_eval/recall_class_1": best_eval_metrics["recall_class_1"],
            })
        else:
            print("Best eval results does not exist!")
        
        test_file = os.path.join(output_dir, "test_results.json")
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_results = json.load(f)
            test_metrics = test_results["out_hospital_mortality_30"]
            print("Test results exist!")

            # 使用wandb section来分别记录test结果（包含所有per-class metrics）
            wandb.log({
                "test/accuracy": test_metrics["acc"],
                "test/macro_f1": test_metrics["macro_f1"],
                "test/micro_f1": test_metrics["micro_f1"],
                "test/auroc": test_metrics["auroc"],
                "test/f1_class_0": test_metrics["f1_class_0"],
                "test/f1_class_1": test_metrics["f1_class_1"],
                "test/acc_and_f1_class_0": test_metrics["acc_and_f1_class_0"],
                "test/acc_and_f1_class_1": test_metrics["acc_and_f1_class_1"],
                "test/precision_class_0": test_metrics["precision_class_0"],
                "test/precision_class_1": test_metrics["precision_class_1"],
                "test/recall_class_0": test_metrics["recall_class_0"],
                "test/recall_class_1": test_metrics["recall_class_1"],
            })
        else:
            print("Test results does not exist!")
        wandb.finish()
            
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
