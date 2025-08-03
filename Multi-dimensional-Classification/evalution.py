import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, hamming_loss
)
from sklearn.preprocessing import MultiLabelBinarizer
def evaluate_model(classification_result,model_name="qwen3-32B"):
    # ==== 工具函数：数据验证 ====
    def is_valid_single_label(value):
        """检查单标签值是否有效（整数或可转换为整数）"""
        if value is None:
            return False
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False

    def clean_single_label(value):
        """清洗单标签值，转换为整数"""
        try:
            return int(value)
        except:
            return None  # 标记为无效值

    # ==== 多标签评估指标：计算每个类别的准确率 ====
    def per_class_accuracy(y_true, y_pred):
        """计算每个类别的准确率"""
        # 确保输入是numpy数组
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        
        # 计算每个类别的准确率
        class_acc = []
        for i in range(y_true.shape[1]):
            # 正例预测正确或负例预测正确都算对
            correct = np.sum((y_true[:, i] == y_pred[:, i]))
            total = len(y_true[:, i])
            if total == 0:
                class_acc.append(0.0)
            else:
                class_acc.append(correct / total)
        return np.array(class_acc)

    # ==== 加载数据 ====
    with open(classification_result, "r", encoding="utf-8") as f:
        all_paired_data = json.load(f)

    # ==== 任务定义 ====
    single_label_tasks = [
        "entity", 
        "intent", 
        "time", 
        "eu_ai_act_risk_level"
    ]
    # 多标签任务现在分为主领域和子领域两个层级
    multi_label_tasks = {
        "domain_main": "domain",  # 第一层级
        "domain_sub": "subdomain"    # 第二层级
    }

    # 初始化存储真实标签和预测标签的字典
    y_true_single = {k: [] for k in single_label_tasks}
    y_pred_single = {k: [] for k in single_label_tasks}
    y_true_multi = {k: [] for k in multi_label_tasks.keys()}
    y_pred_multi = {k: [] for k in multi_label_tasks.keys()}

    # ==== 遍历配套数据并进行清洗 ====
    total_items = len(all_paired_data)
    valid_items = 0
    invalid_items = []

    print(f"共找到 {total_items} 组数据，开始处理并验证...")

    for idx, item in enumerate(all_paired_data):
        try:
            # 检查是否包含必要的字段
            if "label" not in item or "predict" not in item:
                raise ValueError("缺少label或predict字段")
                
            gt = item["label"]
            pred = item["predict"]
            item_valid = True
            
            # 处理单标签任务（带验证）
            for task in single_label_tasks:
                # 检查任务是否存在
                if task not in gt or task not in pred:
                    raise ValueError(f"任务 {task} 缺少标签或预测结果")
                    
                # 验证并清洗标签值
                true_val = gt[task]
                pred_val = pred[task]
                
                if not is_valid_single_label(true_val) or not is_valid_single_label(pred_val):
                    item_valid = False
                    raise ValueError(f"任务 {task} 包含无效值 (真实值: {true_val}, 预测值: {pred_val})")
                
                # 清洗并添加到列表
                y_true_single[task].append(clean_single_label(true_val))
                y_pred_single[task].append(clean_single_label(pred_val))
            
            # 处理领域分类的两个层级
            if "domain_classification" not in gt or "domain_classification" not in pred:
                raise ValueError("缺少domain_classification字段")
            
            # 提取主领域（第一层级）和子领域（第二层级）
            # 真实标签
            true_main = [int(item[0]) for item in gt["domain_classification"] if isinstance(item, list) and len(item)>=2]
            true_sub = [int(item[1]) for item in gt["domain_classification"] if isinstance(item, list) and len(item)>=2]
            
            # 预测标签
            pred_main = [int(item[0]) for item in pred["domain_classification"] if isinstance(item, list) and len(item)>=2]
            pred_sub = [int(item[1]) for item in pred["domain_classification"] if isinstance(item, list) and len(item)>=2]
            
            # 验证领域标签
            if not (isinstance(true_main, list) and isinstance(true_sub, list) and 
                    isinstance(pred_main, list) and isinstance(pred_sub, list)):
                raise ValueError("领域标签格式不正确，应为列表类型")
            
            # 添加到多标签任务列表
            y_true_multi["domain_main"].append(true_main)
            y_pred_multi["domain_main"].append(pred_main)
            y_true_multi["domain_sub"].append(true_sub)
            y_pred_multi["domain_sub"].append(pred_sub)
            
            if item_valid:
                valid_items += 1
                
        except Exception as e:
            invalid_items.append(f"第 {idx} 组数据无效: {str(e)}")

    # 打印数据验证结果
    print(f"数据处理完成 - 有效数据: {valid_items}/{total_items}")
    if invalid_items:
        print("无效数据详情:")
        for msg in invalid_items[:5]:  # 只显示前5条
            print(f"  - {msg}")
        if len(invalid_items) > 5:
            print(f"  - 还有 {len(invalid_items)-5} 条无效数据...")

    # 如果没有有效数据，终止程序
    if valid_items == 0:
        print("没有有效的数据用于评估，程序终止")
        exit()

    # ==== 多标签数据转换 ====
    # 为每个层级创建独立的MultiLabelBinarizer
    mlb_main = MultiLabelBinarizer()
    mlb_sub = MultiLabelBinarizer()

    # 拟合主领域标签
    all_main_labels = y_true_multi["domain_main"] + y_pred_multi["domain_main"]
    mlb_main.fit(all_main_labels)

    # 拟合子领域标签
    all_sub_labels = y_true_multi["domain_sub"] + y_pred_multi["domain_sub"]
    mlb_sub.fit(all_sub_labels)

    # 转换为二进制矩阵
    y_true_multi_bin = {
        "domain_main": mlb_main.transform(y_true_multi["domain_main"]),
        "domain_sub": mlb_sub.transform(y_true_multi["domain_sub"])
    }
    y_pred_multi_bin = {
        "domain_main": mlb_main.transform(y_pred_multi["domain_main"]),
        "domain_sub": mlb_sub.transform(y_pred_multi["domain_sub"])
    }

    # ==== 写入评估结果 ====
    output_path=os.path.join("evaluation_result",model_name+".txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("==== single label tasks evaluation====\n")
        f.write(f"valid samples: {valid_items}\n\n")
        
        for task in single_label_tasks:
            # 确保标签是numpy数组格式
            y_true = np.array(y_true_single[task])
            y_pred = np.array(y_pred_single[task])
            
            # 检查是否有足够的类别进行评估
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            if len(unique_labels) < 2:
                f.write(f"[{task}] 样本类别不足，无法计算评估指标\n")
                continue
                
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(
                y_true, 
                y_pred, 
                average="macro", 
                zero_division=0
            )
            rec = recall_score(
                y_true, 
                y_pred, 
                average="macro", 
                zero_division=0
            )
            f1 = f1_score(
                y_true, 
                y_pred, 
                average="macro", 
                zero_division=0
            )
            f.write(f"[{task}] Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}\n")

        f.write("\n==== multi label tasks evaluation ====\n")
        for task, task_name in multi_label_tasks.items():
            y_true = y_true_multi_bin[task]
            y_pred = y_pred_multi_bin[task]
            
            # 计算微平均准确率
            # 微平均：计算总体的正确预测比例
            total_correct = np.sum(y_true == y_pred)
            total_samples = y_true.size
            acc_micro = total_correct / total_samples if total_samples > 0 else 0
            
            # 计算宏平均准确率
            # 宏平均：计算每个类别的准确率，再取平均
            class_accs = per_class_accuracy(y_true, y_pred)
            acc_macro = np.mean(class_accs)
            
            # 其他指标
            h_loss = hamming_loss(y_true, y_pred)
            
            prec_micro = precision_score(
                y_true, 
                y_pred, 
                average="micro", 
                zero_division=0
            )
            rec_micro = recall_score(
                y_true, 
                y_pred, 
                average="micro", 
                zero_division=0
            )
            f1_micro = f1_score(
                y_true, 
                y_pred, 
                average="micro", 
                zero_division=0
            )
            
            prec_macro = precision_score(
                y_true, 
                y_pred, 
                average="macro", 
                zero_division=0
            )
            rec_macro = recall_score(
                y_true, 
                y_pred, 
                average="macro", 
                zero_division=0
            )
            f1_macro = f1_score(
                y_true, 
                y_pred, 
                average="macro", 
                zero_division=0
            )

            f.write(f"[{task_name}]\n")
            f.write(f"  Hamming loss: {h_loss:.4f}\n")
            f.write(f"  acc_micro: {acc_micro:.4f}  prec_micro: {prec_micro:.4f}  recall_micro: {rec_micro:.4f}  f1_micro: {f1_micro:.4f}\n")
            f.write(f"  acc_macro: {acc_macro:.4f}  prec_macro: {prec_macro:.4f}  recall_macro: {rec_macro:.4f}  f1_macro: {f1_macro:.4f}\n\n")
                    # 在计算宏平均后添加
            print(f"类别准确率: {class_accs}")
            print(f"每个类别的样本量: {y_true.sum(axis=0)}")

    print("✅ 评估完成，结果已保存至", output_path)
if __name__ == "__main__":
    evaluate_model("classification_result/kimi-k2-prompt.json","kimi-k2-prompt")
    evaluate_model("classification_result/qwen3-14B.json","qwen3-14B")
    evaluate_model("classification_result/qwen3-14B-sft.json","qwen3-14B-sft")
    evaluate_model("classification_result/qwen3-32B.json","qwen3-32B")
    evaluate_model("classification_result/qwen3-32B-sft.json","qwen3-32B-sft")