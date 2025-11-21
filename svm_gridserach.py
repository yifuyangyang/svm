import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import joblib

# ========== 1. 32x32 文本图片 -> 1x1024 向量 ==========
def img2vector(file_path):
    """
    将一个 32x32 的 0/1 文本图片转成 1x1024 的 numpy 向量
    """
    vec = np.zeros((1, 1024), dtype=np.float32)
    with open(file_path, 'r') as f:
        for i in range(32):
            line_str = f.readline().strip()
            # 防御：有时候行可能比 32 长/短
            line_str = line_str[:32].ljust(32, '0')
            for j in range(32):
                vec[0, 32 * i + j] = int(line_str[j])
    return vec

# ========== 2. 读取整个数据集 ==========
def load_dataset(dir_path):
    """
    遍历 dir_path 下所有 txt 文件
    文件名格式假定为：  digit_index.txt  例如：1_0.txt, 9_12.txt
    标签 = 文件名中 '_' 前面的数字
    """
    file_list = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    num_files = len(file_list)

    data_mat = np.zeros((num_files, 1024), dtype=np.float32)
    label_list = []

    for i, file_name in enumerate(file_list):
        full_path = os.path.join(dir_path, file_name)
        data_mat[i, :] = img2vector(full_path)

        # 提取标签
        class_str = file_name.split('_')[0]  # '1_7.txt' -> '1'
        label_list.append(int(class_str))

    return data_mat, np.array(label_list, dtype=np.int32)

# ========== 3. 指定你的训练集 / 测试集目录 ==========
train_dir = r"C:\Users\Administrator\Desktop\lesson3\digits\trainingDigits"   # 改成你的 402 个训练文件所在文件夹
test_dir  = r"C:\Users\Administrator\Desktop\lesson3\digits\testDigits"       # 改成你的 186 个测试文件所在文件夹

X_train, y_train = load_dataset(train_dir)
X_test,  y_test  = load_dataset(test_dir)

print("训练集形状：", X_train.shape, " 标签形状：", y_train.shape)
print("测试集形状：", X_test.shape,  " 标签形状：", y_test.shape)

# ========== 4. 配置 SVM + GridSearchCV ==========
print("开始进行参数搜索...")

# 创建SVC模型
svc = SVC(kernel="rbf", random_state=42)

# 设置参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']  # 同时搜索不同核函数
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,          # 5折交叉验证
    n_jobs=-1,     # 使用所有可用的CPU核心
    verbose=2      # 输出详细日志
)

# 在训练集上进行参数搜索
print("正在进行网格搜索，这可能需要一些时间...")
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"参数搜索完成，耗时: {end_time - start_time:.2f} 秒")
print("最优参数：", grid_search.best_params_)
print("交叉验证下的最佳平均准确率：", grid_search.best_score_)

# 显示所有参数组合的结果
print("\n所有参数组合的交叉验证结果：")
results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f"参数: {params} -> 准确率: {mean_score:.4f}")

# ========== 5. 使用最优模型在测试集上评估 ==========
print("\n" + "="*50)
print("在测试集上评估最优模型...")

# 获取最优模型
best_clf = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_clf.predict(X_test)

# 计算测试集准确率
test_acc = accuracy_score(y_test, y_pred)

print(f"测试集准确率：{test_acc:.4f}")

# 检查是否达到98%的目标
if test_acc >= 0.98:
    print("🎉 恭喜！已达到98%以上的准确率目标！")
else:
    print("⚠️  未达到98%的准确率目标，尝试增强参数搜索...")
    
    # 增强版参数搜索
    def enhanced_parameter_search():
        print("使用增强版参数搜索...")
        
        # 更精细的参数网格
        enhanced_param_grid = {
            'C': [1, 10, 50, 100, 200],
            'gamma': [0.0001, 0.001, 0.005, 0.01, 0.05],
            'kernel': ['rbf']
        }
        
        enhanced_svc = SVC(random_state=42)
        
        enhanced_grid_search = GridSearchCV(
            estimator=enhanced_svc,
            param_grid=enhanced_param_grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        
        enhanced_grid_search.fit(X_train, y_train)
        
        print("增强搜索最优参数：", enhanced_grid_search.best_params_)
        print("增强搜索最佳交叉验证准确率：", enhanced_grid_search.best_score_)
        
        return enhanced_grid_search

    # 运行增强搜索
    enhanced_grid_search = enhanced_parameter_search()
    best_clf = enhanced_grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"增强搜索后的测试集准确率：{test_acc:.4f}")
    
    # 再次检查是否达到目标
    if test_acc >= 0.98:
        print("🎉 恭喜！增强搜索后已达到98%以上的准确率目标！")
    else:
        print("⚠️  仍然未达到98%的准确率目标")

# 打印详细的分类报告
print("\n详细分类报告：")
print(classification_report(y_test, y_pred))

# 显示混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵：")
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - SVM Handwritten Digit Recognition')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存模型
model_filename = 'best_svm_digit_classifier.pkl'
joblib.dump(best_clf, model_filename)
print(f"最优模型已保存为 '{model_filename}'")

# 显示每个数字的分类准确率
print("\n各数字分类准确率：")
for digit in range(10):
    digit_indices = y_test == digit
    if np.sum(digit_indices) > 0:
        digit_accuracy = accuracy_score(y_test[digit_indices], y_pred[digit_indices])
        print(f"数字 {digit}: {digit_accuracy:.4f} ({np.sum(digit_indices)} 个样本)")

# 最终总结
print("\n" + "="*60)
print("项目总结：")
print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"最优参数: {grid_search.best_params_}")
print(f"交叉验证最佳准确率: {grid_search.best_score_:.4f}")
print(f"测试集最终准确率: {test_acc:.4f}")
print(f"目标达成: {'是' if test_acc >= 0.98 else '否'}")
print("="*60)