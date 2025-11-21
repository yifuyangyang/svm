import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

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
train_dir = r"C:\dataset\trainingDigits"   # 改成你的 402 个训练文件所在文件夹
test_dir  = r"C:\dataset\testDigits"       # 改成你的 186 个测试文件所在文件夹

X_train, y_train = load_dataset(train_dir)
X_test,  y_test  = load_dataset(test_dir)

print("训练集形状：", X_train.shape, " 标签形状：", y_train.shape)
print("测试集形状：", X_test.shape,  " 标签形状：", y_test.shape)

# ========== 4. 配置 SVM + GridSearchCV ==========
# 这里以 RBF 核为主（手写识别常用），搜索 C 和 gamma
"""
【任务 1】：
    使用 SVC + GridSearchCV 在训练集上搜索最优参数（例如 C 和 gamma）。
"""

# 创建SVC模型
svc = SVC(kernel="rbf")

# 构造参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1]
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,          # 5折交叉验证
    n_jobs=1,      # 改为1，避免多进程导致的编码问题
    verbose=1      # 输出详细日志
)

# 在训练集上进行网格搜索
print("开始网格搜索...")
grid_search.fit(X_train, y_train)

# 输出最优参数和得分
print("最优参数：", grid_search.best_params_)
print("交叉验证下的最佳平均准确率：", grid_search.best_score_)


# ========== 5. 使用最优模型在测试集上评估 ==========

"""
【任务 2】：
    使用从 GridSearchCV 中得到的最优模型，在测试集上评估性能。
"""

# 从 grid_search 中取出最优模型
best_clf = grid_search.best_estimator_

# 使用 best_clf 对测试集进行预测
y_pred = best_clf.predict(X_test)

# 计算测试集上的准确率
test_acc = accuracy_score(y_test, y_pred)

# 打印结果
print("测试集准确率：", test_acc)

# 打印更详细的分类报告
print("\n详细分类报告：")
print(classification_report(y_test, y_pred))