# ====================== 导入依赖 ======================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

# 设置中文绘图（解决图表乱码）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 图表保存文件夹
os.makedirs("eda_plots", exist_ok=True)

# ====================== 1. 加载TAAC2026 Demo数据集 ======================
df=pd.read_parquet("demo_1000.parquet",engine="pyarrow")
print("数据集全部列名：", df.columns.tolist())

# ====================== 2. 样本维度分析（实训要求第一部分） ======================
print("【一、样本维度统计分析】")
# 2.1 总样本量
total_sample = len(df)
print(f"1. 样本总量：{total_sample}")

# 2.2 label正负样本比例（广告点击标签，1=点击正样本，0=未点击负样本）
label_counts = df["label_type"].value_counts()
pos_num = label_counts[2]
neg_num = label_counts[1]
pos_ratio = pos_num / total_sample
neg_ratio = neg_num / total_sample
imb_ratio = neg_num / pos_num  # 类别不平衡系数

print(f"2. 标签分布：")
print(f"   正样本(点击)：{pos_num} 条，占比 {pos_ratio:.2%}")
print(f"   负样本(未点击)：{neg_num} 条，占比 {neg_ratio:.2%}")
print(f"   正负样本不平衡比例(负/正)：{imb_ratio:.2f}")

# 绘制label分布柱状图
plt.figure(figsize=(8, 5))
sns.barplot(x=label_counts.index, y=label_counts.values, palette=["#ff7f0e", "#2ca02c"])
plt.title("样本Label正负分布（点击/未点击）", fontsize=14)
plt.xlabel("label（1=未点击，2=点击）")
plt.ylabel("样本数量")
plt.savefig("eda_plots/01_label_dist.png", dpi=300, bbox_inches="tight")
plt.close()

# 2.3 时间戳时间跨度分析（数据集含time字段）
df["time_datetime"] = pd.to_datetime(df["label_time"], unit="s")
time_min = df["time_datetime"].min()
time_max = df["time_datetime"].max()
time_span_days = (time_max - time_min).days
print(f"3. 时间跨度：最早{time_min}, 最晚{time_max}，覆盖时长 {time_span_days} 天")

# 时间分布折线图
plt.figure(figsize=(10,4))
df["time_datetime"].dt.date.value_counts().sort_index().plot()
plt.title("每日曝光样本量时间分布")
plt.xlabel("日期")
plt.ylabel("当日样本数")
plt.xticks(rotation=45)
plt.savefig("eda_plots/02_time_dist.png", dpi=300, bbox_inches="tight")
plt.close()

print("="*60)

# ====================== 3. 特征维度分析（实训要求第二部分） ======================
print("【二、特征维度统计分析】")
# 3.0 区分特征类型：标签列、非序列标量特征、序列行为特征、稠密向量特征
all_cols = df.columns.tolist()
label_col = ["label_type", "label_time", "user_id", "item_id"]  # 标签/主键/时间
# 序列特征：存储为list数组（用户历史行为序列，官方定义4个行为域序列）
seq_cols = []
for col in all_cols:
    # 跳过整列全空
    non_null = df[col].dropna()
    if len(non_null) == 0:
        continue
    val = non_null.iloc[0]
    if isinstance(val, (list, np.ndarray)):
        seq_cols.append(col)
# 稠密向量特征：float数组稠密向量
dense_vec_cols = [col for col in all_cols if (type(df[col].iloc[0]) is np.ndarray and len(df[col].iloc[0])>5)]
# 普通非序列离散/连续特征（标量int/float）
non_seq_cols = [col for col in all_cols if col not in label_col+seq_cols+dense_vec_cols]

print(f"特征分类统计：")
print(f"1. 主键/标签列：{len(label_col)} 个")
print(f"2. 非序列标量特征：{len(non_seq_cols)} 个")
print(f"3. 用户行为序列特征：{len(seq_cols)} 个，序列字段名：{seq_cols}")
print(f"4. 稠密向量特征：{len(dense_vec_cols)} 个")

# 3.1 非序列特征：缺失值、稀疏度统计
missing_info = {}
sparse_info = {}
print("\n【非序列特征缺失&稀疏统计】")
for col in tqdm(non_seq_cols, desc="遍历标量特征"):
    # 缺失值统计
    miss_cnt = df[col].isna().sum()
    miss_ratio = miss_cnt / total_sample
    missing_info[col] = {"缺失数量": miss_cnt, "缺失占比": miss_ratio}
    # 稀疏度：离散特征0值占比（推荐系统稀疏特征核心指标）
    zero_cnt = (df[col] == 0).sum()
    zero_ratio = zero_cnt / total_sample
    sparse_info[col] = zero_ratio

# 缺失值可视化
miss_df = pd.DataFrame(missing_info).T
plt.figure(figsize=(12,5))
sns.barplot(x=miss_df.index, y="缺失占比", data=miss_df)
plt.title("非序列特征缺失值占比分布")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("eda_plots/03_feature_missing.png", dpi=300)
plt.close()

# 3.2 序列特征：序列长度分布（核心分析点）
seq_len_all = []
seq_name_list = []
print("\n【行为序列特征长度分布统计】")
for seq_col in seq_cols:
    seq_lengths = [len(x) if x is not None else 0 for x in df[seq_col]]
    seq_len_all.extend(seq_lengths)
    seq_name_list.extend([seq_col]*len(seq_lengths))
    mean_len = np.mean(seq_lengths)
    max_len = np.max(seq_lengths)
    min_len = np.min(seq_lengths)
    print(f"{seq_col}：平均序列长度{mean_len:.1f}，最长{max_len}，最短{min_len}")

# 绘制所有序列长度分布直方图
seq_df = pd.DataFrame({"序列字段": seq_name_list, "序列长度": seq_len_all})
plt.figure(figsize=(10,6))
sns.histplot(data=seq_df, x="序列长度", hue="序列字段", multiple="dodge")
plt.title("各行为序列特征长度分布")
plt.savefig("eda_plots/04_seq_length_dist.png", dpi=300, bbox_inches="tight")
plt.close()

# 3.3 全局缺失总览（全部字段）
total_missing = df.isna().sum().sum()
total_missing_ratio = total_missing / (df.shape[0] * df.shape[1])
print(f"\n【全局数据缺失汇总】")
print(f"全表缺失单元格总数：{total_missing}，全局缺失占比：{total_missing_ratio:.4%}")

# ====================== 4. 统计结果导出（用于EDA报告） ======================
# 把全部统计数据保存为csv，写报告直接引用
# 样本统计
sample_stat = pd.DataFrame([
    {"指标":"总样本量", "数值": total_sample},
    {"指标":"正样本数量", "数值": pos_num},
    {"指标":"负样本数量", "数值": neg_num},
    {"指标":"正负不平衡比例", "数值": round(imb_ratio, 2)},
    {"指标":"数据时间跨度(天)", "数值": time_span_days}
])
sample_stat.to_csv("eda_report/sample_stat.csv", index=False, encoding="utf-8-sig")

# 特征缺失统计导出
os.makedirs("eda_report", exist_ok=True)
miss_df.to_csv("eda_report/feature_missing_stat.csv", encoding="utf-8-sig")
print("\n✅ 全部统计结果已导出至 ./eda_report 文件夹，图表保存在 ./eda_plots")
print("✅ EDA分析全部完成，可基于图表与csv撰写分析报告！")