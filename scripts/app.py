import os
import logging
import pandas as pd
import tensorflow as tf
import json
import streamlit as st
import numpy as np
import altair as alt
import shutil
import stat
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置固定路径
_base_dir = "/home/metadata/mlmd_penguin_pipeline"
_data_root = os.path.join(_base_dir, 'data')
_model_dir = os.path.join(_base_dir, 'model')
_serving_model_dir = os.path.join(_base_dir, 'serving_model')
_metadata_dir = os.path.join(_base_dir, 'metadata')


# 创建目录并检查写入权限
def ensure_directory(directory):
    try:
       os.makedirs(directory, exist_ok=True)
       if not os.access(directory, os.W_OK):
          st.error(f"无写入权限: {directory}")
          logger.error(f"无写入权限: {directory}")
          raise PermissionError(f"无写入权限: {directory}")
       if os.stat(directory).st_mode & stat.S_IWUSR == 0:
          st.error(f"目录 {directory} 无写权限，可能为只读文件系统")
          logger.error(f"目录 {directory} 无写权限，可能为只读文件系统")
          raise PermissionError(f"目录 {directory} 无写权限")
       os.chmod(directory, 0o755)
       total, used, free = shutil.disk_usage(directory)
       if free < 1024 * 1024:
          st.error(f"磁盘空间不足: {directory}，可用空间 {free / (1024 * 1024):.2f} MB")
          logger.error(f"磁盘空间不足: {directory}")
          raise RuntimeError(f"磁盘空间不足: {directory}")
       logger.info(f"目录创建或验证成功: {directory}")
       #st.write(f"目录验证成功: {directory}")
    except Exception as e:
       st.error(f"创建目录 {directory} 失败: {str(e)}")
       logger.error(f"创建目录 {directory} 失败: {str(e)}")
       raise


ensure_directory(_data_root)
ensure_directory(_model_dir)
ensure_directory(_serving_model_dir)
ensure_directory(_metadata_dir)


# 定义元数据结构
def create_artifact_type(name, properties):
    return {"name": name, "properties": properties}


def create_execution_type(name, properties):
    return {"name": name, "properties": properties}


data_type = create_artifact_type("DataSet", {"split": "STRING", "day": "INT", "dataset_name": "STRING"})
model_type = create_artifact_type("SavedModel", {"version": "INT", "name": "STRING", "dataset_name": "STRING"})
pushed_model_type = create_artifact_type("PushedModel", {"version": "INT", "dataset_name": "STRING"})
data_process_type = create_execution_type("DataProcessor", {"state": "STRING"})
trainer_type = create_execution_type("Trainer", {"state": "STRING"})
pusher_type = create_execution_type("Pusher", {"state": "STRING"})
experiment_type = create_execution_type("Experiment", {"note": "STRING"})


# 保存元数据函数
def save_artifact(artifact_type, artifact, filename, dataset_name):
    try:
       artifact_data = {
          "type": artifact_type["name"],
          "uri": artifact["uri"],
          "properties": artifact["properties"]
       }
       filepath = os.path.join(_metadata_dir, filename)
       with open(filepath, 'w') as f:
          json.dump(artifact_data, f)
       if not os.path.exists(filepath):
          st.error(f"元数据文件 {filepath} 未创建")
          logger.error(f"元数据文件 {filepath} 未创建")
          raise RuntimeError(f"元数据文件 {filepath} 未创建")
       st.write(f"元数据文件内容 ({filename}):", artifact_data)
       logger.info(f"保存元数据文件: {filepath}")
       return filename
    except Exception as e:
       st.error(f"保存工件失败: {filename}, 错误: {str(e)}")
       logger.error(f"保存工件失败: {filename}, 错误: {str(e)}")
       raise


def save_execution(execution_type, execution, filename):
    try:
       execution_data = {
          "type": execution_type["name"],
          "properties": execution["properties"]
       }
       filepath = os.path.join(_metadata_dir, filename)
       with open(filepath, 'w') as f:
          json.dump(execution_data, f)
       if not os.path.exists(filepath):
          st.error(f"元数据文件 {filepath} 未创建")
          logger.error(f"元数据文件 {filepath} 未创建")
          raise RuntimeError(f"元数据文件 {filepath} 未创建")
       st.write(f"元数据文件内容 ({filename}):", execution_data)
       logger.info(f"保存元数据文件: {filepath}")
       return filename
    except Exception as e:
       st.error(f"保存执行失败: {filename}, 错误: {str(e)}")
       logger.error(f"保存执行失败: {filename}, 错误: {str(e)}")
       raise


def save_event(artifact_id, execution_id, event_type, filename):
    try:
       event_data = {
          "artifact_id": artifact_id,
          "execution_id": execution_id,
          "type": event_type
       }
       filepath = os.path.join(_metadata_dir, filename)
       with open(filepath, 'w') as f:
          json.dump(event_data, f)
       if not os.path.exists(filepath):
          st.error(f"元数据文件 {filepath} 未创建")
          logger.error(f"元数据文件 {filepath} 未创建")
          raise RuntimeError(f"元数据文件 {filepath} 未创建")
       st.write(f"元数据文件内容 ({filename}):", event_data)
       logger.info(f"保存元数据文件: {filepath}")
       return filename
    except Exception as e:
       st.error(f"保存事件失败: {filename}, 错误: {str(e)}")
       logger.error(f"保存事件失败: {filename}, 错误: {str(e)}")
       raise


def save_context(context_type, context, filename):
    try:
       context_data = {
          "type": context_type["name"],
          "name": context["name"],
          "properties": context["properties"]
       }
       filepath = os.path.join(_metadata_dir, filename)
       with open(filepath, 'w') as f:
          json.dump(context_data, f)
       if not os.path.exists(filepath):
          st.error(f"元数据文件 {filepath} 未创建")
          logger.error(f"元数据文件 {filepath} 未创建")
          raise RuntimeError(f"元数据文件 {filepath} 未创建")
       st.write(f"元数据文件内容 ({filename}):", context_data)
       logger.info(f"保存元数据文件: {filepath}")
       return filename
    except Exception as e:
       st.error(f"保存上下文失败: {filename}, 错误: {str(e)}")
       logger.error(f"保存上下文失败: {filename}, 错误: {str(e)}")
       raise


# 数据验证和预处理
def validate_and_preprocess_data(df, label_column):
    st.write(f"原始数据集行数: {len(df)}")
    if label_column not in df.columns:
       st.error(f"标签列 '{label_column}' 不存在于数据集中")
       return None, None, None, None, None
    if df[label_column].isna().any():
       st.error(f"标签列 '{label_column}' 包含缺失值")
       return None, None, None, None, None
    feature_columns = [col for col in df.columns if col != label_column]
    if not feature_columns:
       st.error("数据集必须至少包含一列数值特征")
       return None, None, None, None, None
    for col in feature_columns:
       if not np.issubdtype(df[col].dtype, np.number):
          st.error(f"特征列 '{col}' 必须为数值型")
          return None, None, None, None, None
       if df[col].isna().any():
          st.error(f"特征列 '{col}' 包含缺失值")
          return None, None, None, None, None
    if np.any(np.isnan(df[feature_columns].values)) or np.any(np.isinf(df[feature_columns].values)):
       st.error("特征数据包含 NaN 或 inf")
       return None, None, None, None, None
    unique_labels = df[label_column].nunique()
    if unique_labels < 2:
       st.error("标签列必须至少包含两个不同类别")
       return None, None, None, None, None
    if unique_labels > 50:
       st.warning("标签列包含过多唯一值，可能不适合分类任务")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_column])
    num_classes = len(label_encoder.classes_)
    if max(y) >= num_classes or min(y) < 0:
       st.error(f"标签值范围 {min(y)}-{max(y)} 超出预期 [0, {num_classes - 1}]")
       return None, None, None, None, None
    st.write("编码后标签唯一值:", np.unique(y))
    label_mapping = {i: label_encoder.classes_[i] for i in range(num_classes)}
    st.write("原始标签分布:", df[label_column].value_counts().to_dict())
    st.write("编码后标签分布:", pd.Series(y).value_counts().to_dict())
    st.write("标签映射:", label_mapping)
    st.write(f"类别数量: {num_classes}")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns].values)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
       st.error("特征标准化后包含 NaN 或 inf")
       return None, None, None, None, None
    st.write("特征标准化完成，均值:", np.mean(X, axis=0).round(4))
    st.write("特征标准化完成，标准差:", np.std(X, axis=0).round(4))
    return X, y, feature_columns, num_classes, label_encoder


# 清理旧元数据文件
def clear_metadata_directory(timestamp):
    try:
       for filename in os.listdir(_metadata_dir):
          if filename.endswith((".json", ".csv")) and timestamp not in filename:
             try:
                os.remove(os.path.join(_metadata_dir, filename))
                logger.info(f"删除旧元数据文件: {filename}")
                #st.write(f"删除旧元数据文件: {filename}")
             except Exception as e:
                st.warning(f"删除旧元数据文件 {filename} 失败: {str(e)}")
                logger.warning(f"删除旧元数据文件 {filename} 失败: {str(e)}")
    except Exception as e:
       st.warning(f"清理旧元数据文件失败: {str(e)}")
       logger.warning(f"清理旧元数据文件失败: {str(e)}")


# 获取元数据文件列表
def get_metadata_files():
    try:
       files = [entry.name for entry in os.scandir(_metadata_dir) if entry.is_file() and entry.name.endswith(".json")]
       return sorted(files)
    except Exception as e:
       st.warning(f"获取元数据文件列表失败: {str(e)}")
       logger.warning(f"获取元数据文件列表失败: {str(e)}")
       return []


# 显示文件内容
def display_file_content(filename):
    try:
       filepath = os.path.join(_metadata_dir, filename)
       if not os.path.exists(filepath):
          st.warning(f"文件 {filename} 不存在")
          logger.warning(f"文件 {filename} 不存在")
          return
       with open(filepath, 'r') as f:
          content = json.load(f)
       st.write(f"文件内容 ({filename}):", content)
       logger.info(f"显示文件内容: {filename}")
    except Exception as e:
       st.warning(f"读取文件 {filename} 失败: {str(e)}")
       logger.warning(f"读取文件 {filename} 失败: {str(e)}")


# Streamlit 界面
st.title("分类工作流（支持任意数据集）")
#st.write("上传本地 CSV 文件，选择标签列，运行分类工作流并记录元数据到服务器路径。")

# 初始化会话状态
if 'run_timestamp' not in st.session_state:
    st.session_state.run_timestamp = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# 文件上传
uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    dataset_name = uploaded_file.name.replace(" ", "")  # 清理文件名中的空格
    st.write(f"当前数据集: {dataset_name}")
    st.write(f"元数据保存目录: {_metadata_dir}")
    if st.session_state.get('last_uploaded_file') != dataset_name:
       temp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       clear_metadata_directory(temp_timestamp)
       st.session_state.last_uploaded_file = dataset_name
       st.session_state.run_timestamp = None
    df = pd.read_csv(uploaded_file)
    st.write(f"数据集总行数: {len(df)}，以下显示前 10 行原始数据，训练使用归一化后的特征")
    st.dataframe(df.head(10))
    label_column = st.selectbox("请选择标签列（企鹅: species, 红酒: type）", options=df.columns.tolist())

    if st.button("运行分类工作流"):
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       st.session_state.run_timestamp = timestamp
       clear_metadata_directory(timestamp)
       X, y, feature_columns, num_classes, label_encoder = validate_and_preprocess_data(df, label_column)
       if X is None:
          st.error("数据预处理失败，请检查数据集和标签列")
          st.stop()
       # 验证数据集规模
       expected_rows = {'penguins_processed.csv': 334, 'winequality-red.csv': 1599, 'wine.data.csv': 178}
       if dataset_name in expected_rows and len(df) != expected_rows[dataset_name]:
          st.warning(f"数据集行数 ({len(df)}) 与预期 ({expected_rows[dataset_name]}) 不符，请检查文件")
       X_train, X_val, y_train, y_val = train_test_split(
          X, y, test_size=0.3, stratify=y, random_state=42
       )
       st.write(f"训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}")
       st.write("X_train 样本（前 5 行）:", X_train[:5].round(4))
       st.write("y_train 样本（前 5 行）:", y_train[:5])
       st.write("X_val 样本（前 5 行）:", X_val[:5].round(4))
       st.write("y_val 样本（前 5 行）:", y_val[:5])
       if max(y_train) >= num_classes or min(y_train) < 0:
          st.error(f"训练标签范围 {min(y_train)}-{max(y_train)} 超出预期 [0, {num_classes - 1}]")
          st.stop()
       if max(y_val) >= num_classes or min(y_val) < 0:
          st.error(f"验证标签范围 {min(y_val)}-{max(y_val)} 超出预期 [0, {num_classes - 1}]")
          st.stop()
       if len(X_val) < 50:
          st.warning(f"验证集样本数过少 ({len(X_val)})，可能导致评估不稳定")
       _data_filepath = os.path.join(_data_root, f"uploaded_data_{timestamp}.csv")
       try:
          df.to_csv(_data_filepath, index=False)
          if not os.path.exists(_data_filepath):
             st.error(f"数据集文件 {_data_filepath} 未创建")
             logger.error(f"数据集文件 {_data_filepath} 未创建")
             st.stop()
          st.success(f"数据集已保存至服务器路径: {_data_filepath}")
       except Exception as e:
          st.error(f"保存数据集失败: {str(e)}")
          logger.error(f"保存数据集失败: {str(e)}")
          st.stop()
       st.write("### 记录数据处理")
       logger.info("记录数据处理...")
       data_process = {"properties": {"state": "RUNNING"}}
       data_process_filename = save_execution(data_process_type, data_process,
                                              f"data_process_execution_{timestamp}.json")
       data_artifact = {
          "uri": _data_filepath,
          "properties": {"split": "train", "day": int(timestamp[:8]), "dataset_name": dataset_name}
       }
       data_artifact_filename = save_artifact(data_type, data_artifact, f"data_artifact_{timestamp}.json",
                                              dataset_name)
       save_event(data_artifact_filename, data_process_filename, "DECLARED_OUTPUT",
                  f"data_process_event_{timestamp}.json")
       data_process["properties"]["state"] = "COMPLETED"
       save_execution(data_process_type, data_process, data_process_filename)
       st.write("数据处理记录完成")
       logger.info("数据处理记录完成")
       st.write("### 训练分类模型")
       logger.info("训练分类模型...")
       trainer_run = {"properties": {"state": "RUNNING"}}
       trainer_run_filename = save_execution(trainer_type, trainer_run, f"trainer_execution_{timestamp}.json")
       save_event(data_artifact_filename, trainer_run_filename, "DECLARED_INPUT",
                  f"trainer_input_event_{timestamp}.json")
       class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
       class_weights = dict(enumerate(class_weights))
       if any(w <= 0 or np.isnan(w) or np.isinf(w) for w in class_weights.values()):
          st.error("错误：类权重包含无效值")
          logger.error("类权重包含无效值")
          st.stop()
       st.write("类权重:", {k: round(v, 4) for k, v in class_weights.items()})
       model = tf.keras.Sequential([
          tf.keras.layers.Dense(64, activation='relu', input_shape=(len(feature_columns),),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.3),
          tf.keras.layers.Dense(32, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.3),
          tf.keras.layers.Dense(num_classes, activation='softmax')
       ])
       try:
          model.compile(
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
          )
          st.write("模型编译成功")
       except Exception as e:
          st.error(f"模型编译失败: {str(e)}")
          logger.error(f"模型编译失败: {str(e)}")
          st.stop()
       try:
          history = model.fit(
             X_train, y_train,
             validation_data=(X_val, y_val),
             epochs=10,
             batch_size=64,
             class_weight=class_weights,
             verbose=1
          )
       except Exception as e:
          st.error(f"模型训练失败: {str(e)}")
          logger.error(f"模型训练失败: {str(e)}")
          st.stop()
       train_acc = history.history['accuracy']
       val_acc = history.history['val_accuracy']
       st.write("原始训练准确率:", [round(acc, 4) for acc in train_acc])
       st.write("原始验证准确率:", [round(acc, 4) for acc in val_acc])
       if np.any(np.isnan(train_acc)) or np.any(np.isinf(train_acc)) or \
             np.any(np.isnan(val_acc)) or np.any(np.isinf(val_acc)):
          st.error("错误：训练或验证准确率包含 NaN 或 inf")
          logger.error("训练或验证准确率包含 NaN 或 inf")
          st.stop()
       train_acc = np.clip(train_acc, 0, 1)
       val_acc = np.clip(val_acc, 0, 1)
       st.write("训练和验证准确率（裁剪后，范围 0 到 1）：")
       plot_data = pd.DataFrame({
          '训练准确率': train_acc,
          '验证准确率': val_acc,
          '训练损失': history.history['loss'],
          '验证损失': history.history['val_loss']
       }, index=range(1, len(train_acc) + 1))
       chart = alt.Chart(plot_data.reset_index().melt('index')).mark_line().encode(
          x=alt.X('index:O', title='Epoch', axis=alt.Axis(values=list(range(1, len(train_acc) + 1)))),
          y=alt.Y('value:Q', title='Accuracy / Loss', scale=alt.Scale(domain=[0, 1], clamp=True)),
          color=alt.Color('variable:N')
       )
       st.altair_chart(chart, use_container_width=True)
       st.write("### 模型评估")
       y_pred = model.predict(X_val)
       st.write("预测概率样本（前 5 行）:", y_pred[:5].round(4))
       if not np.all((y_pred >= 0) & (y_pred <= 1)) or not np.allclose(np.sum(y_pred, axis=1), 1, atol=1e-5):
          st.error("模型预测概率异常，非 [0, 1] 或和不为 1")
          logger.error("模型预测概率异常")
          st.stop()
       y_pred_classes = np.argmax(y_pred, axis=1)
       st.write("验证集实际标签分布:", pd.Series(y_val).value_counts().to_dict())
       st.write("验证集预测标签分布:", pd.Series(y_pred_classes).value_counts().to_dict())
       if len(np.unique(y_pred_classes)) < num_classes:
          st.warning("模型预测偏向部分类别，可能由于类别不平衡或训练不足")
       report = classification_report(y_val, y_pred_classes,
                                      target_names=[str(label_encoder.classes_[i]) for i in range(num_classes)],
                                      output_dict=False)
       st.text("分类报告:\n" + report)
       final_val_acc = val_acc[-1]
       if final_val_acc < 0.7:
          st.warning(f"验证准确率 ({final_val_acc:.2f}) 低于预期（企鹅 ~0.95，红酒 ~0.85），请检查数据集或调整模型")
       if final_val_acc > 0.99:
          st.warning(f"验证准确率 ({final_val_acc:.2f}) 过高，可能过拟合")
       try:
          model.save(_model_dir, save_format='tf')
          if not os.path.exists(_model_dir):
             st.error(f"模型目录 {_model_dir} 未创建")
             logger.error(f"模型目录 {_model_dir} 未创建")
             st.stop()
          st.write(f"模型已保存至服务器路径: {_model_dir}")
          logger.info("模型训练完成并保存")
       except Exception as e:
          st.error(f"保存模型失败: {str(e)}")
          logger.error(f"保存模型失败: {str(e)}")
          st.stop()
       model_artifact = {
          "uri": _model_dir,
          "properties": {"version": 1, "name": f"Model_{timestamp}", "dataset_name": dataset_name}
       }
       model_artifact_filename = save_artifact(model_type, model_artifact, f"model_artifact_{timestamp}.json",
                                               dataset_name)
       save_event(model_artifact_filename, trainer_run_filename, "DECLARED_OUTPUT",
                  f"trainer_output_event_{timestamp}.json")
       trainer_run["properties"]["state"] = "COMPLETED"
       save_execution(trainer_type, trainer_run, trainer_run_filename)
       st.write("训练记录完成")
       logger.info("训练记录完成")
       st.write("### 推送模型")
       logger.info("推送模型...")
       pusher_run = {"properties": {"state": "RUNNING"}}
       pusher_run_filename = save_execution(pusher_type, pusher_run, f"pusher_execution_{timestamp}.json")
       save_event(model_artifact_filename, pusher_run_filename, "DECLARED_INPUT",
                  f"pusher_input_event_{timestamp}.json")
       try:
          model.save(_serving_model_dir, save_format='tf')
          if not os.path.exists(_serving_model_dir):
             st.error(f"服务模型目录 {_serving_model_dir} 未创建")
             logger.error(f"服务模型目录 {_serving_model_dir} 未创建")
             st.stop()
          st.write(f"模型已推送至服务器路径: {_serving_model_dir}")
       except Exception as e:
          st.error(f"推送模型失败: {str(e)}")
          logger.error(f"推送模型失败: {str(e)}")
          st.stop()
       pushed_model_artifact = {
          "uri": _serving_model_dir,
          "properties": {"version": 1, "dataset_name": dataset_name}
       }
       pushed_model_artifact_filename = save_artifact(pushed_model_type, pushed_model_artifact,
                                                      f"pushed_model_artifact_{timestamp}.json", dataset_name)
       save_event(pushed_model_artifact_filename, pusher_run_filename, "DECLARED_OUTPUT",
                  f"pusher_output_event_{timestamp}.json")
       pusher_run["properties"]["state"] = "COMPLETED"
       save_execution(pusher_type, pusher_run, pusher_run_filename)
       st.write("模型推送记录完成")
       logger.info("模型推送记录完成")
       st.write("### 记录实验上下文")
       experiment = {
          "name": f"ClassificationExp_{timestamp}",
          "properties": {"note": f"分类实验 - {label_column} - {dataset_name}"}
       }
       experiment_filename = save_context(experiment_type, experiment, f"experiment_context_{timestamp}.json")
       st.write("实验上下文记录完成")
       logger.info("实验上下文记录完成")

    # 元数据文件选择
    st.write("### 查看元数据文件")
    st.write(f"元数据目录: {_metadata_dir}")
    st.write(f"当前工作目录: {os.getcwd()}")
    file_list = get_metadata_files()
    st.write("元数据目录文件列表:", file_list if file_list else "无文件")
    selected_file = st.selectbox("选择元数据文件查看内容", ["无"] + file_list)
    if selected_file != "无":
       display_file_content(selected_file)

    logger.info("元数据查看完成")
else:
    st.info("请上传一个本地 CSV 文件以继续")
