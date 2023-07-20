import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取数据集
data = pd.read_csv("song_data.csv")

# 离散属性连续化
encoder = OneHotEncoder()
key_encoded = encoder.fit_transform(data[["key"]]).toarray()
audio_mode_encoded = encoder.fit_transform(data[["audio_mode"]]).toarray()
time_signature_encoded = encoder.fit_transform(data[["time_signature"]]).toarray()

key_df = pd.DataFrame({
                       "key0": [key_encoded[i][0] for i in range(len(key_encoded))],
                       "key1": [key_encoded[i][1] for i in range(len(key_encoded))],
                       "key2": [key_encoded[i][2] for i in range(len(key_encoded))],
                       "key3": [key_encoded[i][3] for i in range(len(key_encoded))],
                       "key4": [key_encoded[i][4] for i in range(len(key_encoded))],
                       "key5": [key_encoded[i][5] for i in range(len(key_encoded))],
                       "key6": [key_encoded[i][6] for i in range(len(key_encoded))],
                       "key7": [key_encoded[i][7] for i in range(len(key_encoded))],
                       "key8": [key_encoded[i][8] for i in range(len(key_encoded))],
                       "key9": [key_encoded[i][9] for i in range(len(key_encoded))],
                       "key10": [key_encoded[i][10] for i in range(len(key_encoded))],
                       "key11": [key_encoded[i][11] for i in range(len(key_encoded))]},
                       columns=["key0", "key1", "key2", "key3", "key4",
                       "key5", "key6", "key7", "key8", "key9", "key10", "key11"])

audio_mode_df = pd.DataFrame({
                       "audio_mode0": [audio_mode_encoded[i][0] for i in range(len(audio_mode_encoded))],
                       "audio_mode1": [audio_mode_encoded[i][1] for i in range(len(audio_mode_encoded))]},
                       columns=["audio_mode0", "audio_mode1"])

time_signature_df = pd.DataFrame({
                       "time_signature0": [time_signature_encoded[i][0] for i in range(len(time_signature_encoded))],
                       "time_signature1": [time_signature_encoded[i][1] for i in range(len(time_signature_encoded))],
                       "time_signature2": np.zeros(len(time_signature_encoded)),
                       "time_signature3": [time_signature_encoded[i][2] for i in range(len(time_signature_encoded))],
                       "time_signature4": [time_signature_encoded[i][3] for i in range(len(time_signature_encoded))],
                       "time_signature5": [time_signature_encoded[i][4] for i in range(len(time_signature_encoded))]},
                       columns=["time_signature0", "time_signature1", "time_signature2",
                                "time_signature3", "time_signature4", "time_signature5"])

# 归一化连续属性
scaler = MinMaxScaler()
data[["song_popularity", "song_duration_ms", "acousticness", "danceability", "energy", "instrumentalness",
      "liveness", "loudness", "speechiness", "tempo", "audio_valence"]] \
    = scaler.fit_transform(data[["song_popularity", "song_duration_ms", "acousticness", "danceability", "energy", "instrumentalness",
      "liveness", "loudness", "speechiness", "tempo", "audio_valence"]])

# 拼接特征
features = pd.concat([data[["song_duration_ms", "acousticness", "danceability", "energy", "instrumentalness",
      "liveness", "loudness", "speechiness", "tempo", "audio_valence"]], key_df, audio_mode_df, time_signature_df], axis=1)

# 检查共线性并处理
# 计算VIF
X = features.values
vif = pd.DataFrame()
vif["variables"] = features.columns
vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
# 输出结果
print("VIF is: \n", vif)

# 使用PCA进行特征降维
pca = PCA(n_components=X.shape[1])
# pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 输出每个主成分的贡献率
print(pca.explained_variance_ratio_)

# 输出每个主成分的权重
print(pca.components_)


# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, data["song_popularity"], test_size=0.2, random_state=42)

# 构建线性回归模型并训练
reg = LinearRegression()
reg.fit(X_train, y_train)

# 使用测试集进行预测并计算误差(MSE)
y_test_pred = reg.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("测试集误差（MSE）:", test_mse)

# 计算训练误差(MSE)
y_train_pred = reg.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("训练集误差（MSE）:", train_mse)

# 可视化模型和数据集
# 输出测试集前50个数据的song_popularity真实值和预测值
Y_Test = y_test.values[0:50:] * 100
Y_Test_Pred = y_test_pred[0:50:] * 100
label_X = np.arange(50)

plt.title('true values of song_popularity')
plt.scatter(label_X, Y_Test)
plt.show()

plt.title('predicate values of song_popularity')
plt.scatter(label_X, Y_Test_Pred)
plt.show()

# 进行50次随机划分训练集和测试集，进行50次实验，计算出每一次实验的误差
num = 50
TEST_MSE = np.zeros(50)
TRAIN_MSE = np.zeros(50)
for i in range(num):
    X_train, X_test, y_train, y_test = train_test_split(features, data["song_popularity"], test_size=0.2, random_state=i)
    y_test_pred = reg.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    TEST_MSE[i] = test_mse
    y_train_pred = reg.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    TRAIN_MSE[i] =train_mse

label_X = np.arange(50)
plt.title('MSE of test set')
plt.scatter(label_X, TEST_MSE)
plt.show()

plt.title('MSE of train set')
plt.scatter(label_X, TRAIN_MSE)
plt.show()

