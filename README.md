# 📈환율 예측 프로젝트

<hr>

## 주제선정 이유

1. __불안정한 경제 상황__

   경제 상황 악화에 따라 기업의 수익성, 국가 경제 및 정책 결정에 지대한 영향을 미치는 환율에 대한 분석과 예측의 필요성 증대, 환율 예측을 통해 환 리스크 헷지 전략 수립 가능

2. __실시간 대응을 위한 필요성__

    글로벌 금융시장이 24시간 움직이기 때문에 환율 변동성도 실시간이며, 환율은 금리 인플레이션, 국제 정세, 유가, 경제 지표 등 복합적인 변수의 집합체이기에 AI기반 예측 모델로 실시간 대응의 필요성이 요구됨

## 프로젝트 목표

1. __환율 예측 모델 개발__ : 최신 환율정보를 이용해 LSTM모델 테스트 진행 후 GRU/DLinear 등의 시계열 모델들과 비교 분석으로 가장 좋은 지표를 보이는 모델 선정 MAE, RMSE 등 정량적 성능 지표로 모델 성능 평가 후 예측 정확도 향상

2. __실시간 예측 기능 연계__ : Apache Kafka API 기반 실시간 환율 데이터 수집 환경 구축 및 예측 자동화 파이프라인 구축

## 수집 데이터

- __미국 달러 지수(Dallar_Index)__ : 달러 강세일수록 원화 약세 __(🔴양의 상관관계)__
- __원자재 가격 종합 지수(CRB)__ : 원자재가 상승 -> 무역적자 부담 __(코로나 이전 🔵음의 상관관계, 펜데믹 이후 구조적 변화 있음)__
- __변동성 지수,공포지수(VIX)__ : 시장 불안 커지면 안전자산 선호 -> 환율 상승 __(🔴약한 양의 상관, 불확실성 상승 시 환율 상승)__
- __금 값(Gold)__ : 큰 영향은 없으나 리스크 헷지 역할 __(🔵불확실성 반영, 환율과는 혼합적인 경향)__
- __한국 주식시장 지수(KOSPI)__ : 외국인 자금 유출입과 직접 관련 __(🔵음의 상관관계)__
- __미국 주식시장 지수(NASDAQ/S&P500)__ : __(⚪간접 영향, 위험 선호/회피 분위기 반영)__
- __미국 10년물 국채금리(US10Y)__ : 미국 금리 인상 -> 달러 강세 -> 환율 상승 __(🔴양의 상관관계)__
- __비트코인 가격 지수(Bitcoin)__ : 대체 자산 __(⚪최근 부각, 시기별로 구조적 해석 상이)__

### [2000~2025 변수 상관관계]
![image](https://github.com/user-attachments/assets/3ab2b3e9-1fca-40f8-8d59-e85769b0982b)

### [2020~2025 변수 상관관계(코로나 펜데믹 이후)]
![image](https://github.com/user-attachments/assets/2bf57bc8-f6ed-460f-b4e9-abf9fd027f7d)

>코로나 펜데미 이전과 이후의 변수 상관관계가 역전되는 경우가 보임(CRB)
>
>펜데믹 이전과 이후의 변수 차이로 테스트 필요!

<hr>

## LSTM 모델 테스트
### 공통 조건
- 공통변수 : 환율데이터(USD/KRW)
- 10일 시퀀스로 다음날(하루) 예측
- 학습 데이터 90% / 검증 데이터 10%
- 과적합 방지 : EarlyStopping, Dropout
- 평가 : RMSE, MAE

### 학습 기간, 사용 변수에 차이를 두며 테스트 진행

```
import seaborn as sns
import copy

#변수 설정
features = [
    'USD/KRW', 'Dollar_Index', 'CRB', 'VIX',
    'KOSPI', 'NASDAQ', 'S&P500', 'WTI', 'Gold', 'US10Y'
]
seq_len = 10

#시퀀스 생성 함수
def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

#모델 학습 함수
def train_and_evaluate(df_input, model_label, return_model=False):
    df_model = df_input[features].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_model)

    X, y = create_sequences(scaled, seq_len)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = Sequential([
        LSTM(64, dropout=0.2, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=50, batch_size=32,
              callbacks=[early_stop], verbose=1)

    pred_scaled = model.predict(X_val)
    usd_idx = features.index('USD/KRW')
    y_val_rescaled = scaler.inverse_transform(
        np.concatenate([y_val.reshape(-1, 1), np.zeros((len(y_val), len(features)-1))], axis=1)
    )[:, usd_idx]
    pred_rescaled = scaler.inverse_transform(
        np.concatenate([pred_scaled, np.zeros((len(pred_scaled), len(features)-1))], axis=1)
    )[:, usd_idx]

    rmse = np.sqrt(mean_squared_error(y_val_rescaled, pred_rescaled))
    mae = mean_absolute_error(y_val_rescaled, pred_rescaled)

    plt.figure(figsize=(12, 4))
    plt.plot(y_val_rescaled, label='실제 환율')
    plt.plot(pred_rescaled, label='예측 환율')
    plt.title(f'{model_label} (RMSE: {rmse:.2f}, MAE: {mae:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    result = {'Model': model_label, 'RMSE': rmse, 'MAE': mae}
    if return_model:
        return result, model, X_val, y_val
    else:
        return result

#훈련 기간별 테스트 성능 비교
train_periods = [
    ("2000-01-01", "2009-12-31"),
    ("2010-01-01", "2019-12-31"),
    ("2020-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31")
]
test_start, test_end = "2024-01-01", "2024-12-31"
df_test = df[(df['날짜'] >= test_start) & (df['날짜'] <= test_end)].copy()

multi_period_results = []

for start, end in train_periods:
    label = f"Train: {start} ~ {end}"
    df_train = df[(df['날짜'] >= start) & (df['날짜'] <= end)].copy()

    result, model, X_val, y_val = train_and_evaluate(df_train, label, return_model=True)

    df_model = df_test[features].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_model)
    X_test, y_test = create_sequences(scaled, seq_len)

    pred_scaled = model.predict(X_test)
    usd_idx = features.index('USD/KRW')
    y_test_rescaled = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features)-1))], axis=1)
    )[:, usd_idx]
    pred_rescaled = scaler.inverse_transform(
        np.concatenate([pred_scaled, np.zeros((len(pred_scaled), len(features)-1))], axis=1)
    )[:, usd_idx]

    rmse = np.sqrt(mean_squared_error(y_test_rescaled, pred_rescaled))
    mae = mean_absolute_error(y_test_rescaled, pred_rescaled)

    multi_period_results.append({
        "Train Period": f"{start} ~ {end}",
        "Test Period": f"{test_start} ~ {test_end}",
        "RMSE": rmse,
        "MAE": mae
    })

# 결과 시각화 및 출력
result_df = pd.DataFrame(multi_period_results)

plt.figure(figsize=(10, 5))
sns.barplot(data=result_df, x='Train Period', y='RMSE')
plt.title("훈련 기간별 테스트 성능 (2024년 기준)")
plt.ylabel("RMSE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 결과 테이블 출력
print(result_df)
```

![image](https://github.com/user-attachments/assets/db004635-c9f0-4364-90c5-70dbeada57df)
![image](https://github.com/user-attachments/assets/7028b3c4-03c4-44ce-b0d5-10742c2494cc)
![image](https://github.com/user-attachments/assets/a85f4a79-f941-4f8d-8f39-23a1c968de4c)
![image](https://github.com/user-attachments/assets/49ec7f53-7073-46d2-a92c-71224c3f2d30)

__최근 데이터로 학습기간을 설정한 모델이 더 좋은 성능을 보임__ 

<hr>

### 2022년도 부터 2023년도 까지 변수의 중요도를 분석(비트코인 제외)__

__중요한 변수만 추출 후 재학습__

```
# 변수 목록
features = [
    'USD/KRW', 'Dollar_Index', 'CRB', 'VIX',
    'KOSPI', 'NASDAQ', 'S&P500', 'WTI', 'Gold', 'US10Y'
]
seq_len = 10

# 시퀀스 생성 함수
def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# 모델 학습 함수
def train_and_evaluate(df_input, model_label, return_model=False):
    df_model = df_input[features].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_model)
    X, y = create_sequences(scaled, seq_len)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = Sequential([
        LSTM(64, dropout=0.2, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

    pred_scaled = model.predict(X_val)
    usd_idx = features.index('USD/KRW')
    y_val_rescaled = scaler.inverse_transform(
        np.concatenate([y_val.reshape(-1, 1), np.zeros((len(y_val), len(features)-1))], axis=1)
    )[:, usd_idx]
    pred_rescaled = scaler.inverse_transform(
        np.concatenate([pred_scaled, np.zeros((len(pred_scaled), len(features)-1))], axis=1)
    )[:, usd_idx]

    rmse = np.sqrt(mean_squared_error(y_val_rescaled, pred_rescaled))
    mae = mean_absolute_error(y_val_rescaled, pred_rescaled)

    result = {'Model': model_label, 'RMSE': rmse, 'MAE': mae}
    if return_model:
        return result, model, X_val, y_val
    else:
        return result
# 2022~2023 데이터셋
df_train_recent = df[(df['날짜'] >= "2022-01-01") & (df['날짜'] <= "2023-12-31")].copy()

# 모델 학습
results_recent, model_recent, X_val_recent, y_val_recent = train_and_evaluate(df_train_recent, '2022~2023 변수 분석용 모델', return_model=True)

# baseline RMSE 계산
baseline_pred = model_recent.predict(X_val_recent)
baseline_rmse = np.sqrt(mean_squared_error(y_val_recent, baseline_pred))

# Permutation Importance 함수
def permutation_importance(model, X_val, y_val, features, baseline_rmse):
    importances = []
    for i in range(X_val.shape[2]):
        X_temp = copy.deepcopy(X_val)
        np.random.shuffle(X_temp[:, :, i])
        pred = model.predict(X_temp)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        delta = rmse - baseline_rmse
        importances.append(delta)
    return pd.DataFrame({'Feature': features, 'ΔRMSE': importances}).sort_values(by='ΔRMSE', ascending=False)

# 중요도 분석 실행
importance_df = permutation_importance(model_recent, X_val_recent, y_val_recent, features, baseline_rmse)
plt.figure(figsize=(10, 5))
sns.barplot(data=importance_df, x='ΔRMSE', y='Feature', palette='viridis')
plt.title("2022~2023 훈련 모델의 변수 중요도 (ΔRMSE 기준)")
plt.tight_layout()
plt.show()
print(importance_df)
```

![image](https://github.com/user-attachments/assets/f1748849-0584-47ce-9d05-c1cf3b43405f)

__Dollar_Index, NASDAQ, US10Y, Gold에서 높은 연관성을 보임__

### 상위 4개의 변수로 재학습__

```
#사용 변수 설정
top_features = ['USD/KRW', 'Dollar_Index', 'NASDAQ', 'US10Y', 'Gold']
seq_len = 10

#시퀀스 생성 함수 (수정된 features 사용)
def create_sequences_lite(data, seq_len=10):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])  # 항상 첫 번째가 타겟
    return np.array(X), np.array(y)

#모델 학습 함수 (상위 변수용)
def train_lite_model(df_input, model_label):
    df_model = df_input[top_features].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_model)

    X, y = create_sequences_lite(scaled, seq_len)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = Sequential([
        LSTM(64, dropout=0.2, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

    # 예측
    pred_scaled = model.predict(X_val)
    usd_idx = top_features.index('USD/KRW')
    y_val_rescaled = scaler.inverse_transform(
        np.concatenate([y_val.reshape(-1, 1), np.zeros((len(y_val), len(top_features)-1))], axis=1)
    )[:, usd_idx]
    pred_rescaled = scaler.inverse_transform(
        np.concatenate([pred_scaled, np.zeros((len(pred_scaled), len(top_features)-1))], axis=1)
    )[:, usd_idx]

    # 평가
    rmse = np.sqrt(mean_squared_error(y_val_rescaled, pred_rescaled))
    mae = mean_absolute_error(y_val_rescaled, pred_rescaled)

    # 결과 시각화
    plt.figure(figsize=(12, 4))
    plt.plot(y_val_rescaled, label='실제 환율')
    plt.plot(pred_rescaled, label='예측 환율')
    plt.title(f'{model_label} (RMSE: {rmse:.2f}, MAE: {mae:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {'Model': model_label, 'RMSE': rmse, 'MAE': mae}

#2022~2023 데이터 추출
df_lite = df[(df['날짜'] >= "2022-01-01") & (df['날짜'] <= "2023-12-31")].copy()

#학습 및 평가 실행
lite_results = train_lite_model(df_lite, '상위 변수 5개 모델 (2022~2023)')
print(lite_results)
```

![image](https://github.com/user-attachments/assets/2456c2fe-f958-4d82-9a2d-635526345f84)

### LSTM 최종 성능 지표

![image](https://github.com/user-attachments/assets/d70a4248-031a-4d51-8460-b5d58dd1c2db)
