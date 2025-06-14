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

>코로나 펜데미 이전과 이후의 변수 상관관계가 역전되는 경우가 보임(CRB)<br>
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
#변수 설정
features = [
    'USD/KRW', 'Dollar_Index', 'CRB', 'VIX',
    'KOSPI', 'NASDAQ', 'S&P500', 'WTI', 'Gold', 'US10Y'
]
```
```
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
```

![image](https://github.com/user-attachments/assets/f1748849-0584-47ce-9d05-c1cf3b43405f)

__Dollar_Index, NASDAQ, US10Y, Gold에서 높은 연관성을 보임__

### 상위 4개의 변수로 재학습__

```
#사용 변수(상위 4개)
top_features = ['USD/KRW', 'Dollar_Index', 'NASDAQ', 'US10Y', 'Gold']
```

![image](https://github.com/user-attachments/assets/2456c2fe-f958-4d82-9a2d-635526345f84)

### LSTM 최종 성능 지표

![image](https://github.com/user-attachments/assets/d70a4248-031a-4d51-8460-b5d58dd1c2db)

<hr>

## LSTM 이외의 시계열 모델

__GRU모델과 최근 시계열 예측에서 주목받는 비딥러닝 기반 모델인<br> DLinear모델과의 비교 테스트를 통해 환율 예측에서의 모델 적합성을 판단__

### DLinear
- 비딥러닝 기반의 경량 모델
- 간단한 linear layer를 사용하여 __추세(trend)와 계절성(seasonal)__ 성분으로 분해, 선형적으로 예측

### GRU
- RNN의 일종, 시계열 데이터의 __시간적 패턴__ 을 기억하고 예측
- 시퀀스 전체를 GRU 셀에 넣어 각 시점에 대한 hidden state를 얻음

<hr>

## 모델 테스트 & 비교

### Best Parameter

- Optuna 함수 사용, best 파라미터 도출
- 하이퍼파라미터 샘플링
```
seq_len = trial.suggest_int("seq_len", 24, 96, step=12)
pred_len = trial.suggest_int("pred_len", 5, 20)
lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
```
```
Best Params: {'seq_len': 96, 'pred_len': 5, 'lr': 0.00020605303810428396, 'batch_size': 16}
```

### 공통 조건
- 공통변수 : 환율데이터(USD/KRW)
- 입력 기간 : 96, 예측 기간 : 1, 5, 10
- 학습 데이터 80% / 검증 데이터 20%
- 평가 : RMSE, MAE
- 슬라이딩 윈도우 예측으로 반복 평가

#### sliding_predict
```
def sliding_predict(model, test_data, scaler, seq_len, pred_len, target_idx):
  model.eval()
  true_all, pred_all = [], []
  with torch.no_grad():
    for i in range(0, len(test_data) - seq_len - pred_len + 1):
      x = test_data[i:i+seq_len].reshape(1, seq_len, input_dim)
      y = test_data[i+seq_len:i+seq_len+pred_len, target_idx].reshape(pred_len, 1)

      x_tensor = torch.tensor(x, dtype=torch.float32)
      pred_tensor = model(x_tensor).squeeze(0).numpy()
      y_true = scaler.inverse_transform(
        np.pad(np.zeros((pred_len, input_dim)), ((0,0),(0,0)), constant_values=0)
      )
      y_pred = scaler.inverse_transform(
        np.pad(np.zeros((pred_len, input_dim)), ((0,0),(0,0)), constant_values=0)
      )
      y_true[:, target_idx] = y.squeeze()
      y_pred[:, target_idx] = pred_tensor.squeeze()

      true_all.extend(y_true[:, target_idx])
      pred_all.extend(y_pred[:, target_idx])

  rmse = mean_squared_error(true_all, pred_all) ** 0.5
  mae = mean_absolute_error(true_all, pred_all)
  return true_all, pred_all, rmse, mae
```

### 모델별로 같은 조건으로 예측 기간, 사용 변수를 변경해가며 테스트
- 예측 기간 : 1, 5, 10
- 변수 :
   - 단변량(USD/KRW 환율 단독)
   - 다변량(환율, 달러지수, NASDAQ, US10Y, 금)
 
```
#파라미터
seq_len = 96
pred_len = 예측 기간 조정
input_dim = len(columns)
target_idx = 0  # 'USD/KRW'
train_size = int(len(scaled_df) * 0.8)
lr = 0.001
batch_size = 16
epochs = 20
```
```
#모델 정의
class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim):
        super(DLinear, self).__init__()
        self.linear_s = nn.Linear(seq_len, pred_len)
        self.linear_t = nn.Linear(seq_len, pred_len)
        self.input_dim = input_dim

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        trend = torch.mean(x, dim=1, keepdim=True).expand_as(x)
        seasonal = x - trend
        s_out = self.linear_s(seasonal.permute(0, 2, 1))
        t_out = self.linear_t(trend.permute(0, 2, 1))
        output = s_out + t_out + seq_last.permute(0, 2, 1)
        return output[:, 0, :].unsqueeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, pred_len, input_dim, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out.unsqueeze(-1)

class GRUModel(nn.Module):
    def __init__(self, pred_len, input_dim, hidden_size=64):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])
        return out.unsqueeze(-1)
```

![image](https://github.com/user-attachments/assets/ffb76fbe-cc7c-42f2-bbbd-46250d85d895)

### 최종 모델 테스트 결과
![image](https://github.com/user-attachments/assets/b889e65c-1bdb-4f1a-9485-e53f10d60e58)

### 결과 요약
- __예측일에 따른 경향__
   - 1일 예측 : GRU 모델이 단변량•다변량 모두에서 가장 우수, 빠른 반응성 학습에 강함
   - 5,10일 예측 :DLinear가 단/다변량 모두 안정적 → 중기 트렌드 반영에 강점

- __입력 방식에 따른 경향__
   - 단변량 입력 : 대체로 더 낮은 RMSE/MAE, 불필요한 변수 없이 학습 집중 가능
   - 다변량 입력 : 일부 모델(LSTM, GRU)의 경우 오히려 성능 저하 경향 보임 → 과적합 가능성?, 입력 차원 저하

- __모델별 특성 정리__
   - LSTM : 시계열 구조에 적합하지만 과적합 경향 있음
   - GRU : 짧은 시계열 예측에 가장 효율적, 빠른 수렴, 예측 기간 증가 시 성능 흔들림
   - DLinear : 단/다변량 상관없이 안정적, 장기 예측에 특히 강함, 복잡한 시계열에서는 적합하지 않을 것으로 보임

## 수집 데이터 .csv

[최종 병합 데이터](https://drive.google.com/file/d/1eDzd9QgtyD1pLJvPjnybaofBXrVC8RAV/view?usp=sharing)

## 최종 보고서 .pptx

[환율 예측 최종](https://drive.google.com/file/d/1IPi-5Jqd2lS8LKSVhVOpTy0ylRK5D1np/view?usp=sharing)

## 참고 문헌

딥러닝을 활용한 원화 환율 예측: 시장 및 웹데이터와 거시경제 지표의 활용<https://koasas.kaist.ac.kr/handle/10203/285143><br>
대용량 거시･금융 자료를 이용한 원/달러 환율 변동의 예측력 평가<https://www.smu.ac.kr/_attach/file/2022/07/MoOhsWhoeftIsdRypDKc.pdf><br>
Are Transformers Effective for Time Series Forecasting?<https://arxiv.org/pdf/2205.13504v2><br>
1차 프로젝트 - 데이터분석_경제지표를 통한 환율 예측 모델 생성<https://github.com/ganjjiang/first_project>

## 데이터 출처

한국은행 경제 통계 시스템 http://ecos.bok.or.kr <br>
실시간 환율 데이터,실시간 환율 데이터 Investing.com<br>
금융 데이터 Yahoo Finance<br>
