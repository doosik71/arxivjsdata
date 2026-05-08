# PATE: Proximity-Aware Time series anomaly Evaluation

Ramin Ghorbani, Marcel J.T. Reinders, David M.J. Tax (2024)

## 🧩 Problem to Solve

본 논문은 시계열 데이터의 이상치 탐지(Anomaly Detection) 알고리즘을 평가하는 기존 지표들의 한계를 해결하고자 한다. 시계열 데이터에서의 이상치는 단순한 점(point)이 아니라 특정 시간 구간(interval)으로 발생하며, 이에 따라 탐지 성능을 평가할 때 다음과 같은 복잡한 시간적 역학을 고려해야 한다.

1. **조기 탐지(Early Detection):** 이상치가 완전히 나타나기 전, 미세한 패턴 변화를 통해 미리 탐지하는 경우이다. 이는 조기 대응 측면에서 가치가 높으므로 평가에 반영되어야 한다.
2. **지연 탐지(Delayed Detection):** 이상치가 발생한 후 어느 정도 시간이 지나 탐지되는 경우이다. 완벽한 일치는 아니더라도 모델의 탐지 능력을 반영하므로 고려될 필요가 있다.
3. **시작 응답 시간(Onset Response Time):** 이상치 시작 시점과 탐지 시점이 얼마나 가까운가에 대한 문제이다. 즉각적인 대응이 필요한 도메인에서 매우 중요하다.
4. **예측 커버리지(Coverage level of Predictions):** 예측 구간이 실제 이상치 구간을 얼마나 많이 포괄하는지에 대한 문제이다.

기존의 Precision, Recall과 같은 지표는 데이터가 독립 동일 분포(iid)라고 가정하여 점 기반으로 평가하므로 이러한 시간적 특성을 포착하지 못한다. 특히 널리 쓰이는 Point Adjusted F1 Score(PA-F1)는 이상치 구간 내 단 한 점만 탐지해도 구간 전체를 탐지한 것으로 간주하여 성능을 지나치게 낙관적으로 과대평가하는 경향이 있다. VUS-ROC/PR과 같은 지표는 임계값(threshold) 설정에서 자유롭지만, 시간적 순서를 무시하거나 레이블을 변형하여 비현실적인 점수를 산출하는 한계가 있다.

따라서 본 논문의 목표는 조기/지연 탐지, 응답 시간, 커버리지를 모두 고려하면서도 임계값 설정의 주관성을 배제한 새로운 평가 지표인 **PATE (Proximity-Aware Time series anomaly Evaluation)**를 제안하는 것이다.

## ✨ Key Contributions

PATE의 핵심 아이디어는 이상치 구간 주변에 **버퍼 존(Buffer Zones)**을 설정하고, 탐지 지점과 실제 이상치 구간 사이의 **근접도 기반 가중치(Proximity-based weighting)**를 적용하는 것이다.

- **근접도 기반 가중치 부여:** 탐지 결과가 실제 이상치와 얼마나 가까운지에 따라 가중치를 다르게 부여함으로써, 단순히 '맞았다/틀렸다'가 아닌 '얼마나 유의미하게 탐지했는가'를 정량화한다.
- **임계값 독립적 평가:** 다양한 임계값 범위에 대해 가중 Precision과 Recall을 계산하고, 이를 통해 weighted AUC-PR을 도출함으로써 임계값 설정으로 인한 주관성을 제거한다.
- **버퍼 존의 유연성:** 사전 버퍼(Pre-buffer)와 사후 버퍼(Post-buffer) 크기를 조정하며 여러 조합에 대해 평균 성능을 측정함으로써, 다양한 응용 분야의 특성을 반영할 수 있도록 설계되었다.

## 📎 Related Works

논문에서는 시계열 특성을 반영하려는 기존의 다양한 시도들을 소개한다.

- **R-based, TS-Aware, ETS-Aware:** 구간 기반의 Precision/Recall을 도입하여 순차적 적응성(Sequential Adaptability)을 높이려 했다. 하지만 여전히 임계값 설정이 필요하며, 조기 탐지나 응답 시간 등의 세부 요소를 완전히 포착하지 못한다.
- **Affiliation Metric:** 예측 구간과 실제 구간 사이의 거리를 측정하여 근접성을 평가하지만, 조기/지연 탐지의 세밀한 차이를 구분하는 데 한계가 있다.
- **Point Adjusted F1 Score (PA-F1):** 실무에서 많이 쓰이지만, 무작위 점수(Random Score)가 SOTA 모델보다 높게 나올 수 있을 정도로 성능을 과대평가한다는 비판을 받는다.
- **VUS-ROC/PR:** 임계값 없이 평가 가능하지만, 실제 레이블을 0과 1 사이의 값으로 변경함으로써 최대 점수 1에 도달할 수 없게 만들며 시간적 선후 관계를 충분히 고려하지 않는다.

PATE는 이러한 지표들과 달리 **순차적 적응성, 조기 탐지, 지연 탐지, 응답 시간, 커버리지, 임계값 독립성**이라는 6가지 핵심 요소를 모두 충족하는 것을 목표로 한다.

## 🛠️ Methodology

### 1. 이벤트 범주화 (Categorizing the Events)

PATE는 예측 이벤트 $\mathcal{P}$와 실제 이상치 이벤트 $\mathcal{A}$의 시간적 관계에 따라 다음과 같이 범주화한다.

#### 예측 이벤트 ($\mathcal{P}$) 기준

- **True-Detection:** 예측 구간이 실제 이상치 구간과 겹치는 부분이다.
- **Post-Buffer Detection:** 실제 이상치 구간 직후의 버퍼 존($d$) 내에 위치한 구간으로, 지연 탐지 능력을 나타낸다.
- **Pre-Buffer Detection:** 실제 이상치 구간 직전의 버퍼 존($e$) 내에 위치한 구간으로, 조기 탐지 능력을 나타낸다. 단, 이후에 실제 이상치를 탐지(True-Detection)했을 때만 유효하며, 그렇지 않으면 False Positive(Outside)로 간주한다.
- **Outside:** 이상치 구간 및 버퍼 존 외부에 위치한 구간으로, 명백한 오탐(False Positive)이다.

#### 이상치 이벤트 ($\mathcal{A}$) 기준

- **Total Missed Anomalies:** 이상치 구간 전체가 탐지되지 않은 경우(False Negative)이다.
- **Partial Missed Anomalies:** 일부만 탐지되고 일부는 누락된 경우이다.

### 2. 가중치 부여 프로세스 (Weighting Process)

각 시점 $t$에 대해 True Positive($w_{TP}$), False Positive($w_{FP}$), False Negative($w_{FN}$) 가중치를 부여한다.

- **True-Detection:** $w_{TP}(t) = 1$.
- **Post-Buffer:** 이상치 구간과의 거리에 따라 가중치가 감소한다.
  $$w_{TP}(t) = 1 - \frac{\sum_{y=i_k}^{n_k} |t-y|}{\sum_{y=i_k}^{n_k} |(n_k+d)-y|}$$
  여기서 $w_{FP}(t) = 1 - w_{TP}(t)$ 로 정의된다.
- **Pre-Buffer:** 다가올 이상치 구간과의 거리에 따라 가중치가 부여된다.
  $$w_{TP}(t) = 1 - \frac{\sum_{y=i_k}^{n_k} |y-t|}{\sum_{y=i_k}^{n_k} |(i_k-e)-y|}$$
  마찬가지로 $w_{FP}(t) = 1 - w_{TP}(t)$ 이다.
- **Total Missed:** $w_{FN}(t) = 1$.
- **Partial Missed:** 이상치 시작 시점 근처에서 누락될수록 더 큰 페널티를 부여하여 응답 시간을 반영한다.
  $$w_{FN}(t) = \begin{cases} 1 & \text{if } t \le i_k + \ell \\ 1 - \frac{\sum_{y=i_k}^{i_k+\ell} |t-y|}{\sum_{y=i_k}^{n_k} |n_k-y|} & \text{otherwise} \end{cases}$$
  여기서 $\ell$은 예측이 이상치를 커버한 비율(fraction of coverage)이다.

### 3. 최종 PATE 점수 산출

임계값 $\tau$에 따른 가중 Precision과 Recall을 계산한다.

$$\text{Precision}_{e,d}(\tau) = \frac{\sum_{t=1}^T w_{TP}(t)}{\sum_{t=1}^T w_{TP}(t) + \sum_{t=1}^T w_{FP}(t)}$$
$$\text{Recall}_{e,d}(\tau) = \frac{\sum_{t=1}^T w_{TP}(t)}{\sum_{t=1}^T w_{TP}(t) + \sum_{t=1}^T w_{FN}(t)}$$

각 버퍼 조합 $(e, d)$에 대해 AUC-PR을 계산하고, 가능한 모든 버퍼 조합의 집합 $\mathcal{E}$와 $\mathcal{D}$에 대해 평균을 내어 최종 PATE 점수를 구한다.
$$\text{PATE} = \frac{1}{|\mathcal{E}| \times |\mathcal{D}|} \sum_{e \in \mathcal{E}, d \in \mathcal{D}} \text{AUC-PR}_{e,d}$$

## 📊 Results

### 1. 합성 데이터 실험

다양한 탐지 시나리오($S_1 \sim S_{10}$)를 통해 PATE가 temporal proximity, duration, coverage, response timing을 효과적으로 구분함을 보였다. 예를 들어, 운 좋게 맞춘 무작위 탐지($S_1$)에는 낮은 점수를 주고, 실제 이상치를 일부 포함하며 조기에 탐지한 경우($S_2$)에는 높은 점수를 부여했다. VUS-ROC/PR 등은 이러한 세부 시나리오를 제대로 구분하지 못했다.

### 2. 실제 데이터 실험 (Weather Temperature, ECG)

Perfect Model, MVN, LOF, AE 및 Random Score를 비교했다. PATE는 Perfect Model을 가장 높게, Random Score를 가장 낮게 평가하여 변별력이 뛰어남을 입증했다. 반면 VUS-ROC와 AUC-ROC는 성능이 낮은 모델(Model 2)의 점수를 과도하게 높게 측정하는 경향을 보였다.

### 3. SOTA 모델 재평가

SMD, MSL, SWaT, PSM 데이터셋에서 DCdetector, AnomalyTrans, USAD, LSTM, Transformer 모델을 평가했다.

- **PA-F1의 기만성:** AnomalyTrans와 DCdetector는 PA-F1에서는 매우 높은 점수(0.91, 0.87 등)를 기록했으나, PATE에서는 매우 낮은 점수(0.06, 0.07 등)를 기록했다. 시각화 결과, 이 모델들은 실제 구간과 일치하지 않는 '뾰족한(peaky)' 탐지 결과만을 내놓고 있었음이 드러났다.
- **모델 순위의 변화:** PA-F1 기준으로는 DCdetector $\rightarrow$ AnomalyTrans 순으로 우수했으나, PATE 기준으로는 **Transformer $\rightarrow$ LSTM $\rightarrow$ USAD** 순으로 순위가 완전히 뒤바뀌었다. 이는 기존 SOTA 모델들이 실제 탐지 성능보다는 지표의 허점을 이용해 높은 점수를 얻었을 가능성을 시사한다.

## 🧠 Insights & Discussion

**강점 및 기여:**
PATE는 시계열 이상치 탐지 평가에서 '진보의 환상(illusion of progress)'을 제거했다. 특히 PA-F1과 같은 지표가 모델의 실제 효용성을 얼마나 심각하게 과대평가할 수 있는지를 정량적으로 보여주었으며, 모델이 단순히 점을 맞추는 것이 아니라 시간적 맥락에서 유의미하게 탐지해야 함을 강제한다.

**한계 및 논의:**

- **버퍼 사이즈 설정:** 버퍼 사이즈 $e, d$의 범위 설정이 필요하며, 이는 전문가의 지식이나 도메인 특성에 따라 달라질 수 있다. 다만 논문의 ablation study를 통해 버퍼 사이즈 변화에도 모델 간의 상대적 순위는 일정하게 유지됨(Robustness)을 확인했다.
- **계산 복잡도:** 임계값과 버퍼 조합을 모두 순회하므로 단순 지표보다 계산 시간이 더 걸리지만, 실험 결과 데이터 길이 $T$에 대해 선형적으로 증가하며 실제 사용 시 수 초 내에 계산이 가능하여 실용적인 수준임을 보였다.

**비판적 해석:**
본 연구는 최신 딥러닝 기반 이상치 탐지 모델들이 실제로는 단순한 재구성 기반 모델(LSTM, Transformer)보다 시간적 일관성 면에서 떨어질 수 있음을 시사한다. 이는 향후 연구 방향이 단순한 지표 최적화가 아닌, 시간적 근접성과 커버리지를 높이는 방향으로 전환되어야 함을 의미한다.

## 📌 TL;DR

본 논문은 시계열 이상치 탐지 평가의 고질적인 문제인 **'성능 과대평가'**와 **'시간적 특성 무시'**를 해결하기 위해, 버퍼 존과 근접도 가중치를 도입한 **PATE** 지표를 제안한다. PATE는 조기/지연 탐지와 응답 시간을 정밀하게 평가하며, 기존 SOTA 모델들이 PA-F1 지표 덕분에 실제보다 높게 평가되었다는 사실을 밝혀냈다. 이 연구는 향후 더 실용적이고 신뢰할 수 있는 이상치 탐지 모델 개발을 위한 새로운 평가 표준을 제시한다.
