# Conformal forecasting for surgical instrument trajectory

Sara Sangalli, Gary Sarwin, Ertunc Erdil, Alessandro Carretta, Victor Staartjes, Carlo Serra, and Ender Konukoglu (2025)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 내시경 수술 시 수술 도구의 궤적을 예측할 때 발생하는 불확실성을 신뢰할 수 있는 방법으로 정량화하는 것이다. 수술 도구의 미래 위치를 예측하는 것은 수술 자동화 및 보조 시스템 구축에 필수적이지만, 수술은 안전이 매우 중요한(safety-critical) 영역이므로 단순한 예측값뿐만 아니라 그 예측이 얼마나 정확한지에 대한 이론적 보장이 필요하다.

기존의 딥러닝 기반 예측 모델들은 점 추정(point estimation)을 제공하거나, Temperature Scaling, Monte Carlo Dropout, Deep Ensembles와 같은 방법으로 불확실성을 추정하지만, 이는 확률적인 추측일 뿐 통계적으로 유효한 커버리지(coverage)를 보장하지 않는다. 따라서 본 논문의 목표는 Conformal Prediction(CP) 프레임워크를 적용하여, 사용자 정의 확률 $1-\alpha$로 실제 값이 포함됨을 이론적으로 보장하는 예측 구간(Prediction Intervals, PIs)을 생성하고 이를 통해 신뢰할 수 있는 불확실성 히트맵(uncertainty heatmaps)을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 주요 기여는 수술 가이드 분야에 Conformal Prediction을 적용한 첫 번째 연구라는 점에 있다. 구체적인 설계 아이디어는 다음과 같다.

1. **성분별 불확실성 정량화**: 수술 도구의 이동 벡터를 각도(angle)와 크기(magnitude)로 분리하여 각각에 대해 독립적인 예측 구간을 생성함으로써 해석력을 높였다.
2. **적응형 구간 생성**: 단순한 Split CP뿐만 아니라 Conformalized Quantile Regression(CQR)을 도입하여 데이터의 특성에 따라 구간의 크기가 변하는 적응형(adaptive) 예측 구간을 구현하였다.
3. **다중 가설 검정 보정**: 각도와 크기를 동시에 고려하는 결합 구간(joint interval)을 생성할 때 발생하는 커버리지 하락 문제를 해결하기 위해 Bonferroni, Sidak, Max-Rank 보정 기법을 적용하여 이론적 보장 수준을 회복시켰다.

## 📎 Related Works

기존의 수술 가이드 연구들은 주로 신경 항법 시스템(neuronavigation), 수술 단계 인식, 해부학적 구조 탐지 등에 집중해 왔다. 최근에는 수술 도구의 궤적을 예측하려는 시도가 있었으나, 미래 시점으로의 시간적 외삽(temporal extrapolation) 과정에서 발생하는 불확실성을 다루는 데 한계가 있었다.

불확실성 정량화를 위해 사용되었던 기존의 방법들(Temperature Scaling, MC Dropout 등)은 신경망의 소프트맥스 확률을 조정하여 유사-신뢰도(pseudo-confidence)를 제공하지만, 본 논문에서 사용하는 CP와 달리 분포에 무관하게(distribution-free) 작동하거나 수학적으로 엄격한 커버리지 보장을 제공하지 못한다는 차별점이 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

시스템은 다음과 같은 파이프라인으로 구성된다.

- **입력 및 탐지**: 내시경 영상 시퀀스 $s_t = \{x_\tau\}_{\tau=t-d}^t$를 입력받아 YOLO 모델을 통해 해부학적 구조와 수술 도구의 바운딩 박스를 탐지한다.
- **인코더**: 탐지된 정보를 Transformer Encoder에 통과시켜 16차원의 잠재 표현 $z_t$를 생성한다.
- **디코더 및 예측**: 선형 레이어로 구성된 디코더가 다음 $h$ 프레임 동안의 도구 중심점 이동 벡터 $v$를 예측한다. 이때 벡터 $v$는 위상(각도) $\angle v$와 크기 $\|v\|$로 분리되어 처리된다.

### 2. Conformal Prediction (CP) 절차

Split CP는 데이터셋을 훈련 세트와 보정 세트($D_{cal}$)로 나누어 수행한다.

- **적합도 점수(Conformity Score)**: 예측값 $\hat{y}$와 실제값 $y$ 사이의 절대 오차를 점수로 정의한다.
  $$R_{CP} = |\angle v - \angle \hat{v}|$$
- **임계값 계산**: 보정 세트에서 계산된 점수들의 $(1-\alpha)(1 + 1/|D_{cal}|)$ 번째 분위수 $Q_{1-\alpha}$를 구한다.
- **예측 구간 생성**: 새로운 테스트 샘플에 대해 다음과 같은 구간을 생성한다.
  $$\text{PI}_\alpha = [\hat{y}_{n+1} - Q_{1-\alpha}, \hat{y}_{n+1} + Q_{1-\alpha}]$$
  이 구간은 실제 값이 포함될 확률이 최소 $1-\alpha$임을 보장한다.

### 3. Conformalized Quantile Regression (CQR)

더 정밀한 적응형 구간을 위해 CQR을 사용한다.

- **분위수 회귀**: Pinball Loss를 사용하여 하위 분위수 $\hat{q}_{\alpha/2}$와 상위 분위수 $\hat{q}_{1-\alpha/2}$를 예측하는 네트워크를 학습시킨다.
  $$\text{Pinball Loss: } L_\alpha(y, \hat{y}) = \begin{cases} \alpha(y - \hat{y}) & \text{if } y - \hat{y} > 0 \\ (1-\alpha)(\hat{y} - y) & \text{otherwise} \end{cases}$$
- **적합도 점수**: 예측된 두 분위수 중 실제값과 더 먼 거리의 오차를 점수로 정의한다.
  $$R_{CQR} = \max\{\hat{q}_{\alpha/2}(s) - y, y - \hat{q}_{1-\alpha/2}(s)\}$$
- **예측 구간 생성**: 위에서 구한 $Q_{1-\alpha}$를 분위수 예측값에 가감하여 구간을 생성한다.
  $$\text{PI}_\alpha = [\hat{q}_{\alpha/2}(s_{n+1}) - Q_{1-\alpha}, \hat{q}_{1-\alpha/2}(s_{n+1}) + Q_{1-\alpha}]$$

### 4. 다중 검정 보정 (Multiple Testing Corrections)

각도와 크기라는 두 개의 변수에 대해 각각 PI를 적용하고 단순히 교집합을 취하면, 전체 결합 커버리지는 $1-\alpha$보다 낮아지게 된다. 이를 해결하기 위해 다음과 같은 보정법을 사용한다.

- **Bonferroni**: 개별 테스트의 유의 수준을 $\alpha_{corr} = \alpha/k$ ($k=2$)로 낮게 설정하여 보수적으로 접근한다.
- **Sidak**: 변수 간 독립성을 가정하여 $\alpha_{corr} = 1 - (1-\alpha)^{1/k}$로 설정한다.
- **Max-Rank**: 모든 변수의 적합도 점수를 $[0, 1]$ 범위로 스케일링한 후, 가장 높은 순위의 점수를 사용하여 구간을 구성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 144개의 뇌하수체 수술(pituitary surgery) 영상. 77개로 YOLO 학습, 57개로 예측 모델 및 CQR 헤드 학습, 10개는 테스트용으로 사용하였다.
- **평가 지표**: 실제 값이 예측 구간 내에 들어오는 비율인 경험적 커버리지(Empirical Coverage)와 구간의 너비(PI Size)를 측정하였다.
- **비교 대상**: Split CP vs CQR / 보정 없음 vs 보정 적용(Bonf, Sidak, Max-Rank).

### 주요 결과

- **CQR의 우수성**: CQR은 고정된 임계값을 사용하는 CP보다 더 좁은 구간(Smaller PI Size)을 유지하면서도 타겟 커버리지를 정확하게 달성하였다. 이는 CQR이 데이터의 분포를 적응적으로 학습하기 때문이다.
- **결합 커버리지의 붕괴 및 회복**: 보정 없이 각도와 크기의 구간을 단순히 결합했을 때, 커버리지가 타겟 대비 $25\sim30\%$ 가량 급격히 하락함을 확인하였다. 하지만 보정 기법을 적용했을 때 다시 $1-\alpha$ 수준으로 회복되었다.
- **불확실성 히트맵**: 생성된 PI를 시각화한 결과, CQR 기반의 히트맵이 실제 궤적(GT)을 더 잘 중심으로 포함하며, 보정된 구간이 실제 값의 끝점(vector head) 주변을 더 정확하게 감싸는 것을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 수술 도구 궤적 예측이라는 매우 불안정한 도메인에서 통계적 보장이 있는 불확실성 정량화가 가능함을 보여주었다.

**강점 및 통찰**:

- **모델 독립성**: CP는 post-hoc 방법이므로 기존의 어떤 예측 모델에도 적용 가능하다는 점에서 범용성이 매우 높다.
- **CQR의 효율성**: 분위수 회귀 헤드를 추가하는 것만으로도 추론 속도에 거의 영향을 주지 않으면서 실시간 수술 가이드에 적합한 적응형 구간을 얻을 수 있다.

**한계 및 논의**:

- **데이터 노이즈**: 도구의 길이(magnitude) 예측에서 CP의 변동성이 크게 나타났는데, 이는 도구가 화면 밖으로 사라지거나 나타나는 등의 실데이터 노이즈에 민감하기 때문으로 분석된다.
- **교환 가능성 가정**: CP의 전제 조건인 교환 가능성(exchangeability) 가정이 실제 수술 영상과 같은 시계열 데이터에서 완벽히 성립하는지에 대한 추가 검토가 필요하다.

## 📌 TL;DR

이 논문은 수술 도구 궤적 예측의 불확실성을 정량화하기 위해 **Conformal Prediction(CP)**과 **Conformalized Quantile Regression(CQR)**을 최초로 적용하였다. 특히 각도와 크기를 분리해 예측하고, 다중 검정 보정 기법을 통해 **이론적으로 보장된 결합 예측 구간**을 생성함으로써 신뢰할 수 있는 수술 가이드용 불확실성 히트맵을 제시하였다. 이 연구는 향후 수술 자동화 시스템에서 위험 평가 및 안전한 경로 선택을 위한 수학적 근거를 제공할 가능성이 크다.
