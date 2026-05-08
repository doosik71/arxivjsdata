# Conformal forecasting for surgical instrument trajectory

Sara Sangalli, Gary Sarwin, Ertunc Erdil, Alessandro Carretta, Victor Staartjes, Carlo Serra, and Ender Konukoglu (2025)

## 🧩 Problem to Solve

본 논문은 내시경 수술에서 수술 도구의 궤적(trajectory)을 예측할 때 발생하는 불확실성을 정량적으로 측정하고 이를 신뢰할 수 있는 구간으로 제시하는 문제를 해결하고자 한다. 수술 도구의 미래 위치를 예측하는 것은 수술 자동화 및 보조 시스템 구축에 있어 매우 중요하지만, 미래의 움직임을 예측하는 temporal extrapolation 과정에서 필연적으로 높은 불확실성이 수반된다.

특히 수술은 안전이 최우선인 safety-critical한 환경이므로, 단순한 점 예측(point prediction)보다는 예측 결과에 대한 이론적 보장이 있는 불확실성 측정(uncertainty quantification)이 필수적이다. 따라서 본 연구의 목표는 Conformal Prediction(CP) 프레임워크를 적용하여, 실제 도구의 위치가 특정 확률 $1-\alpha$로 포함될 것임을 이론적으로 보장하는 예측 구간(Prediction Intervals, PIs)을 생성하고, 이를 통해 신뢰할 수 있는 불확실성 히트맵(uncertainty heatmaps)을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 수술 가이던스 분야에 Conformal Prediction을 최초로 적용하여, 예측된 궤적의 각도(angle)와 크기(magnitude)에 대해 통계적 보장이 있는 예측 구간을 구축했다는 점이다.

주요 설계 아이디어는 다음과 같다.

- **개별 및 결합 불확실성 정량화**: 예측 벡터의 방향($\angle v$)과 크기($\|v\|$)를 분리하여 각각에 대해 불확실성을 계산하고, 이를 다시 결합하여 전체 궤적의 불확실성을 도출한다.
- **다중 가설 검정 보정(Multiple-testing corrections)**: 각도와 크기에 대해 독립적으로 생성된 예측 구간을 단순 결합할 경우 전체 커버리지(coverage)가 하락하는 문제를 확인하고, 이를 해결하기 위해 Bonferroni, Sidak, Max-Rank 보정 기법을 적용하여 결합 커버리지를 복구하였다.
- **적응형 구간 생성**: 고정된 임계값을 사용하는 standard CP 외에도, 데이터의 특성에 따라 구간의 너비가 변하는 Conformalized Quantile Regression(CQR)을 도입하여 더 정밀한 불확실성 추정을 구현하였다.

## 📎 Related Works

기존의 수술 도구 궤적 예측 연구는 주로 미래의 위치를 예측하는 모델 구조에 집중해 왔으나, 예측 결과의 신뢰도를 정량화하는 방법론에 대해서는 충분히 다루지 않았다.

불확실성 측정과 관련하여 Temperature Scaling, Monte Carlo Dropout, Deep Ensembles와 같은 기존의 딥러닝 기반 방법론들이 존재한다. 이러한 방법들은 신경망의 softmax 확률을 조정하여 pseudo-confidence level을 제공하지만, 본 논문에서 사용한 Conformal Prediction과 달리 분포에 무관(distribution-free)하며 이론적으로 유효한 커버리지를 보장하는 강건한 통계적 보장(robust guarantees)이 부족하다는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구는 기존의 궤적 예측 프레임워크를 기반으로 하며, 전체 파이프라인은 다음과 같이 구성된다.

1. **입력 및 검출**: 내시경 프레임 시퀀스 $s_t = \{x_\tau\}_{\tau=t-d}^t$를 입력받아 YOLO 모델을 통해 해부학적 구조와 수술 도구를 검출한다.
2. **특징 추출 및 예측**: Transformer Encoder를 통해 잠재 표현(latent representation) $z_t$를 생성하고, Decoder를 통해 다음 $h$ 프레임 동안의 바운딩 박스 중심 좌표 변화량을 예측한다.
3. **벡터 표현**: 예측된 움직임을 하나의 벡터 $v$로 정의하며, 이를 해석 가능하도록 위상(phase, $\angle v$)과 크기(magnitude, $\|v\|$)로 분리하여 처리한다.

### Conformal Prediction (CP) 절차

데이터셋을 훈련 세트와 캘리브레이션 세트 $\mathcal{D}_{cal} = \{(f_i, y_i)\}_{i=1}^n$으로 분리한다.

- **Conformity Score**: 예측값 $\hat{y}_i$와 실제값 $y_i$ 사이의 괴리를 측정하는 비음수 점수 $R(f, y)$를 정의한다. Split CP에서는 절대 오차 잔차(absolute error residuals)를 사용한다:
  $$R_{CP, i} = |\angle v_i - \angle \hat{v}_i|$$
- **Quantile 계산**: 사용자가 설정한 에러율 $\alpha$에 대해, $\mathcal{D}_{cal}$에서 계산된 $R_{CP, i}$들의 $(1-\alpha)(1 + 1/|D_{cal}|)$-번째 경험적 분위수 $Q_{1-\alpha}$를 구한다.
- **예측 구간 생성**: 새로운 테스트 샘플 $n+1$에 대한 예측 구간은 다음과 같이 정의되며, 이는 $\mathbb{P}(y_{n+1} \in PI_\alpha) \ge 1-\alpha$를 보장한다.
  $$PI_\alpha = [\hat{y}_{n+1} - Q_{1-\alpha}, \hat{y}_{n+1} + Q_{1-\alpha}]$$

### Conformalized Quantile Regression (CQR)

더 적응적인(adaptive) 구간을 생성하기 위해 CQR을 사용한다.

- **훈련**: Pinball loss를 사용하여 하위 분위수 $\hat{q}_{\alpha/2}$와 상위 분위수 $\hat{q}_{1-\alpha/2}$를 예측하는 회귀 네트워크를 학습시킨다. Pinball loss $L_\alpha$는 다음과 같다:
  $$L_\alpha(y, \hat{y}) = \begin{cases} \alpha(y - \hat{y}) & \text{if } y - \hat{y} > 0 \\ (1-\alpha)(\hat{y} - y) & \text{otherwise} \end{cases}$$
- **Conformity Score**: 두 분위수 예측값 중 더 큰 오차를 점수로 선택한다:
  $$R_{CQR, i} = \max\{\hat{q}_{\alpha/2}(s_i) - \angle v_i, \angle v_i - \hat{q}_{1-\alpha/2}(s_i)\}$$
- **예측 구간**: CQR의 구간은 데이터에 따라 너비가 변하며, 역시 동일한 커버리지 보장을 제공한다:
  $$PI_\alpha = [\hat{q}_{\alpha/2}(s_{n+1}) - Q_{1-\alpha}, \hat{q}_{1-\alpha/2}(s_{n+1}) + Q_{1-\alpha}]$$

### 다중 테스트 보정 (Multiple testing corrections)

각도와 크기에 대해 독립적으로 생성된 구간을 단순히 교차시키면 전체 커버리지가 $1-\alpha$보다 훨씬 낮아진다. 이를 해결하기 위해 다음 보정법을 적용한다.

- **Bonferroni correction**: 개별 테스트의 $\alpha$를 $\alpha_{corr} = \alpha/k$ (여기서 $k=2$)로 수정하여 보수적으로 접근한다.
- **Sidak correction**: 독립성을 가정하여 $\alpha_{corr} = 1 - (1-\alpha)^{1/k}$를 적용한다.
- **Max-Rank correction**: 모든 변수의 nonconformity score의 랭킹 중 최대값을 사용하여 구간을 구축한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 144개의 뇌하수체 수술(pituitary surgery) 영상.
- **평가 지표**: 경험적 커버리지(Empirical Coverage)와 예측 구간의 너비(PI Size).
- **비교 대상**: Split CP vs CQR, 그리고 보정 전/후의 결합 구간.

### 정량적 결과

- **CQR의 우수성**: 표 1의 결과에 따르면, CQR은 CP보다 대체로 더 작은 예측 구간(PI Size)을 생성하면서도 목표 커버리지를 잘 유지한다. 이는 CQR이 데이터 분포를 학습하여 적응형 구간을 생성하기 때문이다.
- **결합 구간의 커버리지 하락**: 보정 없이 각도와 크기 구간을 결합했을 때, 커버리지가 원래 목표(60~70%)에서 약 25~30% 가량 급격히 하락(약 31~45% 수준)하는 것이 확인되었다.
- **보정 효과**: Bonferroni, Sidak, Max-Rank 보정을 적용했을 때 커버리지가 다시 목표 수준으로 복구되었으며, 특히 CQR 기반의 보정된 구간이 가장 효율적인(좁은) 너비를 보였다.

### 정성적 결과 (Uncertainty Heatmaps)

- CQR 기반의 히트맵은 GT(Ground Truth) 벡터 주변에 더 집중된 형태를 보이며, CP보다 더 날카롭고(sharp) 정확한 불확실성 영역을 제시한다.
- 보정된 결합 구간 히트맵은 보정 전보다 GT 벡터의 끝점(head) 주변을 더 잘 포괄하여 실제 수술 가이던스 시스템에서 더 높은 신뢰도를 제공할 수 있음을 보여준다.

## 🧠 Insights & Discussion

본 연구는 수술 궤적 예측이라는 고위험 작업에 통계적 보장이 있는 불확실성 정량화 기법을 도입했다는 점에서 큰 의미가 있다. 특히, 단순히 딥러닝 모델의 출력값에 의존하는 것이 아니라, 캘리브레이션 세트를 통한 사후 처리(post-hoc) 방식으로 이론적 커버리지를 보장함으로써 임상의의 신뢰를 얻을 수 있는 기반을 마련하였다.

**강점 및 한계**

- **강점**: 모델 구조에 상관없이 적용 가능한 model-agnostic한 특성을 가지며, 계산 오버헤드가 거의 없어 실시간 수술 가이던스에 적합하다.
- **한계 및 가정**: 데이터의 교환 가능성(exchangeability) 가정을 전제로 한다. 그러나 실제 수술 데이터에서는 시계열적 특성이나 환자별 차이로 인해 이 가정이 깨질 수 있으며, 이는 향후 해결해야 할 과제이다. 또한, 현재의 모델은 단기 예측에 집중하고 있어 장기 예측(long-term forecasting) 시의 불확실성 확산 문제는 충분히 다루어지지 않았다.

**비판적 해석**
본 논문은 CP와 CQR의 성능 차이를 명확히 보여주었지만, 실제 수술 환경에서 "어느 정도의 커버리지($1-\alpha$)가 임상적으로 적절한가"에 대한 기준은 제시하지 않았다. 통계적 보장이 반드시 임상적 유용성으로 직결되는지는 추가적인 사용자 연구(surgeon-in-the-loop)를 통해 검증될 필요가 있다.

## 📌 TL;DR

본 논문은 수술 도구 궤적 예측의 불확실성을 정량화하기 위해 **Conformal Prediction(CP)** 및 **Conformalized Quantile Regression(CQR)**을 도입하였다. 특히 예측 벡터의 각도와 크기에 대해 개별적/결합적 예측 구간(PI)을 생성하고, 다중 테스트 보정을 통해 통계적으로 보장된 커버리지를 확보하였다. 실험 결과, **CQR이 CP보다 더 정밀하고 적응적인 구간을 제공**함이 입증되었으며, 이를 통해 생성된 **불확실성 히트맵**은 향후 안전한 수술 자동화 및 보조 시스템 구축에 핵심적인 역할을 할 것으로 기대된다.
