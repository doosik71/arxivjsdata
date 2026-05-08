# Conformal forecasting for surgical instrument trajectory

Sara Sangalli, Gary Sarwin, Ertunc Erdil, Alessandro Carretta, Victor Staartjes, Carlo Serra, and Ender Konukoglu (2025)

## 🧩 Problem to Solve

본 논문은 내시경 수술 시 수술 도구의 궤적(Trajectory)을 예측하는 과정에서 발생하는 불확실성을 정량적으로 측정하고, 이를 이론적으로 보장된 예측 구간(Prediction Interval, PI)으로 제시하는 문제를 해결하고자 한다.

수술 도구의 움직임을 예측하는 것은 수술 자동화 및 보조 시스템 구축에 있어 매우 중요하다. 그러나 수술 환경은 매우 동적이며, 미래의 움직임을 예측하는 Temporal Extrapolation 과정에서 필연적으로 불확실성이 발생한다. 특히 수술은 안전이 최우선인 Safety-critical한 영역이므로, 단순히 예측값만을 제시하는 것이 아니라 그 예측이 얼마나 신뢰할 수 있는지를 나타내는 신뢰성 있는 불확실성 정량화(Uncertainty Quantification)가 필수적이다.

따라서 본 연구의 목표는 Conformal Prediction(CP) 프레임워크를 수술 도구 궤적 예측에 적용하여, 사용자가 설정한 확률 $1-\alpha$로 실제 값이 포함됨을 보장하는 분포 독립적(Distribution-free)이고 이론적으로 유효한 예측 구간을 생성하고, 이를 시각적인 불확실성 히트맵(Uncertainty Heatmap)으로 구현하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 수술 안내(Surgical Guidance) 분야에 Conformal Prediction을 최초로 적용하여 통계적 보장(Formal Coverage Guarantees)을 갖춘 예측 구간을 구축했다는 점이다.

구체적인 설계 아이디어는 다음과 같다.

1. **성분별 불확실성 분리**: 예측된 움직임 벡터를 각도(Angle)와 크기(Magnitude)로 분리하여 각각에 대해 불확실성을 정량화함으로써 해석력을 높였다.
2. **적응형 구간 생성**: 고정된 임계값을 사용하는 Split CP 외에도, 데이터의 특성에 따라 구간의 너비가 변하는 Conformalized Quantile Regression(CQR)을 도입하여 더 정밀한 예측 구간을 생성하였다.
3. **결합 커버리지 보정**: 각도와 크기에 대해 독립적으로 생성된 구간을 단순히 결합할 경우 전체 커버리지가 하락하는 다중 가설 검정(Multiple Hypothesis Testing) 문제가 발생함을 지적하고, 이를 해결하기 위해 Bonferroni, Sidak, Max-Rank 보정법을 적용하였다.
4. **시각적 가이드 제공**: 계산된 예측 구간을 바탕으로 수술자에게 실시간으로 제공 가능한 불확실성 히트맵을 생성하는 방법론을 제시하였다.

## 📎 Related Works

논문은 수술 가이드를 위한 기존 연구들(수술 단계 인식, 해부학적 구조 검출, 워크플로우 인식 등)을 언급하며, 특히 최근의 수술 도구 궤적 예측 연구([17])를 기반으로 하고 있음을 밝힌다.

기존의 불확실성 측정 방식인 Temperature Scaling, Monte Carlo Dropout, Deep Ensembles 등은 신경망의 Softmax 확률을 조정하여 유사-신뢰 수준(Pseudo-confidence levels)을 제공하지만, 이는 통계적인 강건한 보장(Robust guarantees)이 결여되어 있다는 한계가 있다. 반면, Conformal Prediction은 모델에 관계없이(Model-agnostic) 분포에 제약을 받지 않고 이론적으로 유효한 예측 구간을 제공한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 문제 정의 및 데이터 표현

수술 도구의 움직임은 이전 프레임 대비 다음 $h$개 프레임 동안의 bounding box 중심 좌표 변화량으로 정의된다. 이 변화량을 나타내는 그라운드 트루스(GT) 벡터를 $v$, 예측 벡터를 $\hat{v}$라고 한다. 분석의 편의와 해석력을 위해 벡터 $v$를 다음 두 가지 성분으로 분리한다.

- **위상(Phase)**: $\angle v$ (양의 x축 기준 각도)
- **크기(Magnitude)**: $\|v\|$

### 2. Conformal Prediction (CP) 방법론

#### Split Conformal Prediction

데이터셋을 훈련 세트와 캘리브레이션 세트 $D_{cal} = \{(f_i, y_i)\}_{i=1}^n$으로 분리한다.

- **Conformity Score**: 예측값 $\hat{y}_i$와 실제값 $y_i$ 사이의 절대 오차를 점수로 정의한다.
  $$R_{CP, i} = |\angle v_i - \angle \hat{v}_i|$$
- **분위수 계산**: 사용자가 설정한 에러율 $\alpha$에 대해, $D_{cal}$에서 얻은 점수들의 $(1-\alpha)(1 + 1/|D_{cal}|)$ 번째 경험적 분위수 $Q_{1-\alpha}$를 계산한다.
- **예측 구간(PI)**: 새로운 테스트 샘플 $s_{n+1}$에 대한 구간은 다음과 같다.
  $$\text{PI}_\alpha(s_{n+1}) = [\angle \hat{v}_{n+1} - Q_{1-\alpha}, \angle \hat{v}_{n+1} + Q_{1-\alpha}]$$

#### Conformalized Quantile Regression (CQR)

입력 데이터에 따라 구간의 너비가 변하는 적응형 구간을 생성하기 위해 사용한다.

- **분위수 회귀**: Pinball Loss를 사용하여 하위 분위수 $\hat{q}_{\alpha/2}$와 상위 분위수 $\hat{q}_{1-\alpha/2}$를 예측하는 네트워크를 학습시킨다.
  $$\text{Loss}_{\alpha}(y, \hat{y}) = \begin{cases} \alpha(y - \hat{y}) & \text{if } y - \hat{y} > 0 \\ (1 - \alpha)(\hat{y} - y) & \text{otherwise} \end{cases}$$
- **Conformity Score**: 두 분위수 예측값 중 실제값에서 더 먼 쪽의 오차를 점수로 정의한다.
  $$R_{CQR, i} = \max\{\hat{q}_{\alpha/2}(s_i) - \angle v_i, \angle v_i - \hat{q}_{1-\alpha/2}(s_i)\}$$
- **예측 구간(PI)**: 캘리브레이션 세트에서 얻은 $Q_{1-\alpha}$를 이용하여 구간을 보정한다.
  $$\text{PI}_\alpha(s_{n+1}) = [\hat{q}_{\alpha/2}(s_{n+1}) - Q_{1-\alpha}, \hat{q}_{1-\alpha/2}(s_{n+1}) + Q_{1-\alpha}]$$

### 3. 다중 검정 보정 (Multiple Testing Corrections)

각도와 크기에 대해 각각 독립적인 $\text{PI}_\alpha$를 생성하여 결합하면, 전체 결합 커버리지는 $(1-\alpha)$보다 낮아지게 된다. 이를 해결하기 위해 다음과 같은 보정법을 적용한다.

- **Bonferroni Correction**: 개별 테스트의 유의 수준을 $\alpha_{corr} = \alpha/k$ (여기서 $k=2$)로 설정하여 보수적으로 구간을 넓힌다.
- **Sidak Correction**: 독립성을 가정하여 $\alpha_{corr} = 1 - (1-\alpha)^{1/k}$로 설정한다.
- **Max-Rank Correction**: 모든 변수의 nonconformity score의 랭킹 중 최댓값을 사용하여 구간을 구성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 144개의 뇌하수체 수술(pituitary surgery) 비디오를 사용하였으며, YOLOv7을 통해 도구와 해부학적 구조를 검출하였다.
- **모델 아키텍처**: Transformer Encoder와 Linear Decoder로 구성된 궤적 예측 모델을 사용하였다. CQR의 경우, Frozen Encoder 뒤에 4개의 FC 레이어로 구성된 분위수 회귀 헤드를 추가하여 학습시켰다.
- **지표**: 경험적 커버리지(Empirical Coverage)와 예측 구간의 너비(PI Size)를 측정하였다.

### 주요 결과

1. **CQR vs CP**: CQR이 CP보다 더 좁은 예측 구간을 유지하면서도 타겟 커버리지를 잘 달성하였다. 이는 CQR이 데이터 분포에 따라 적응적으로 구간을 조정할 수 있기 때문이다.
2. **결합 커버리지**: 보정 없이 각도와 크기 구간을 결합했을 때, 커버리지가 약 $25\sim30\%$ 급격히 하락하였다. 하지만 보정법(특히 CQR + 보정)을 적용했을 때 타겟 커버리지가 성공적으로 회복되었다.
3. **정량적 성능 (Table 1 요약)**:
   - Target Coverage 60% 기준, CQR angle의 PI Size는 $69.3^\circ$로 CP angle($78.5^\circ$)보다 정밀하였다.
   - 보정된 joint 구간에서도 CQR이 CP보다 더 좁은 구간을 형성하며 효율적인 불확실성 정량화를 수행하였다.
4. **시각적 분석**: 불확실성 히트맵에서 CQR 기반의 구간이 GT 벡터 주변에 더 밀집되어 나타나며, 보정된 구간이 보정되지 않은 구간보다 GT 벡터의 끝점을 더 정확하게 포함하는 것을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 수술 도구 궤적 예측이라는 고위험 작업에 있어, 단순한 예측값 제시를 넘어 이론적 보장이 있는 불확실성 구간을 제공할 수 있음을 입증하였다. 특히 CQR이 단순한 Split CP보다 실제 수술 데이터의 변동성에 더 유연하게 대응하여 정밀한 구간을 생성한다는 점은 실용적인 가치가 크다.

**한계점 및 논의사항:**

- **데이터 노이즈**: 수술 도구가 시야에서 사라지거나 나타나는 경우 벡터 크기에 큰 변동이 발생하며, 이로 인해 크기(length) 예측의 변동성이 각도보다 높게 나타났다.
- **교환 가능성(Exchangeability)**: CP의 기본 가정인 데이터의 교환 가능성이 실제 수술의 시계열 데이터에서 완벽하게 유지되는지에 대한 추가적인 검토가 필요하다.
- **실시간성**: CP는 post-hoc 방식이고 CQR은 가벼운 헤드만 추가되므로 연산 오버헤드가 매우 적어 실시간 시스템 적용 가능성이 높다.

## 📌 TL;DR

이 논문은 수술 도구 궤적 예측의 안전성을 높이기 위해 **Conformal Prediction(CP)** 및 **Conformalized Quantile Regression(CQR)**을 도입하여, 통계적으로 보장된 예측 구간(Prediction Interval)을 생성하는 방법론을 제안한다. 특히 각도와 크기를 분리하여 분석하고 다중 검정 보정법을 통해 결합 커버리지를 확보함으로써, 신뢰할 수 있는 **불확실성 히트맵**을 구현하였다. 이는 향후 자율 수술 시스템의 리스크 평가 및 수술 보조 시스템의 신뢰도 향상에 중요한 기초 연구가 될 것으로 보인다.
