# Less is More: Surgical Phase Recognition from Timestamp Supervision

Xinpeng Ding, Xinjianian Yan, Zixun Wang, Wei Zhao, Jian Zhuang, Xiaowei Xu and Xiaomeng Li (2021)

## 🧩 Problem to Solve

본 논문은 컴퓨터 보조 수술 시스템의 핵심 요소인 **Surgical Phase Recognition**(수술 단계 인식)에서 발생하는 데이터 어노테이션의 비효율성 문제를 해결하고자 한다.

기존의 수술 단계 인식 모델들은 대부분 모든 프레임에 대해 정답이 지정된 **Full Annotations**를 필요로 한다. 하지만 전문 외과의가 수술 비디오를 반복 시청하며 각 단계의 정확한 시작 시간과 종료 시간을 찾는 작업은 매우 많은 시간과 비용이 소모된다. 또한, 수술 단계 간의 경계(boundary)는 매우 모호하여 외과의마다 어노테이션 결과가 일관되지 않은 주관적인 특성을 가진다.

따라서 본 논문의 목표는 각 수술 단계 내에서 단 하나의 타임스탬프(timestamp)만 지정하는 **Timestamp Supervision**을 도입하여, 어노테이션 비용을 획기적으로 줄이면서도 Full Supervision에 근접하거나 이를 능가하는 인식 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 비디오의 특성, 즉 "각 단계는 연속된 프레임으로 구성된 긴 이벤트"라는 점을 활용하는 것이다. 이를 위해 다음과 같은 핵심 기여를 제시한다.

1. **Timestamp Supervision 도입**: 단계별 시작/종료 시간이 아닌, 단 하나의 타임스탬프만 레이블링 하는 효율적인 어노테이션 방식을 제안한다.
2. **Uncertainty-Aware Temporal Diffusion (UATD)**: 단일 타임스탬프 레이블을 주변의 불확실성이 낮은(즉, 신뢰도가 높은) 프레임으로 확산시켜 신뢰할 수 있는 **Pseudo Labels**를 생성하는 모듈을 제안한다.
3. **Loop Training (LP)**: 긴 수술 비디오의 메모리 문제와 포지티브/네거티브 샘플의 불균형 문제를 해결하기 위해 공간적 특징 추출기(Spatial feature extractor)와 시간적 특징 추출기(Temporal feature extractor)를 독립적이고 반복적으로 최적화하는 학습 전략을 제안한다.
4. **"Less is More" 통찰 제시**: 모호한 경계 프레임을 포함한 전체 레이블보다, 적지만 판별력이 높은(discriminative) 의사 레이블을 사용하는 것이 더 높은 성능을 낼 수 있음을 입증하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **Fully-supervised Learning**: 모든 프레임에 레이블을 부여하는 방식으로 높은 성능을 보이지만, 어노테이션 비용이 극심하게 높고 경계의 모호성으로 인해 노이즈가 포함될 가능성이 크다.
- **Semi-supervised Learning**: 일부 비디오만 풀 레이블링하고 나머지는 레이블 없이 학습하는 방식(예: SurgSSL, LRTD)이다. 하지만 여전히 일부 비디오에 대해서는 막대한 어노테이션 비용이 발생한다.
- **Weakly Supervised Action Segmentation**: 타임스탬프 기반의 학습이 제안된 바 있으나(Li et al.), 이는 주로 에너지 함수를 이용해 액션 변화를 감지하는 방식을 사용한다. 수술 비디오는 경계가 매우 모호하기 때문에 이러한 방식은 노이즈가 많은 의사 레이블을 생성하여 성능을 저하시킨다.

### 차별점

본 논문은 단순한 액션 변화 감지가 아니라, **Uncertainty Estimation**을 통해 신뢰도가 높은 프레임만을 선택적으로 확산(diffusion)시킴으로써 모호한 경계 영역의 노이즈를 배제하고 신뢰할 수 있는 의사 레이블만을 확보한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인

시스템은 비디오 입력 $\mathbf{X}$를 받아 공간적 특징 추출기 $f(\cdot)$와 시간적 특징 추출기 $g(\cdot)$를 통해 각 프레임의 단계 레이블을 예측한다. 학습 과정은 `UATD`를 통한 의사 레이블 생성과 `Loop Training`을 통한 모델 최적화가 반복되는 구조이다.

### 2. Uncertainty-Aware Temporal Diffusion (UATD)

단일 타임스탬프 $Y_{ts}$로부터 신뢰할 수 있는 의사 레이블 $Y$를 생성하는 과정이다.

- **Uncertainty Estimation**: Monte Carlo Dropout을 사용하여 각 프레임의 불확실성을 측정한다. 입력 $z$에 대해 $K$번의 서로 다른 드롭아웃을 적용하여 확률 분포 $P = \{p^k = o(z)\}_{k=1}^K$를 얻는다.
  - 평균 확률 $\mu(P)$를 통해 클래스 $c = \text{argmax} \mu(P)$를 결정한다.
  - 해당 클래스에 대한 표준편차 $\sigma(P_c)$를 통해 불확실성 점수 $u$를 계산한다:
    $$u = \sigma(P_c)$$
  - $u$가 높을수록 모델의 예측 신뢰도가 낮음을 의미한다.
- **Temporal Diffusion**: 타임스탬프로 지정된 프레임을 앵커(anchor)로 삼아 양옆의 인접 프레임으로 레이블을 확산시킨다. 특정 프레임이 의사 레이블로 채택되기 위해서는 다음 두 조건을 만족해야 한다:
  1. 불확실성 점수 $u$가 임계값 $\tau$보다 낮아야 한다 ($u < \tau$).
  2. 예측된 클래스 레이블이 인접한 앵커 타임스탬프의 레이블과 일치해야 한다.

### 3. Loop Training (LP)

메모리 효율성과 샘플 불균형 문제를 해결하기 위해 공간 모델과 시간 모델을 분리하여 학습한다.

- **공간 특징 추출기 최적화**: 레이블이 있는 프레임만을 샘플링하여 Cross-Entropy Loss ($L_{ce}$)로 최적화한다.
  $$L_{ce} = -\frac{1}{T} \sum_{t=1, y_t \neq 0}^{T} y_t \log(\hat{y}_t)$$
- **시간 특징 추출기 최적화**: 공간 추출기에서 추출된 특징 $F$를 입력으로 받으며, $L_{ce}$와 더불어 프레임 간의 부드러운 전이를 유도하는 **Smoothing Loss** ($L_{smooth}$)를 함께 사용한다.
  - $L_{smooth}$는 인접 프레임 간 로그 확률의 차이를 최소화하는 Truncated Mean Squared Error 방식을 사용한다:
    $$\Delta_{t,c} = |\log(\hat{y}_{t,c}) - \log(\hat{y}_{t-1,c})|$$
    $$\tilde{\Delta}_{t,c}^2 = \begin{cases} \Delta_{t,c}^2, & \Delta_{t,c} < \gamma \\ \gamma, & \text{otherwise} \end{cases}$$
    $$L_{smooth} = \frac{1}{TC} \sum_{t,c} \tilde{\Delta}_{t,c}^2$$
  - 최종 손실 함수: $L = L_{ce} + \lambda L_{smooth}$ (단, $\lambda = 0.015$).

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80, M2CAI16
- **평가 지표**: Accuracy (AC), Precision (PR), Recall (RE), Jaccard Index (JA)
- **기준선(Baseline)**: Full Supervision 모델들(TMRNet, Trans-SVNet, TCN 등) 및 Semi-supervised 모델(SurgSSL, LRTD)

### 주요 결과

- **어노테이션 비용 절감**: 외과의를 통한 실험 결과, Timestamp annotation은 Full annotation 대비 평균 **74%의 시간 비용을 절감**하였다 (Full: 562.83s/video $\rightarrow$ Timestamp: 148.53s/video).
- **성능**: TCN 백본을 사용했을 때, 26%의 어노테이션 비용만으로 Cholec80 데이터셋에서 **91.9%의 정확도**를 달성하여, Full Supervision (91.1%)과 경쟁 가능한 수준임을 보였다.
- **SOTA 비교**: 제안 방법은 기존의 Semi-supervised 방법(SurgSSL 등)보다 적은 비용으로 더 높은 성능을 기록하였다.
- **강건성**: 타임스탬프를 무작위로 선택하여 10번 반복 실험한 결과, 성능 편차가 매우 적어 타임스탬프 선택 위치에 대해 강건함을 확인하였다.

## 🧠 Insights & Discussion

### 1. "Less is More"의 의미

본 논문의 가장 흥미로운 발견은 **모든 프레임을 레이블링 하는 것보다, 신뢰할 수 있는 프레임만 레이블링 하는 것이 성능이 더 좋다**는 점이다. 실제 수술 비디오의 경계 영역은 매우 모호하여, 이 영역을 억지로 레이블링 하여 학습시키면 오히려 모델에 노이즈로 작용한다. UATD를 통해 이러한 모호한 영역을 마스킹(masking)한 "Clean Ground-Truth"로 학습시킨 결과, 일반적인 Ground-Truth보다 성능이 향상되었다.

### 2. 외과의의 어노테이션 경향

분석 결과, 외과의들은 주로 단계의 **중간 부분(middle of phases)**을 타임스탬프로 지정하는 경향이 있다. 이는 중간 부분이 해당 단계를 가장 잘 대표하는 판별력 있는 프레임이기 때문이며, 결과적으로 UATD의 확산 기반 학습이 효율적으로 작동할 수 있는 기반이 된다.

### 3. 한계 및 향후 과제

- **전제 조건**: 본 방법은 수술 워크플로우가 급격한 변화 없이 부드럽게 진행된다는 가정을 전제로 한다. 단계 내부에서 불연속적인 변화가 발생하는 경우 성능이 저하될 수 있다.
- **학습 시간**: 모델을 처음부터 여러 번 반복해서 학습시켜야 하므로 학습 시간이 다소 오래 걸리는 단점이 있다.

## 📌 TL;DR

본 논문은 수술 단계 인식에서 막대한 비용이 드는 풀 레이블링 대신, **단계당 단 하나의 타임스탬프만 사용하는 효율적인 학습 체계**를 제안한다. **UATD**를 통해 불확실성이 낮은 프레임으로 레이블을 확산시켜 고품질의 의사 레이블을 생성하고, **Loop Training**으로 공간/시간 모델을 최적화한다. 실험 결과, 어노테이션 비용을 74% 줄이면서도 Full Supervision에 육박하는 성능을 보였으며, 특히 **모호한 경계 데이터를 제거하는 것이 성능 향상에 도움이 된다는 "Less is More"**의 통찰을 제시하였다. 이 연구는 향후 의료 영상 분석 분야에서 레이블 효율적인(label-efficient) 학습 방향을 제시하는 중요한 역할을 할 것으로 보인다.
