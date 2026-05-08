# Distribution-aware Fairness Learning in Medical Image Segmentation From A Control-Theoretic Perspective

Yujin Oh, Pengfei Jin, Sangjoon Park, Sekeun Kim, Siyeop Yoon, Kyungsang Kim, Jin Sung Kim, Xiang Li, Quanzheng Li (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 인구통계학적 속성(나이, 성별, 인종 등)과 임상적 요인(질환의 중증도 등)으로 인한 데이터 불균형은 모델의 편향(Bias)을 야기한다. 딥러닝 모델은 데이터 기반의 최적화 과정을 거치기 때문에, 다수 그룹의 패턴에 과적합(Overfitting)되고 소수 그룹의 특성을 충분히 학습하지 못하는 경향이 있으며, 이는 결국 특정 그룹에 대한 성능 저하라는 불평등한 결과로 이어진다.

기존의 공정성 학습(Fairness Learning) 연구들은 주로 인구통계학적 속성에만 집중했으며, 질병의 진행 단계나 중증도와 같은 임상적 요인이 주는 영향은 상대적으로 간과해 왔다. 본 논문의 목표는 이러한 인구통계학적 및 임상적 요인 모두를 고려하여, 데이터 분포의 불균형으로 인한 편향을 완화하고 다양한 하위 그룹 전체에서 강건하고 공평한 성능을 달성하는 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **제어 이론(Control Theory)**의 관점에서 Mixture of Experts (MoE) 구조를 재해석하고, 이를 확장하여 **Distribution-aware Mixture of Experts (dMoE)**를 설계한 것이다.

단순히 모델의 용량을 늘리는 것이 아니라, 입력 데이터의 분포 특성(인구통계학적/임상적 속성)을 외부 제어 신호로 활용하여 최적의 파라미터 세트(Expert)를 선택하는 **모드 전환 제어(Mode-switching Control)** 메커니즘을 도입하였다. 이를 통해 모델이 서로 다른 데이터 분포에 적응적으로 대응하게 함으로써, 소수 그룹에서도 높은 성능을 유지하는 공정성을 확보하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계

1. **생성 모델 기반 증강(Generative approaches):** 불균형한 분포에서 다양한 샘플을 생성하여 편향을 줄이려 하지만, 3D 전신 CT와 같은 고차원 의료 영상에서는 계산 비용이 매우 크고 고품질 샘플 생성이 어렵다는 한계가 있다.
2. **손실 함수 수정(Loss function modifications):** Distributionally Robust Optimization (DRO)나 Fair Error-Bound Scaling (FEBS) 등이 제안되었다. 그러나 이러한 방법들은 학습 배치(Batch) 내의 데이터 분포에 민감하며, 메모리 제한으로 인해 배치 사이즈를 크게 가져가기 어려운 3D 의료 영상 분할 작업에 적용하기 어렵다.
3. **기존 MoE 연구:** 주로 지속 학습(Continual Learning)이나 다중 모달 통합에 사용되었으나, 서로 다른 데이터 분포가 타겟 분포로 어떻게 적응하는지에 대한 이론적 분석과 공정성 학습으로의 응용은 부족했다.

### 차별점

dMoE는 단순한 데이터 증강이나 손실 함수 수정을 넘어, 네트워크 아키텍처 자체를 제어 시스템으로 해석하여 속성 플래그(Attribute flag)에 따라 동적으로 파라미터를 선택하는 구조를 가진다. 이는 계산 효율성을 유지하면서도 임상적/인구통계학적 요인을 명시적으로 반영할 수 있다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

dMoE는 기존의 신경망 백본(Transformer 또는 CNN) 내에 통합되는 모듈이다. 중간 단계의 이미지 임베딩과 해당 환자의 속성 정보($attn$)를 입력으로 받아, 여러 전문가 네트워크(Experts) 중 가장 적합한 전문가들을 선택하여 출력을 생성한다.

### 주요 구성 요소 및 작동 원리

1. **Experts ($\mathcal{E}$):** 얕은 다층 퍼셉트론(MLP)으로 구성된 $n$개의 전문가 네트워크 집합이다. $\mathcal{E} = \{\text{Expert}_1, \text{Expert}_2, \dots, \text{Expert}_n\}$.
2. **Distribution-aware Router ($K^{attn}$):** 입력 임베딩과 속성 플래그($attn$)를 기반으로 어떤 전문가를 활성화할지 결정하는 게이팅 네트워크이다. 속성별로 서로 다른 라우터를 가짐으로써 분포 특성에 맞는 파라미터 선택이 가능하게 한다.
3. **Noisy Top-K Gating:** 계산 효율성을 위해 상위 $k$개의 전문가만 선택하며, 학습의 안정성과 탐색을 위해 가우시안 노이즈를 추가한다.

### 주요 방정식

dMoE 층의 최종 출력 $\bar{h}_l$은 공유 경로(Shared path)와 전문가 경로의 가중 합으로 계산된다.

$$\bar{h}_l = \tilde{h}_l + \sum_{i=1}^{k} K^{attn}_i(\tilde{h}_l) \cdot \mathcal{E}_i(\tilde{h}_l)$$

여기서 $K^{attn}$은 다음과 같이 계산되는 희소 가중치 행렬이다.

$$K^{attn}(x) = \text{Softmax}(\text{KeepTop-}k(R(x), k))$$
$$R(x)_i = (x^\top \cdot W)_i + \mathcal{N}(0,1) \cdot \text{Softplus}((x^\top \cdot W_{\text{noise}})_i)$$

최종 네트워크는 일반적인 분할 손실 함수인 교차 엔트로피(Cross-entropy)를 사용하여 최적화된다.

$$\min_{\mathcal{N}} \mathcal{L}(\hat{y}, y) = -\mathbb{E}_{x \sim \mathcal{P}} \left[ \sum y_i \log p(\hat{y}_i) \right]$$

### 제어 이론적 해석 (Control-Theoretic Perspective)

본 논문은 신경망의 층별 계산을 다음과 같은 동적 시스템으로 해석한다.
$$h_{l+1} = h_l + f(h_l, \theta_l)$$

1. **Non-feedback Control:** 일반적인 피드포워드 네트워크는 입력에 관계없이 고정된 전략을 사용하는 비피드백 제어에 해당한다.
2. **Feedback Control (MoE):** MoE는 현재의 은닉 상태 $h_t$를 기반으로 파라미터를 선택하므로, 상태에 따라 제어 입력을 조정하는 피드백 제어(Closed-loop) 시스템으로 볼 수 있다.
3. **Mode-switching Control (dMoE):** dMoE는 여기에 외부 환경 변수(속성 플래그 $attn$)를 추가하여, 환경에 따라 제어 모드를 전환하는 **모드 전환 제어** 시스템을 구현한 것이다.

## 📊 Results

### 실험 설정

- **데이터셋:**
  - **Harvard-FairSeg (2D):** 망막 영상 데이터셋 (속성: 인종 - Asian, Black, White).
  - **HAM10000 (2D):** 피부 병변 데이터셋 (속성: 나이 - 20세 단위 5개 그룹).
  - **Radiotherapy Target Dataset (3D):** 전립선암 CT 데이터셋 (속성: T-stage - T1, T2, T3, T4).
- **백본:** TransUNet (2D), 3D Residual U-Net (3D).
- **지표:** Dice Similarity Coefficient, IoU, 그리고 공정성을 측정하기 위한 **ESSP (Equity-Scaled Segmentation Performance)**.

### 주요 결과

1. **2D 분할 성능:** Harvard-FairSeg에서 dMoE는 ES-Dice 및 ES-IoU에서 SOTA 성능을 달성하였다. 특히 소수 그룹인 Black 그룹의 Dice score를 FairDiff(0.743)나 FEBS(0.733)보다 높은 **0.776**까지 끌어올렸다.
2. **3D 분할 성능:** 전립선암 데이터셋에서 dMoE는 데이터 수가 매우 적은 T1, T4 단계에서 탁월한 성능 향상을 보였다. 특히 T4 그룹의 Dice score는 **0.778**로, FEBS(0.685)나 일반 MoE(0.708)를 크게 상회하였다.
3. **효율성:** dMoE는 속성별로 별도의 네트워크를 학습시켜 앙상블 하는 방식보다 계산 비용이 훨씬 적으면서도(GFlops 기준), 성능은 더 뛰어남을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 연구는 딥러닝 아키텍처 설계를 제어 이론이라는 수학적 프레임워크와 연결함으로써, 단순한 경험적 설계가 아닌 이론적 근거를 바탕으로 공정성 문제를 해결하였다. 특히 인구통계학적 요인뿐 아니라 임상적 요인(T-stage 등)을 속성으로 통합하여 실제 의료 현장에서 발생할 수 있는 다양한 편향을 효과적으로 완화했다는 점이 고무적이다.

### 한계 및 논의 사항

1. **태스크별 성능 편차:** 데이터셋과 태스크의 특성에 따라 성능 향상의 폭이 다르게 나타났으며, 최적의 dMoE 설정(위치, 파라미터 공유 여부 등)이 아키텍처마다 상이했다. 이는 태스크 범용적인 모듈 설계가 여전히 과제로 남아있음을 시사한다.
2. **단일 속성 분석의 한계:** 본 논문은 한 번에 하나의 속성(예: 인종 혹은 나이)만을 다루었다. 실제 환자는 여러 속성이 복합적으로 작용(예: 고령의 특정 인종 환자)하므로, 향후 계층적 dMoE(Hierarchical dMoE) 구조를 통해 다중 속성 불균형을 해결할 필요가 있다.
3. **최적화 알고리즘:** 현재는 역전파(Back-propagation)를 통해 파라미터를 학습하지만, 제어 이론의 고급 수치 해석 방법을 도입하여 공정성 학습을 위한 새로운 최적화 알고리즘을 개발할 가능성이 열려 있다.

## 📌 TL;DR

본 논문은 의료 영상 분할의 데이터 불균형 문제를 해결하기 위해 **제어 이론의 모드 전환 제어(Mode-switching Control)** 개념을 도입한 **dMoE(Distribution-aware Mixture of Experts)** 프레임워크를 제안한다. dMoE는 인구통계학적 및 임상적 속성을 라우팅 신호로 활용하여 데이터 분포에 따라 최적의 전문가 네트워크를 선택함으로써, 소수 그룹의 성능을 획기적으로 개선하여 공정성을 확보한다. 2D 및 3D 의료 영상 데이터셋 모두에서 SOTA 수준의 공정성과 성능을 입증하였으며, 이는 향후 다양한 임상 환경에서 범용적으로 적용 가능한 의료 AI 시스템 구축에 기여할 것으로 기대된다.
