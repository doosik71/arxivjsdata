# AINet+: Advancing Superpixel Segmentation via Cascaded Association Implantation

Yaxiong Wang, Yunchao Wei, Yujiao Wu, Xueming Qian, Li Zhu, and Yi Yang (2021)

## 🧩 Problem to Solve

본 논문은 이미지의 픽셀들을 유사한 특성을 가진 그룹으로 묶는 Superpixel segmentation의 성능 향상을 목표로 한다. 기존의 딥러닝 기반 Superpixel segmentation 방식들은 일반적으로 이미지를 격자(grid) 형태로 나누고, 각 픽셀을 인접한 격자 세그먼트에 할당하는 방식을 취한다.

그러나 기존 방식들은 주로 제한된 수용 영역(receptive field)을 가진 Convolution 연산에 의존하며, 이는 픽셀과 격자 간의 상호작용을 명시적(explicit)이 아닌 암시적(implicit)으로만 이해하게 만든다. 이러한 한계로 인해 픽셀-격자 간의 연관성(association)을 매핑하는 과정에서 문맥 정보(contextual information)가 부족해지며, 특히 경계 영역의 정확도가 떨어지는 문제가 발생한다. 따라서 본 연구는 픽셀과 격자 간의 관계를 명시적으로 학습하여 경계 정밀도를 높이고 전체적인 세그먼트 품질을 개선하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 픽셀과 주변 격자(grid) 간의 관계를 네트워크가 직접적으로 인지할 수 있도록 하는 **Association Implantation (AI) 모듈**의 도입이다.

1.  **Association Implantation (AI) 모듈**: 픽셀의 주변에 대응하는 격자 특징(grid features)을 직접 심음(implant)으로써, 단순한 픽셀 간 상호작용이 아닌 '픽셀-격자' 수준의 문맥을 명시적으로 추출한다.
2.  **Cascaded/Hierarchical 구조 (AINet+)**: AI 모듈을 단일 층이 아닌 여러 계층(penultimate 및 last layers)에 배치하여, 거친 단계(coarse)부터 세밀한 단계(fine)까지 픽셀-슈퍼픽셀 관계를 점진적으로 정교화한다.
3.  **Boundary-perceiving Loss (BPL)**: 특징 수준에서 경계 인접 픽셀들을 효과적으로 구분하기 위해, 경계 영역의 동일 라벨 특징은 가깝게, 서로 다른 라벨 특징은 멀게 밀어내는 손실 함수를 설계하여 경계 정밀도를 높였다.

## 📎 Related Works

### 기존 연구 및 한계
- **전통적 방식**: SLIC와 같은 클러스터링 기반 방식이나 ERS와 같은 그래프 기반 방식이 존재한다. 이들은 계산 효율성이 좋으나, 수작업으로 설계된 특징(hand-craft features)에 의존하므로 학습 가능한 딥러닝 프레임워크와 통합하기 어렵다는 한계가 있다.
- **딥러닝 기반 방식**: SSN, SCN 등 U-Net 구조를 활용해 9가지 방향의 확률(association map)을 예측하는 방식이 제안되었다. 하지만 이들은 단순히 Convolution 층을 쌓아 수용 영역을 넓히는 방식에 의존하므로, 픽셀-격자 간의 명시적인 관계를 학습하기보다는 픽셀-픽셀 간의 관계를 학습하는 경향이 강하다.

### 차별점
AINet+는 단순한 Convolution의 반복이 아니라, 격자 특징을 픽셀 주변에 물리적으로 배치하는 Implantation 과정을 통해 슈퍼픽셀 세그먼테이션의 목적(픽셀을 격자에 할당)에 훨씬 더 부합하는 아키텍처를 설계했다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인
AINet+는 Encoder-Decoder 구조를 따른다.
- **Encoder**: 입력 이미지를 압축하여 격자 셀의 특징을 인코딩한 **Superpixel Embedding**을 생성한다.
- **Decoder**: 생성된 임베딩을 바탕으로 픽셀과 주변 9개 격자 간의 연관성을 나타내는 **Association Map** $Q \in \mathbb{R}^{H \times W \times 9}$를 출력한다. 이 과정에서 AI 모듈이 계층적으로 적용된다.

### 2. Association Implantation (AI) Module
AI 모듈은 픽셀 임베딩 $e_p$와 슈퍼픽셀 임베딩 $M$을 결합하여 명시적인 문맥을 생성한다.

- **작동 절차**:
    1. 픽셀 $p$를 중심으로 주변 9개의 슈퍼픽셀 임베딩 $\{\hat{m}_{tl}, \hat{m}_{t}, \dots, \hat{m}_{br}\}$을 추출한다.
    2. 이 임베딩들을 픽셀 $p$의 주변에 배치하여 $3 \times 3$ 행렬 $SP$를 구성한다. 이때 중앙에는 픽셀 임베딩 $e_p$가 위치한다.
    $$ SP = \begin{bmatrix} \hat{m}_{tl} & \hat{m}_{t} & \hat{m}_{tr} \\ \hat{m}_{l} & \hat{m}_{c} + e_p & \hat{m}_{r} \\ \hat{m}_{bl} & \hat{m}_{b} & \hat{m}_{br} \end{bmatrix} $$
    3. 구성된 $SP$ 행렬에 $3 \times 3$ Convolution을 적용하여 최종적으로 픽셀-슈퍼픽셀 문맥이 반영된 새로운 임베딩 $e'_p$를 도출한다.
    $$ e'_p = \sum_{ij} SP_{ij} \times w_{ij} + b $$

### 3. 계층적 학습 (Hierarchical Learning)
AINet+는 AI 모듈을 마지막 층뿐만 아니라 그 이전 층(penultimate layer)에도 적용한다. 이전 층에서는 더 넓은 영역을 커버하는 다운샘플링된 슈퍼픽셀 임베딩 $M_{\downarrow}$를 사용하며, 이를 통해 거친 수준에서 세밀한 수준으로 픽셀-격자 관계를 점진적으로 학습한다.

### 4. Boundary-perceiving Loss (BPL)
경계 영역의 특징 변별력을 높이기 위해 설계된 손실 함수이다.
- 경계 주변의 로컬 패치를 샘플링하여 동일한 시맨틱 라벨을 가진 픽셀 쌍은 가깝게, 다른 라벨을 가진 쌍은 멀게 학습시킨다.
- 유사도 함수 $\text{sim}(f, g) = \frac{2}{1 + \exp(\|f-g\|_1)}$를 사용하여 다음과 같이 정의된다.
$$ L_B = \frac{1}{|B|} \sum_{B \in \mathcal{B}} L_B $$
여기서 $L_B$는 동일 클래스 간의 유사도는 높이고, 타 클래스 간의 유사도는 낮추도록 유도하는 분류 기반 손실이다.

### 5. 전체 학습 목표 (Total Loss)
최종 손실 함수는 시맨틱 라벨에 대한 Cross-Entropy(CE), 위치 벡터에 대한 $L_2$ 재구성 손실, 그리고 BPL의 합으로 구성된다.
$$ L = \sum_{p} \text{CE}(l'_s(p), l_s(p)) + \lambda \|p - p'\|_2^2 + \alpha L_B $$

## 📊 Results

### 실험 설정
- **데이터셋**: BSDS500, NYUv2 (일반 이미지), ISIC-2017, ACDC (의료 이미지).
- **평가 지표**:
    - **ASA (Achievable Segmentation Accuracy)**: 슈퍼픽셀을 전처리로 사용했을 때 도달 가능한 최대 세그먼테이션 정확도.
    - **BR (Boundary Recall)** 및 **BP (Boundary Precision)**: 경계 식별 능력을 측정.
- **비교 대상**: SLIC, SNIC, LSC, SEEDS, ERS (전통적 방식) / SEAL, SCN, SSN (딥러닝 방식).

### 주요 결과
- **정량적 결과**: AINet+는 네 가지 벤치마크 데이터셋 모두에서 ASA, BR, BP 지표 모두 SOTA(State-of-the-art) 성능을 달성하였다. 특히 의료 데이터셋(ISIC-2017, ACDC)에서 기존 딥러닝 방식들보다 월등히 높은 성능 향상을 보였다.
- **정성적 결과**: 시각화 결과, AINet+로 생성된 슈퍼픽셀의 경계가 타 방법론에 비해 훨씬 더 정확하고 명확하게 객체의 윤곽을 따라 형성됨을 확인하였다.
- **추론 효율성**: SCN보다는 약간 느리지만, K-means 반복 연산이 필요한 SSN이나 복잡한 후처리가 필요한 SEAL보다 훨씬 빠른 추론 속도를 보였다.

### 다운스트림 작업 적용
- **Object Proposal Generation**: DEL 프레임워크에 AINet+ 슈퍼픽셀을 적용한 결과, 더 정확한 객체 제안 영역을 생성하여 ASA 스코어가 향상되었다.
- **Stereo Matching**: PSMNet에 AINet+를 통합하여 메모리 사용량을 줄이면서도 End-point-error (EPE)를 낮추어 성능을 개선하였다. (SceneFlow 데이터셋 기준 EPE 0.84 달성)

## 🧠 Insights & Discussion

본 논문은 슈퍼픽셀 세그먼테이션에서 '무엇을 학습해야 하는가'에 대한 근본적인 접근 방식을 제안했다. 기존의 딥러닝 방식들이 단순한 Convolution 층의 깊이를 통해 문맥을 파악하려 했다면, AINet+는 **"픽셀은 주변 격자와의 관계를 통해 정의된다"**는 슈퍼픽셀의 본질적 정의를 네트워크 구조(AI 모듈)에 직접 반영했다.

특히 의료 이미지 데이터셋에서 성능 향상이 두드러진 점은, 의료 영상의 특성상 미세한 경계 식별이 매우 중요한데, BPL과 AI 모듈이 이러한 정밀한 구조적 특징을 포착하는 데 효과적이었음을 시사한다.

다만, 격자 간격($S=16$)이 고정되어 있어, 입력 이미지의 해상도나 객체의 크기 변화에 따라 최적의 $S$ 값이 달라질 수 있다는 점은 향후 해결해야 할 과제로 보인다. 또한, BPL을 적용할 때 초기 학습 후 파인튜닝 단계에서만 사용하는 전략을 취했는데, 이는 두 손실 함수 간의 최적화 방향이 일부 충돌할 수 있음을 암시한다.

## 📌 TL;DR

본 연구는 픽셀-격자 간의 관계를 명시적으로 학습하는 **Association Implantation (AI) 모듈**과 경계 정밀도를 높이는 **Boundary-perceiving Loss (BPL)**를 제안한 **AINet+**를 개발하였다. 이를 통해 일반 및 의료 영상 데이터셋에서 SOTA 성능을 기록하였으며, 객체 제안 생성 및 스테레오 매칭과 같은 후속 작업에서도 성능 향상을 입증하였다. 이 연구는 슈퍼픽셀의 정의를 네트워크 구조에 직접 반영함으로써 딥러닝 기반 세그먼테이션의 효율성과 정확성을 동시에 잡은 사례라고 평가할 수 있다.