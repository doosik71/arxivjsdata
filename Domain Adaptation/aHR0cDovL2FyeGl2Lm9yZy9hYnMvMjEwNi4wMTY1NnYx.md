# Generalized Domain Adaptation

Yu Mitsuzumi, Go Irie, Daiki Ikami, Takashi Shibata (2021)

## 🧩 Problem to Solve

본 논문은 Unsupervised Domain Adaptation (UDA) 분야에서 개별적으로 제안되어 온 수많은 변형 문제들이 서로 파편화되어 있다는 점에 주목한다. 기존의 UDA 연구들은 특정 조건(예: 소스 도메인은 완전히 라벨링되어 있고 타겟 도메인은 완전히 라벨링되지 않음) 하에서만 작동하도록 설계되었으며, 이로 인해 한 가지 변형 문제에 최적화된 방법론이 다른 변형 문제에서는 효과가 없거나 적용조차 불가능한 경우가 많다.

특히, 실제 환경에서는 어떤 UDA 변형 문제에 직면해 있는지 정확히 식별하기 어렵고, 여러 변형 문제가 복합적으로 나타나는 경우가 많다. 저자들은 이러한 문제를 해결하기 위해 모든 주요 UDA 변형을 포괄할 수 있는 일반화된 표현 방식인 Generalized Domain Adaptation (GDA)을 정의한다.

또한, GDA라는 일반화된 프레임워크를 통해 기존 방법론들이 해결하지 못하는 새로운 도전적인 설정(new challenging setting)을 제시한다. 이는 모든 샘플에 대해 도메인 라벨(domain labels)이 완전히 알려지지 않았으며($\forall \delta_d = 0$), 클래스 라벨 또한 각 도메인의 일부에만 부분적으로 주어진 상황을 의미한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1.  **Generalized Domain Adaptation (GDA) 프레임워크 제안**: 기존의 다양한 UDA 변형 문제(MSDA, MTDA, OSDA, PDA 등)를 하나의 일관된 수학적 표현으로 통합하여 체계적인 분석이 가능하게 하였다.
2.  **새로운 UDA 문제 설정 정의**: 도메인 라벨이 전혀 제공되지 않고 클래스 라벨이 불완전한 상황이라는, 기존 UDA 방법론들이 대응할 수 없는 새로운 문제 상황을 정의하고 그 어려움을 입증하였다.
3.  **Self-supervised Class-destructive Learning 제안**: 도메인 라벨 없이도 각 샘플의 도메인을 추정하기 위해, 이미지의 지역적 구조를 파괴하여 클래스 정보를 제거하고 도메인 정보만을 추출하는 자가 지도 학습 기반의 새로운 방법론을 제안하였다.

## 📎 Related Works

기존의 UDA 연구들은 소스 도메인과 타겟 도메인의 간극(domain shift)을 줄이는 데 집중해 왔다. 주요 변형 연구들은 다음과 같다.

-   **Multi-Source/Target UDA**: 소스나 타겟 도메인이 여러 개인 경우를 다루며, 일반적으로 하위 도메인 라벨이 존재한다고 가정한다.
-   **Open Set 및 Partial UDA**: 소스와 타겟의 클래스 집합이 일치하지 않는 경우를 다룬다. Open Set Domain Adaptation (OSDA)은 타겟 도메인에 소스에는 없는 클래스가 존재하는 경우를, Partial Domain Adaptation (PDA)은 소스의 클래스 집합이 타겟의 상위 집합인 경우를 다룬다.

기존 접근 방식들의 한계는 대부분 도메인 라벨이 최소한 하나의 도메인에 대해서는 가시적($\forall \delta_d = 1$)이라고 가정한다는 점이다. 반면, 본 논문이 제안하는 GDA는 도메인 라벨이 완전히 부재한 상황까지 확장하여 기존 연구들과 차별화된다.

## 🛠️ Methodology

본 논문이 제안하는 방법론은 크게 두 단계로 구성된다: (1) 모든 샘플에 대한 도메인 라벨 추정, (2) 추정된 라벨을 이용한 도메인 불변 분류기 학습이다.

### 1. Domain Label Estimation (도메인 라벨 추정)

도메인 라벨이 없는 상황에서 단순히 특징 공간의 클러스터링을 수행하면, 클래스 정보가 도메인 정보보다 강하게 작용하여 클래스 기반의 클러스터가 형성되는 문제가 발생한다. 이를 방지하기 위해 **Self-supervised Class-destructive Learning**을 도입한다.

-   **Class-destructive Transformation**: 이미지의 지역적 구조(형태, 부품 간의 연결성 등)가 클래스 정보의 핵심이라는 점에 착안하여, 이미지를 여러 픽셀 블록으로 나눈 뒤 그 위치를 무작위로 섞는(shuffle) 변환을 수행한다. 이를 통해 클래스 식별 정보는 파괴되지만, 전역적인 픽셀 통계 정보인 도메인 정보는 어느 정도 유지된다.
-   **Self-supervised Learning**: 변환된 이미지들을 사용하여 자가 지도 학습을 수행한다. 구체적으로, SimCLR와 유사하게 동일 이미지에서 생성된 두 개의 증강된 뷰 사이의 $\text{normalized temperature-scaled cross-entropy loss}$를 최소화하여 클래스 불변(class-invariant) 특징을 학습한다.
-   **Clustering**: 학습된 특징에 대해 Gaussian Mixture 모델을 이용한 클러스터링을 수행하여 각 샘플에 도메인 라벨을 부여한다.

### 2. Classifier Learning (분류기 학습)

추정된 도메인 라벨을 바탕으로 도메인 적대적 학습(Domain-adversarial Learning)을 수행하여 도메인 불변 분류기를 학습시킨다.

-   **시스템 구조**: 공유 특징 추출기 $G_f$, 클래스 예측기 $F_y$, 도메인 분류기 $F_d$로 구성된다.
-   **학습 목표**: 클래스 분류 성능은 높이되, 도메인 분류기는 도메인을 구분하지 못하도록 특징 추출기를 학습시킨다. 이는 다음과 같은 목적 함수로 표현된다.
    $$\min_{G_f, F_y} L_y - \lambda L_d$$
    $$\min_{F_d} L_d$$
    여기서 $L_y$는 클래스 분류 손실, $L_d$는 도메인 분류 손실이며, Gradient Reversal Layer (GRL)를 통해 효율적으로 최적화한다.

-   **Unknown Class 및 Unlabeled Sample 처리**:
    라벨이 없는 샘플에 대해 엔트로피(entropy) 기반의 의사 라벨링(pseudo-labeling)을 적용한다. 클래스 확률 분포의 엔트로피 $H(y|x)$가 임계값 $\sigma$보다 크면 미지의 클래스($\text{UNK}$)로 간주하고, 그렇지 않으면 가장 확률이 높은 클래스로 할당한다.
    $$y = \begin{cases} \text{UNK} & (H(y|x) > \sigma) \\ \text{argmax}_k F(x)[k] & (\text{otherwise}) \end{cases}$$
    이후 Joint label-network optimization 프레임워크를 통해 네트워크 파라미터와 의사 라벨을 동시에 업데이트한다.

## 📊 Results

### 실험 설정
-   **데이터셋**: Digits (SVHN, SynDigits, MNIST, MNIST-M), Office-31, Office-Home.
-   **평가 지표**: 알려진 클래스 정확도 ($\text{OS}^*$), 미지 클래스 정확도 ($\text{UNK}$), 그리고 이 둘의 조화 평균인 $\text{HOS (Harmonic Mean)}$를 사용한다.
-   **비교 대상**: Labeled Only, MCD, OSBP, ROS, UAN 등 최신 UDA 방법론들을 baseline으로 설정하여 비교하였다.

### 주요 결과
1.  **새로운 UDA 문제(GDA1, GDA2)에서의 성능**:
    기존 방법론들은 도메인 라벨 부재 시 클래스 정보와 도메인 정보를 분리하지 못해 성능이 급격히 저하되었다. 반면, 제안 방법은 $\text{HOS}$ 지표에서 baseline들을 압도하며 탁월한 성능을 보였다. 특히 Digits 데이터셋의 특정 설정에서 baseline들이 $15\%$ 미만의 $\text{HOS}$를 기록할 때, 제안 방법은 $77\%$ 이상의 성능을 달성하였다.
2.  **기존 UDA 변형 문제에서의 성능**:
    OSDA, MS-OSDA, BTDA 실험 결과, 제안 방법은 특정 문제에 특화되어 설계되지 않았음에도 불구하고 최신 SOTA 방법론들과 경쟁 가능한 수준의 성능을 보였다. 이는 GDA 프레임워크가 범용적으로 적용 가능함을 시사한다.
3.  **분석 결과**:
    -   클러스터 개수가 실제 도메인 수와 일치할 때 $\text{NMI}$와 $\text{HOS}$가 최대가 됨을 확인하였다.
    -   Class-destructive transformation의 그리드 크기가 너무 세밀하면 도메인 정보까지 파괴되어 성능이 하락하며, $3 \times 3$ 설정이 적절함을 보였다.

## 🧠 Insights & Discussion

본 논문의 강점은 UDA의 다양한 변형들을 하나의 통합된 프레임워크(GDA)로 정의함으로써, 연구의 파편화를 막고 새로운 문제 영역을 발굴했다는 점에 있다. 특히, 이미지의 지역적 구조를 파괴하는 단순하면서도 강력한 직관을 통해 도메인 라벨 없이 도메인 정보를 추출해 낸 점이 인상적이다.

다만, 몇 가지 한계와 논의 사항이 존재한다.
첫째, 제안 방법이 기존 UDA 변형에서는 경쟁력이 있으나, 일부 SOTA 방법론(예: MS-OSDA의 MOSDANET)보다는 약간 낮은 성능을 보인다. 이는 특정 문제에 최적화된 전용 알고리즘보다는 범용적인 접근 방식이기 때문으로 풀이된다.
둘째, Open Compound Domain Adaptation (OCDA)와 같은 더 복잡한 설정에서는 OCDA 전용 방법론보다 성능이 낮게 나타났다. 이는 도메인 라벨 추정 이후의 과정이 여전히 기존의 적대적 학습에 의존하고 있기 때문일 수 있으며, 향후 연구를 통해 확장 가능성이 남아 있다.

결론적으로, 본 연구는 도메인 라벨이라는 강력한 가이드 없이도 데이터 자체의 특성을 이용해 도메인 적응을 수행할 수 있음을 증명하였다.

## 📌 TL;DR

본 논문은 파편화된 UDA 변형 문제들을 통합하는 **Generalized Domain Adaptation (GDA)** 프레임워크를 제안하고, 도메인 라벨이 전혀 없는 극한의 상황에서도 작동하는 방법론을 제시한다. 핵심 아이디어인 **Self-supervised Class-destructive Learning**은 이미지의 지역 구조를 무작위로 섞어 클래스 정보를 제거함으로써 도메인 특징만을 추출하고, 이를 통해 도메인 라벨을 스스로 추정한다. 실험 결과, 제안 방법은 도메인 라벨이 없는 새로운 설정에서 기존 방법들을 압도하며, 기존의 다양한 UDA 문제에서도 매우 경쟁력 있는 성능을 보였다. 이는 향후 도메인 정보가 부족한 실제 산업 현장의 데이터 적응 연구에 중요한 기초가 될 것으로 보인다.