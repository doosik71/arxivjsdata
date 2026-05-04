# Dummy Prototypical Networks for Few-Shot Open-Set Keyword Spotting

Byeonggeun Kim, Seunghan Yang, Inseop Chung, & Simyung Chang (2022)

## 🧩 Problem to Solve

본 논문은 키워드 스포팅(Keyword Spotting, KWS) 분야에서 **Few-Shot Open-Set Recognition (FSOSR)** 문제를 해결하고자 한다. 일반적인 KWS는 미리 정의된 키워드 집합을 분류하는 것을 목표로 하지만, 실제 환경에서는 다음과 같은 두 가지 복합적인 요구사항이 존재한다.

1.  **Few-Shot Learning (FSL)**: 새로운 키워드를 등록할 때 매우 적은 수의 샘플(support samples)만으로 해당 키워드를 인식할 수 있어야 한다.
2.  **Open-Set Recognition (OSR)**: 학습 과정에서 보지 못한, 그리고 지원 샘플조차 없는 예상치 못한 카테고리(open-set)의 발화가 입력되었을 때, 이를 기존 클래스로 오분류하지 않고 '알 수 없음'으로 거부(reject)할 수 있어야 한다.

특히 FSOSR은 일반적인 OSR보다 더 어렵다. 그 이유는 에피소드마다 선택되는 $N$개의 기지 클래스(known classes)가 달라지므로, 이에 따라 **'알 수 없는 영역(open-set)'의 정의가 에피소드마다 동적으로 변하기 때문**이다. 따라서 본 논문의 목표는 이러한 가변적인 open-set에 적응하여 효과적으로 미지 클래스를 탐지하는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **에피소드 기반의 더미 프로토타입(Episode-known Dummy Prototypes)**을 생성하여 미지 클래스를 탐지하는 것이다.

기존의 FSOSR 방식들이 미지 샘플의 '이상치(abnormality)'를 탐지하는 데 집중했다면, 본 연구에서는 **현재 에피소드에 포함된 기지 클래스들의 정보를 바탕으로, 그들과 구별되는 가상의 '더미 클래스'를 생성**하여 이를 통해 open-set을 직접적으로 분류하는 방식을 제안한다. 이를 구현한 모델이 바로 **Dummy Prototypical Networks (D-ProtoNets)**이다.

## 📎 Related Works

1.  **Few-Shot Learning (FSL)**: 주로 Adaptation, Hallucination, Metric Learning 방법론으로 나뉜다. 본 논문은 거리 기반의 Metric Learning, 특히 Prototypical Networks(ProtoNets)를 기반으로 한다.
2.  **Open-Set Recognition (OSR)**: 고정된 closed-set 환경에서 미지 클래스를 탐지하는 연구들이 진행되었으며, 최근에는 Manifold Mixup 등을 이용한 learnable dummy classifier 연구가 있었다. 하지만 이는 고정된 환경을 전제로 하므로, 에피소드마다 클래스가 변하는 FSOSR에는 적용하기 어렵다.
3.  **Few-Shot Open-Set Recognition (FSOSR)**: PEELER나 SnaTCHer 같은 최신 기법들이 존재한다. PEELER는 엔트로피 최대화 손실과 Gaussian Embedding을 사용하며, SnaTCHer는 변환 일관성(transformation consistency)을 통해 미지 클래스를 탐지한다. 본 제안 방법은 이러한 간접적 탐지 대신 더미 클래스를 직접 학습한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
D-ProtoNets는 기본적으로 Prototypical Networks 구조를 따르되, 더미 생성기(Dummy Generator)가 추가된 형태이다. 인코더 $f_\phi(\cdot)$를 통해 특징 공간으로 투영된 후, 기지 클래스의 프로토타입들과 함께 생성된 더미 프로토타입이 분류에 참여한다.

### 주요 구성 요소 및 절차

**1. Prototypical Networks (기반 모델)**
각 클래스 $n$의 지원 샘플 $S_n$의 평균을 통해 프로토타입 $c_n$을 계산한다.
$$c_n = \frac{1}{M} \sum_{(x_i, y_i) \in S_n} f_\phi(x_i)$$
이후 쿼리 샘플 $x$와 각 프로토타입 간의 거리 $d(\cdot)$를 이용하여 확률 분포를 계산한다.

**2. 에피소드 기반 더미 프로토타입 생성 ($g_\phi$)**
에피소드 내의 기지 프로토타입 집합 $C = [c_1; c_2; \dots; c_N]$을 입력으로 받아, 순열 불변성(permutation invariance)을 갖는 DeepSets 구조의 더미 생성기 $g_\phi$를 통해 더미 프로토타입 $c_d$를 생성한다.
$$c_d = g_\phi(C) = \text{Maxpool}(g_1(C))W_g$$
여기서 $g_1$은 FC 레이어로 구성되며, $\text{Maxpool}$은 $N$개의 클래스 차원에서 수행되어 고정된 크기의 특징을 추출하고, $W_g$는 학습 가능한 행렬이다.

**3. 분류 및 학습 목표**
이제 프로토타입 집합은 $\{c_1, \dots, c_N, c_d\}$로 확장되며, 총 $N+1$개의 클래스를 분류하게 된다. 소프트맥스 함수에 온도 파라미터 $\tau$를 도입하는데, 특히 더미 클래스의 손실을 쉽게 줄여주기 위해 더미의 온도 $\tau_{N+1}$을 더 크게 설정한다 ($\tau_{N+1} = \gamma \cdot \tau_n, \gamma > 1$).

손실 함수는 기지 쿼리($Q_K$)와 미지 쿼리($Q_U$)에 대한 Cross Entropy(CE) 손실의 가중 합으로 정의된다.
$$L_{CE} = L_{CE}^K + \lambda \cdot L_{CE}^U$$

**4. Multiple Dummies 및 Gumbel-Softmax**
단일 더미의 한계를 극복하기 위해 $L$개의 더미를 생성할 수 있다. 이때 비미분 가능한 $\text{arg max}$ 연산을 대체하기 위해 **Gumbel-Softmax**를 사용하여 미분 가능한 형태로 샘플링하고, 학습 시에는 가중 합으로 더미를 생성한다.

**5. 추론 절차**
테스트 시에는 $N$개의 기지 클래스 중 가장 확률이 높은 것을 선택하거나, 더미 클래스의 확률 $p_\theta(y_i = y_d | x_i)$가 특정 임계값 $\delta$보다 큰지를 확인하여 open-set 여부를 판단한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Google Speech Commands (GSC) ver2를 기반으로 한 `splitGSC` 벤치마크를 제안하였다. (Train 15, Val 10, Test 10 키워드로 분할)
- **평가 지표**: FSL 성능은 Accuracy로, OSR 성능은 AUROC(Area Under the ROC Curve)로 측정하였다.
- **비교 대상**: ProtoNet, FEAT, PEELER, SnaTCHer.
- **백본**: Conv4-64, ResNet-12, BCResNet-8.

### 주요 결과
1.  **`splitGSC` 결과**: D-ProtoNets는 모든 백본에서 Vanilla ProtoNet보다 월등한 AUROC 향상을 보였으며, 최신 FSOSR 기법인 SnaTCHer보다도 우수한 성능을 기록하였다. 특히 ResNet-12 백본의 5-shot 설정에서 AUROC 86.7%를 달성하며 SOTA 성능을 보였다.
2.  **miniImageNet 결과**: 이미지 데이터셋에서도 검증한 결과, FSL 성능을 유지하면서도 open-set 탐지율(AUROC)에서 SOTA 수준의 성능을 기록하였다.
3.  **Ablation Study**: 
    - 더미 프로토타입의 도입만으로도 AUROC가 10% 이상 크게 향상되었다.
    - 더미의 개수 $L=3$, 온도 계수 $\gamma=3$에서 성능이 수렴하는 경향을 보였다.
    - **RFN (Relaxed instance Frequency-wise Normalization)**: 오디오 특징의 불필요한 변동(예: 화자 ID)을 줄여주는 RFN 모듈을 추가했을 때, 모든 설정에서 성능이 일관되게 향상됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 FSOSR 문제에서 '미지 영역'이 에피소드마다 변한다는 점에 주목하여, 이를 동적으로 생성하는 **더미 프로토타입**이라는 단순하면서도 강력한 해결책을 제시하였다. 

특히 흥미로운 점은 SnaTCHer와 같은 거리 기반 탐지 방식보다, 명시적인 더미 클래스를 학습시켜 소프트맥스 확률로 판단하는 방식이 더 효과적일 수 있음을 보였다는 것이다. 또한, 오디오 데이터의 특성상 화자나 환경에 따른 변동성이 크기 때문에, RFN과 같은 정규화 기법이 FSOS-KWS 성능 향상에 매우 중요한 역할을 한다는 통찰을 제공하였다.

다만, 더미 생성기 $g_\phi$의 구조가 매우 단순한 FC 레이어 기반이라는 점과, 최적의 임계값 $\delta$를 설정하는 구체적인 방법론에 대한 논의가 더 보강될 필요가 있다.

## 📌 TL;DR

이 논문은 에피소드마다 변하는 미지 클래스를 탐지하기 위해, 기지 클래스의 프로토타입들을 입력으로 받아 가상의 미지 클래스 대표값(더미 프로토타입)을 생성하는 **D-ProtoNets**를 제안한다. 제안된 방법은 KWS를 위한 새로운 벤치마크 `splitGSC`와 이미지 데이터셋 `miniImageNet` 모두에서 기존 FSOSR 기법들을 상회하는 미지 클래스 탐지 성능(AUROC)을 보여주었으며, 오디오 특징 정규화(RFN)의 중요성을 함께 입증하였다. 이는 향후 적은 데이터만으로 새로운 키워드를 등록하면서도 오작동을 방지해야 하는 온디바이스 KWS 시스템 구현에 중요한 기여를 할 것으로 보인다.