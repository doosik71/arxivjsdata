# Fine-grained anomaly detection via multi-task self-supervision

Loïc Jezequel, Ngoc-Son Vu, Jean Beaudet, Aymeric Histace (2021)

## 🧩 Problem to Solve

본 논문은 딥러닝 기반의 이상치 탐지(Anomaly Detection, AD)에서 특히 **세밀한 차이(Fine-grained differences)**를 가진 이상치를 탐지하는 데 어려움이 있다는 점을 해결하고자 한다. 

기존의 자기지도학습(Self-Supervised Learning, SSL) 기반 이상치 탐지 방법들은 주로 기하학적 변환(Geometric transformation) 인식과 같은 단순한 태스크를 활용한다. 하지만 이러한 방법들은 주로 고수준의 형태(High-scale shape) 특징에 의존하기 때문에, 객체 전체의 형태는 유사하지만 국소적인 부분이나 세부 스타일이 다른 'Fine-grained' 문제에서는 성능이 저하되는 한계가 있다.

따라서 본 연구의 목표는 고수준의 형태 특징과 저수준의 세밀한 특징(Low-scale fine features)을 동시에 학습할 수 있는 멀티태스크(Multi-task) 프레임워크를 구축하여, 스타일 이상치(Style anomaly) 및 국소 이상치(Local anomaly)를 포함한 세밀한 이상치 탐지 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **상호 보완적인 두 가지 SSL 태스크를 동시에 수행하는 멀티태스크 학습 구조**를 도입하는 것이다.

1. **멀티태스크 SSL 구조**: 고수준의 형태 정보를 학습하는 기하학적 변환 인식 태스크와 저수준의 국소적 특징을 학습하는 지그소 퍼즐(Jigsaw puzzle) 태스크를 독립적으로 해결함으로써 더 풍부한 표현(Representation)을 학습한다.
2. **Fine-grained AD 성능 향상**: 기존의 단일 태스크 기반 방법론들이 놓치기 쉬운 세밀한 특징을 캡처함으로써, 스타일 및 국소 이상치 탐지 능력을 크게 개선하였다.
3. **광범위한 검증**: One-vs-all, Out-of-distribution(OOD) 탐지, 그리고 얼굴 위조 탐지(Face presentation attack detection) 등 다양한 시나리오에서 제안 방법의 효율성을 입증하였다.

## 📎 Related Works

### 기존 이상치 탐지 접근 방식
- **전통적 방법**: 사전 학습된 신경망으로 특징을 추출한 후 One-Class SVM(OCSVM)이나 Isolation Forest(IF)와 같은 고전적 알고리즘을 적용하는 방식이 사용되었다.
- **준지도학습(Semi-supervised)**: DeepSAD나 Deviation networks와 같이 일부 이상치 샘플을 사용하여 정상 클래스의 경계를 학습하고, 정상 샘플은 구의 중심(Hypersphere center)으로 모으고 이상치는 멀어지게 하는 Compactness 원리를 이용한다.

### SSL 기반 이상치 탐지 (SSL AD)
- **작동 원리**: 정상 데이터만을 사용하여 보조 태스크(Pretext task)를 학습시킨 후, 추론 단계에서 모델이 해당 태스크를 얼마나 잘 수행하는지를 측정하여 이상치 점수를 산출한다. 정상 데이터에는 높은 성능을 보이지만, 학습 시 보지 못한 이상치 데이터에는 낮은 성능을 보일 것이라는 가정에 기반한다.
- **기존 연구의 한계**: GeoTrans나 MHRot 같은 연구들은 기하학적 변환 분류를 사용한다. 그러나 저자들은 이러한 태스크들이 너무 쉬울 경우 모델이 일반적인 시각적 특징만을 학습하게 되어, 세밀한 이상치를 구분하는 데 필요한 미세 특징을 학습하지 못한다는 점을 지적한다.

## 🛠️ Methodology

### 전체 시스템 구조
모델은 공통의 특징 추출기(Feature extractor) $\phi$와 각 태스크별로 독립적인 전결합층(Dense layer) $f^T_1, \dots, f^T_N$으로 구성된다. 모든 태스크는 동일한 $\phi$를 공유하여 학습하며, 이를 통해 더 풍성한 특징 표현을 얻는다.

### 주요 구성 요소 및 태스크
1. **기하학적 변환 태스크 (Geometric Transformation Task)**: 수직 이동(Vertical translation), 수평 이동(Horizontal translation), 90도 회전(90° rotation)을 분류한다. 이는 주로 객체의 전반적인 형태(High-scale shape)를 인식하게 한다.
2. **지그소 퍼즐 태스크 (Jigsaw Puzzle Task)**: 이미지를 $3 \times 3$ 그리드로 나누고 무작위로 섞은 뒤 원래 순서를 예측하는 태스크이다. 저자들은 계산 복잡도를 줄이기 위해 전체 순열이 아닌 $k$개의 무작위 순열(본 연구에서는 $k=3$)만을 분류하는 단순화된 버전을 사용하였다. 또한, 단순한 픽셀 매칭을 방지하기 위해 패치 사이에 무작위 오프셋(Margin)을 추가하였다.

### 학습 및 추론 절차
#### 훈련 목표 및 손실 함수
각 태스크 $T_i$에 대해 교차 엔트로피 손실(Cross-Entropy Loss, $L_{CE}$)을 사용하며, 전체 손실 함수 $L(x)$는 다음과 같이 모든 태스크의 손실 합으로 정의된다.

$$L(x) = \sum_{i=1}^{3} L_{CE}(\phi \circ f_v(T^{(v)}_i(x)), i) + \sum_{i=1}^{3} L_{CE}(\phi \circ f_h(T^{(h)}_i(x)), i) + \sum_{i=1}^{4} L_{CE}(\phi \circ f_{rot}(T^{(rot)}_i(x)), i) + \sum_{i=1}^{k} L_{CE}(\phi \circ f_{puzz}(T^{(puzz)}_i(x)), i)$$

여기서 $f_v, f_h, f_{rot}, f_{puzz}$는 각각 수직 이동, 수평 이동, 회전, 퍼즐 태스크를 위한 헤드이다.

#### 추론 및 이상치 점수 산출
추론 시에는 각 태스크에 대해 정답 클래스에 해당하는 소프트맥스(Softmax) 점수를 계산한다. 최종 이상치 점수 $s_a(x)$는 모든 태스크의 점수 평균으로 계산된다.

$$s_a(x) = \frac{1}{N} \sum_{i=1}^{N} s^{(T_i)}_a(x)$$
$$s^{(T_i)}_a(x) = \sum_{j} \text{softmax}(\phi \circ f^T_i(T^{(i)}_j(x)))_j$$

이 방식은 MHRot과 같이 모든 조합을 테스트하는 방식보다 추론 속도가 약 10배 빠르다는 장점이 있다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - One-vs-all: MNIST, Fashion MNIST, CIFAR-100
    - Fine-grained: Caltech-Birds 200, FounderType (폰트 인식)
    - Anti-spoofing: SiW-M (얼굴 위조 탐지)
    - OOD: CIFAR-10 (In-distribution) vs CIFAR-100, SVHN (Out-of-distribution)
- **지표**: AUROC를 기본으로 사용하며, 얼굴 위조 탐지에서는 EER, APCER@5%BPCER를 추가로 측정하였다.
- **네트워크**: WideResNet 16-4를 특징 추출기로 사용하였다.

### 주요 결과
1. **정량적 성능**: 제안 방법은 대부분의 데이터셋에서 최신 기법(SOTA)을 능가하였다. 특히 세밀한 특징이 중요한 데이터셋에서 큰 향상을 보였으며, SiW-M 데이터셋에서는 AUROC 기준 최대 31%의 상대적 오차 감소를 달성하였다.
2. **Ablation Study**: 기하학적 태스크(G)만 사용했을 때나 퍼즐 태스크(J)만 사용했을 때보다, 두 태스크를 결합(G+J)했을 때 성능이 월등히 높았다. 특히 데이터셋이 세밀할수록(CIFAR-100 $\rightarrow$ Caltech-Birds $\rightarrow$ SiW-M) 결합으로 인한 성능 향상 폭이 커졌다.
3. **얼굴 위조 탐지**: MHRot과 비교했을 때 APCER@5%BPCER 지표가 $77.5$에서 $39.1$로 급격히 감소하여, 실질적인 위조 탐지 성능이 크게 개선되었음을 입증하였다.
4. **OOD 탐지**: 복잡한 정상 클래스 분포를 가진 OOD 상황에서도 다른 SSL 기반 방법들보다 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### 태스크 선정의 원칙 (Rule of Thumb)
본 논문은 SSL 기반 AD에서 태스크 난이도의 중요성을 강조한다.
- **태스크가 너무 어려울 경우**: 정상 샘플에서도 정확도가 낮아 의미 있는 표현을 학습할 수 없으며, 결과적으로 이상치 탐지 성능이 낮아진다.
- **태스크가 너무 쉬울 경우**: 모델이 아주 단순하고 일반적인 특징만으로 정답을 맞힐 수 있게 되어, 정상 클래스에 특화된 세밀한 특징을 학습하지 않는다. 이는 이상치 샘플에서도 높은 정확도를 보여 이상치 탐지에 실패하게 만든다.

### 강점 및 한계
- **강점**: 고수준(형태)과 저수준(국소 특징) 정보를 동시에 학습하는 멀티태스크 전략을 통해, 기존 SSL AD가 해결하지 못했던 'Fine-grained' 이상치 탐지 문제를 효과적으로 해결하였다.
- **한계**: 본 연구에서는 판별적(Discriminative) 태스크만을 사용하였다. 저자들은 향후 연구에서 이미지 인페인팅(In-painting)이나 색상 복원(Re-colorization)과 같은 생성적(Generative) 태스크를 결합하는 것이 성능을 더 높일 수 있는 가능성이 있다고 언급한다.

## 📌 TL;DR

본 논문은 기하학적 변환 인식(고수준 형태 학습)과 지그소 퍼즐(저수준 국소 특징 학습)을 결합한 **멀티태스크 자기지도학습 프레임워크**를 제안하여, 기존 방법론들이 어려움을 겪었던 **세밀한 이상치(Fine-grained anomaly) 탐지 성능을 획기적으로 개선**하였다. 특히 얼굴 위조 탐지와 같은 고난도 작업에서 탁월한 성능을 보였으며, 이는 향후 보안 및 정밀 검수 분야의 이상치 탐지 연구에 중요한 방향성을 제시한다.