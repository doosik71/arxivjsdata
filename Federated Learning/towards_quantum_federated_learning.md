# Towards Quantum Federated Learning

Chao Ren, Rudai Yan, Huihui Zhu, Han Yu, Minrui Xu, Yuan Shen, Yan Xu, Ming Xiao, Zhao Yang Dong, Mikael Skoglund, Dusit Niyato, and Leong Chuan Kwek (2024)

## 🧩 Problem to Solve

본 논문은 양자 컴퓨팅(Quantum Computing, QC)과 연합 학습(Federated Learning, FL)을 결합한 새로운 학문적 분야인 양자 연합 학습(Quantum Federated Learning, QFL)에 대한 포괄적인 분석과 체계적인 정리가 부족하다는 문제를 해결하고자 한다.

전통적인 연합 학습은 데이터 프라이버시를 보호하며 모델을 학습시킬 수 있는 장점이 있으나, 여전히 모델 업데이트 과정에서 민감한 정보가 노출될 위험이 있으며, 대규모 데이터셋과 복잡한 모델을 다룰 때 발생하는 높은 계산 비용과 통신 오버헤드라는 한계가 존재한다. 특히 비볼록 최적화(Non-convex optimization) 문제에서 최적의 모델 성능을 달성하는 데 어려움이 있다.

따라서 본 연구의 목표는 양자 역학의 특성(중첩, 얽힘, 양자 병렬성 등)을 이용하여 연합 학습의 프라이버시, 보안성, 계산 효율성을 획기적으로 향상시키는 QFL의 원리와 기술, 응용 분야를 종합적으로 검토하고, 이를 위한 상세한 분류 체계(Taxonomy)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 QFL 분야의 첫 번째 종합 가이드라인을 제시한 것이며, 주요 내용은 다음과 같다.

1. **QFL의 체계적 분류 체계(Taxonomy) 제안**: QFL을 목적과 사용 기술에 따라 **EffQFL(Efficient QFL)**과 **SecQFL(Secured QFL)**의 두 가지 상위 카테고리로 분류하고, 이를 다시 세부 기술 단위로 구조화하였다.
2. **핵심 양자 기술의 분석**: 양자 신경망(Quantum Neural Networks, QNN), 양자 보안 메커니즘(Quantum-Secured Mechanisms, QSM), 양자 최적화 알고리즘(Quantum Optimization Algorithms, QOA)의 세 가지 관점에서 QFL의 기술적 기반을 상세히 분석하였다.
3. **벤치마킹 가이드라인 제공**: QFL 연구에 사용되는 주요 플랫폼(Qiskit, Cirq, Pennylane 등), 평가 지표(모델 성능, 시스템 성능, 신뢰성 AI 지표), 그리고 실제 적용 사례 및 데이터셋을 정리하여 향후 연구의 기준점을 제시하였다.
4. **미래 연구 방향 제시**: 양자 하드웨어의 한계, 노이즈 및 오류 완화, 모델 및 데이터의 이질성(Heterogeneity) 등 현재의 도전 과제를 식별하고, 개인화 QFL이나 멀티모달 QFL과 같은 장기적인 연구 방향을 제안하였다.

## 📎 Related Works

논문에서는 기존에 발표된 4건의 짧은 리뷰 논문들을 언급하며 본 연구와의 차별점을 강조한다.

- **기존 연구의 한계**: Chen et al.[25]은 하이브리드 양자-고전 분류기의 연합 학습 프레임워크에 집중하였고, Larasati et al.[26]은 QFL의 기초적인 개요와 잠재력만을 논의하였다. Kwak et al.[27]은 양자 분산 딥러닝(QDDL)의 모델 구조 비교에 치중했으며, Qiao et al.[28]은 FL 패러다임 전반을 다루었으나 QFL에 대한 구체적인 분류 체계를 제시하지 않았다.
- **차별점**: 기존 리뷰들이 주로 양자 기계 학습(QML) 측면의 QFL에만 집중한 반면, 본 논문은 **QNN(모델 구조)**, **QSM(보안/통신)**, **QOA(최적화)**라는 세 가지 핵심 축을 모두 포함하여 홀리스틱(Holistic)한 관점에서 QFL을 분석한다. 특히 양자 보안 메커니즘과 최적화 알고리즘을 체계적으로 통합하여 다룬 점이 가장 큰 차별점이다.

## 🛠️ Methodology

본 논문은 QFL의 일반적인 프레임워크를 정의하고, 이를 구현하기 위한 기술적 구성 요소를 체계화하는 방법론을 제시한다.

### 1. QFL의 일반적 동작 절차

QFL의 수학적 프로세스는 고전적 FL과 유사하게 정의된다.

- **로컬 학습**: 각 양자 노드 $i$는 로컬 데이터셋 $\mathcal{D}_i$를 보유하며, QNN을 통해 로컬 모델 $\mathcal{M}_i$를 초기화한다. 이후 QOA를 사용하여 로컬 손실 함수 $\mathcal{L}(\mathcal{M}_i, \mathcal{D}_i)$를 최소화하는 방향으로 학습한다.
- **보안 집계**: QOA 또는 QSM 기반의 보안 집계 프로토콜을 통해 각 노드의 파라미터를 집계하여 글로벌 모델 $\mathcal{M}_{global}$을 생성한다.
  $$\mathcal{M}_{global} = f(\mathcal{M}_1, \mathcal{M}_2, \dots, \mathcal{M}_n)$$
  여기서 $f$는 집계 함수이며, $n$은 참여 노드의 수이다.
- **모델 배포**: 생성된 글로벌 모델은 다시 각 노드로 공유되어 다음 라운드의 로컬 업데이트에 사용된다.

### 2. Proposed Taxonomy: EffQFL vs SecQFL

논문은 QFL을 크게 두 가지 방향으로 분류한다.

#### (1) EffQFL (Efficient QFL)

계산 효율성과 모델 성능 극대화를 목표로 하며, 다음 세 가지 접근 방식으로 나뉜다.

- **Data Processing-based**: 양자 데이터 인코딩(Quantum Data Encoding), 양자 특성 맵핑(Quantum Feature Mapping), 차원 축소 등을 통해 데이터 처리 효율을 높인다.
- **Model Optimization-based**:
  - **Global**: QAOA, VQE 등을 이용한 목적 함수 최적화 및 양자 병렬성을 이용한 모델 집계 가속화.
  - **Local**: 파라미터화된 QNN 설계 및 양자 자연 경사 하강법(Quantum Natural Gradient) 등을 이용한 로컬 모델 최적화.
- **Client Selection-based**: 양자 클러스터링이나 Grover search 등을 이용하여 최적의 클라이언트를 효율적으로 선택하여 통신 비용을 줄인다.

#### (2) SecQFL (Secured QFL)

보안성과 강건성(Robustness) 확보를 목표로 하며, 다음 세 가지 접근 방식으로 나뉜다.

- **Data Privacy-based**: 양자 동형 암호(QHE), 양자 차분 프라이버시(QDP), 양자 보안 다자간 계산(QSMPC)을 통해 데이터 노출을 방지한다.
- **Model Security-based**: 양자 키 분배(QKD), 양자 보안 직접 통신(QSDC), 블라인드 양자 컴퓨팅(BQC)을 사용하여 모델 업데이트 전송 과정을 보호한다.
- **Robustness-based**: 양자 오류 정정(QEC) 및 격자 기반 암호(Lattice-based cryptography) 등을 통해 하드웨어 노이즈와 양자 공격에 대응한다.

## 📊 Results

본 논문은 개별 실험 결과보다는 기존 문헌들을 분석한 종합적인 벤치마킹 결과를 제시한다.

### 1. 구현 플랫폼 및 도구

QFL 연구에서 주로 사용되는 플랫폼은 다음과 같다.

- **SDK/Framework**: IBM의 Qiskit, Google의 Cirq 및 TensorFlow Quantum (TFQ), Xanadu의 Pennylane, Microsoft의 Quantum Development Kit.
- **시뮬레이션 도구**: QuTiP, Strawberry Fields.

### 2. 주요 응용 분야 및 데이터셋

QFL이 적용되고 있는 주요 도메인은 다음과 같다.

- **무선 통신**: NOMA 전력 할당 최적화, 자원 할당 (USNET topology 데이터셋 활용).
- **자연어 처리(NLP)**: 텍스트 분류 (Snips, ATIS 데이터셋), 자동 음성 인식 (Google Speech Command V1 데이터셋).
- **헬스케어**: 의료 이미지 분류 (비알코올성 지방간 질환 데이터셋), 환자 디지털 트윈 생성.
- **스마트 시티/도시 컴퓨팅**: 스마트 에너지 예측, 도시 데이터 관리 (CIFAR, ImageNet 데이터셋).

### 3. 평가 지표

QFL의 성능 측정은 세 가지 관점에서 이루어진다.

- **모델 성능**: 정확도(Accuracy), 수렴 속도(Convergence), 손실 함수 값.
- **시스템 성능**: 통신 효율성(Throughput, Delay), 계산 효율성(CPU 실행 시간), 확장성, 자원 이용률(메모리 사용량).
- **신뢰성 AI**: 강건성(Robustness - Fidelity index 활용), 해석 가능성(Interpretability - Saliency map 활용).

## 🧠 Insights & Discussion

### 강점 및 기회

본 논문은 QFL이라는 신생 분야에 대해 단순한 기술 나열이 아닌, **'효율성(Efficiency)'**과 **'보안성(Security)'**이라는 두 가지 핵심 가치를 중심으로 체계적인 프레임워크를 구축했다는 점에서 학술적 가치가 높다. 특히, 고전적 FL의 한계점을 양자 역학의 특성으로 어떻게 해결할 수 있는지 구체적인 맵핑(예: QKD $\rightarrow$ 모델 보안, VQE $\rightarrow$ 로컬 최적화)을 통해 제시하였다.

### 한계 및 비판적 해석

논문에서 명시한 바와 같이, 대부분의 QFL 연구는 현재 **NISQ(Noisy Intermediate-Scale Quantum)** 시대의 한계 내에 있다. 실제 양자 하드웨어의 큐비트 수 제한과 높은 노이즈 레벨로 인해, 제안된 많은 알고리즘들이 실제 환경보다는 시뮬레이션 단계에 머물러 있다는 점이 가장 큰 한계이다. 또한, 고전적 FL 시스템과의 상호 운용성(Interoperability)을 위한 미들웨어 솔루션에 대한 구체적인 논의가 부족하며, 이는 실제 산업 적용을 위해 반드시 해결되어야 할 과제이다.

## 📌 TL;DR

본 논문은 양자 컴퓨팅과 연합 학습을 결합한 **양자 연합 학습(QFL)** 분야의 첫 번째 종합 서베이 논문이다. 저자들은 QFL을 **효율성 중심의 EffQFL**과 **보안성 중심의 SecQFL**로 분류하는 독자적인 Taxonomy를 제안하였으며, QNN, QSM, QOA 기술이 어떻게 FL의 병목 지점(계산 비용, 통신 오버헤드, 프라이버시 노출)을 해결할 수 있는지 분석하였다. 이 연구는 향후 양자 하드웨어의 발전과 함께 의료, 금융, 통신 등 고도의 보안과 효율성이 요구되는 분야에서 차세대 분산 학습 패러다임을 구축하는 데 핵심적인 지침서 역할을 할 것으로 기대된다.
