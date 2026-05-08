# VISION-QRWKV: EXPLORING QUANTUM-ENHANCED RWKV MODELS FOR IMAGE CLASSIFICATION

Chi-Sheng Chen (2025)

## 🧩 Problem to Solve

본 논문은 이미지 분류 작업에서 고전적인 신경망 구조가 가지는 표현력의 한계를 극복하고자 한다. 특히, Transformer의 대안으로 주목받는 RWKV(Receptance Weighted Key Value) 아키텍처는 효율적인 시퀀스 모델링 능력을 갖추고 있으나, 그 핵심인 channel-mixing(피드포워드 층)의 선형성 및 구조적 특성으로 인해 복잡한 특징 공간(feature space)에서의 표현력에 제약이 있다.

연구의 목표는 Variational Quantum Circuit(VQC)을 RWKV의 channel-mixing 모듈에 통합함으로써 비선형 특징 변환 능력을 향상시키고, 결과적으로 이미지 표현의 표현력을 높이는 hybrid quantum-classical 모델인 Vision-QRWKV를 제안하고 검증하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 고전적인 RWKV 아키텍처의 attention-free한 backbone은 유지하면서, 연산 집약적인 channel-mixing 층에 양자 회로를 결합하는 것이다.

중심적인 설계 직관은 양자 계산이 제공하는 고차원 힐베르트 공간(Hilbert space)에서의 비선형 매핑 능력을 활용하여, 고전적인 MLP(Multi-Layer Perceptron)가 포착하기 어려운 복잡한 이미지 특징 간의 상호작용을 효과적으로 학습하는 데 있다. 이는 특히 클래스 간 경계가 모호하거나 노이즈가 많은 의료 이미지 데이터셋에서 강력한 성능 향상을 가져올 것으로 기대하였다.

## 📎 Related Works

### 2.1 RWKV 및 Attention-Free 모델

RWKV는 Transformer의 성능을 유지하면서도 RNN의 선형 시간/메모리 복잡도를 가지는 모델이다. 기존 연구들은 주로 텍스트나 시계열 데이터에 집중해 왔으며, 이미지 분류 작업으로의 확장 및 양자 강화 적용 사례는 본 논문이 처음이다.

### 2.2 시각화를 위한 양자 신경망(QNN)

양자 분류기, Quantum Convolutional Neural Networks(QCNN) 등이 연구되어 왔으며, 일부 연구에서는 뇌 영상이나 EEG 데이터, MNIST 이미지 생성에 양자 층을 통합하려는 시도가 있었다. 그러나 표준 이미지나 의료 이미지 분류 작업에 대규모로 적용된 사례는 드물다.

### 2.3 하이브리드 양자-고전 아키텍처

현재의 NISQ(Noisy Intermediate-Scale Quantum) 하드웨어 제약으로 인해, 고전 층과 VQC를 결합한 하이브리드 형태가 주를 이룬다. 저자는 이전 연구에서 시계열 예측을 위해 QuantumRWKV를 제안한 바 있으며, 본 논문은 이를 시각 영역으로 확장한 것이다.

## 🛠️ Methodology

### 1. 전체 파이프라인

Vision-QRWKV는 입력 이미지를 1차원 토큰 시퀀스로 평탄화(flatten)한 후, 여러 층의 하이브리드 RWKV 블록을 통과시킨다. 각 블록은 Time Mixing 모듈과 Quantum Channel Mixing 모듈로 구성되며, 최종 출력은 Global Average Pooling 또는 classification head를 통해 분류된다.

### 2. 구성 요소 및 동작 원리

**가. Time Mixing 모듈**
지수 감쇠 가중치(exponential decay weights)를 사용하여 시퀀스 내의 의존성을 처리하며, 이는 고전적인 RWKV의 구조를 그대로 따른다.

**나. Quantum Channel Mixing (QuantumMix)**
이 모듈은 고전적인 MLP와 VQC의 병렬 구조로 설계되었다.

1. **양자 임베딩**: 고전적 은닉 상태 $x$를 하위 차원의 양자 임베딩 공간으로 투영한다.
   $$x^q = W^q x$$
2. **VQC 처리**: 투영된 $x^q$의 각 요소를 각도 임베딩(angle embedding)을 통해 회전 게이트($RX$ 또는 $RY$)에 입력한다. 이후 CNOT 게이트가 래더(ladder) 패턴으로 배치된 $L$개의 얽힘 층(entangling layers)을 통과한다.
3. **측정**: Pauli-Z 연산자의 기댓값을 통해 출력값 $z_i$를 얻는다.
   $$z_i = \langle \psi | Z_i | \psi \rangle$$
4. **융합**: VQC의 출력 $z$를 선형 투영하여 고전적 FFN의 출력과 결합한다.
   $$\text{QuantumMix}(x) = \sigma(r) \odot (W_2(\text{ReLU}(W_1 x)) + W_o z)$$
   여기서 $\sigma(r)$은 학습 가능한 receptance 벡터이며, $W_1, W_2$는 고전적 투영 행렬, $W_o$는 양자 출력을 임베딩 차원으로 매핑하는 행렬이다.

### 3. 학습 절차

모델은 $x^{(l+1)} = x^{(l)} + \text{TimeMix}(\text{LN}(x^{(l)})) + \text{QuantumMix}(\text{LN}(x^{(l)} + \text{TimeMix}))$ 구조를 가지며, 모든 양자 층은 PennyLane의 `default.qubit` 백엔드를 통해 시뮬레이션된다. 이를 통해 전체 파이프라인에서 gradient backpropagation이 가능한 완전 미분 가능한 형태로 학습된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MNIST, FashionMNIST 및 12개의 MedMNIST 데이터셋(Blood, Tissue, OCT, Path, Chest, OrganA/S/C, Derma, Pneumonia, Retina, Breast) 등 총 14종. 모든 이미지는 $8 \times 8$ 해상도로 리사이징되었다.
- **모델 설정**: 임베딩 크기 768, RWKV 블록 4층, 4-qubit VQC(깊이 2층).
- **학습 조건**: Adam optimizer, 학습률 $1 \times 10^{-3}$, 배치 크기 64, 30 epoch 학습. NVIDIA A100 GPU 사용.

### 2. 주요 결과

실험 결과, QuantumRWKV는 14개 데이터셋 중 8개에서 고전적 RWKV보다 높은 정확도를 보였다.

- **성능 향상이 뚜렷한 데이터셋**: BloodMNIST, ChestMNIST, RetinaMNIST와 같이 클래스 경계가 모호하거나 노이즈가 많은 의료 이미지에서 특히 우수한 성능을 보였다. (예: RetinaMNIST의 경우 고전 모델 49.25% $\rightarrow$ 양자 모델 53.75%로 크게 상승)
- **비슷하거나 낮은 성능**: MNIST, OrganAMNIST와 같이 구조가 단순하거나 명확한 데이터셋에서는 성능 차이가 거의 없거나 오히려 고전 모델이 약간 우세하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

VQC의 통합이 이미지 분류, 특히 의료 영상 분야에서 실질적인 이점을 제공함이 확인되었다. 저자는 이를 VQC의 강화된 비선형 표현력이 시각적으로 모호한 설정에서 특징 판별력을 높였기 때문이라고 분석한다. 특히 각도 임베딩과 얽힘 층이 고전적인 선형 층이 포착하지 못하는 특징 간의 상관관계를 효과적으로 모델링했을 가능성이 크다.

### 2. 한계 및 비판적 해석

- **범용성의 부재**: 모든 데이터셋에서 성능 향상이 나타나지 않았다는 점은 VQC의 표현력이 항상 일반화 성능 향상으로 이어지지는 않음을 시사한다. 결정 경계가 단순한 경우 고전적 아키텍처만으로도 충분하며, 양자 회로의 도입이 오히려 오버피팅이나 불필요한 복잡성을 초래할 수 있다.
- **연산 효율성**: 양자 회로 시뮬레이션으로 인한 추가적인 계산 오버헤드가 발생한다. 물리적 양자 하드웨어 없이 시뮬레이션에 의존하는 현재 구조로는 실제 대규모 서비스 배포 시 확장성(scalability) 문제가 심각할 수 있다.

## 📌 TL;DR

본 논문은 RWKV 모델의 channel-mixing 층에 Variational Quantum Circuit(VQC)을 통합한 **Vision-QRWKV**를 제안하여 이미지 분류 작업에 적용하였다. 실험 결과, 특히 클래스 구분이 어려운 의료 이미지 데이터셋에서 고전 모델보다 우수한 성능을 보였으며, 이는 양자 계산의 비선형 표현력이 복잡한 시각적 특징 추출에 유리함을 입증한다. 이 연구는 향후 의료 영상 분석 및 경량화된 양자-고전 하이브리드 비전 모델 연구에 중요한 기초 자료가 될 것으로 보인다.
