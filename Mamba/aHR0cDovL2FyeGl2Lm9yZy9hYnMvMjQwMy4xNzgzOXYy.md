# ReMamber: Referring Image Segmentation with Mamba Twister

Yuhuan Yang, Chaofan Ma, Jiangchao Yao, Zhun Zhong, Ya Zhang, and Yanfeng Wang (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Referring Image Segmentation (RIS) 작업에서 발생하는 계산 효율성과 멀티모달 융합의 한계이다. RIS는 텍스트 설명에 기반하여 이미지 내의 특정 객체를 식별하고 세그멘테이션하는 작업으로, 시각 정보와 언어 정보 간의 복잡한 상호작용을 이해하는 것이 핵심이다.

기존의 Transformer 기반 모델들은 강력한 Attention 메커니즘을 통해 우수한 성능을 보였으나, 연산 비용과 메모리 사용량이 시퀀스 길이의 제곱에 비례하는 Quadratic complexity $\mathcal{O}(N^2)$ 문제를 가지고 있다. 이는 특히 고해상도 이미지나 긴 텍스트 설명을 처리할 때 자원 소모가 극심해지는 원인이 된다.

최근 등장한 State Space Models (SSMs), 특히 Mamba는 선형 복잡도 $\mathcal{O}(N)$로 효율적인 처리가 가능하여 대안으로 주목받고 있다. 그러나 Mamba를 멀티모달 작업에 직접 적용할 경우, 서로 다른 토큰 간의 채널 상호작용(Channel interaction)이 부족하여 이미지와 텍스트 데이터를 효과적으로 융합하지 못하는 문제가 발생한다. 따라서 본 논문의 목표는 Mamba의 효율성을 유지하면서도 이미지-텍스트 간의 강력한 융합을 가능케 하는 새로운 RIS 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba를 RIS 분야에 최초로 탐색하고, 이를 위해 **Mamba Twister**라는 새로운 블록을 설계하여 멀티모달 융합 성능을 극대화한 점이다.

중심적인 아이디어는 **Twisting Mechanism**이다. 이는 단순히 텍스트 토큰을 이미지 토큰 앞에 붙이는 기존의 splicing 방식 대신, 채널(Channel)과 공간(Spatial) 차원을 순차적으로 스캔하여 서로 다른 모달리티의 정보가 서로 얽히도록(twist) 만드는 방식이다. 이를 통해 Mamba의 고질적인 문제인 채널 간 상호작용 부족을 해결하고, 이미지의 각 공간적 위치에서 텍스트 문맥을 정밀하게 인식할 수 있도록 설계하였다.

## 📎 Related Works

### Referring Image Segmentation (RIS)

초기 RIS 연구들은 RNN/LSTM으로 언어를 인코딩하고 CNN으로 시각 특징을 추출한 뒤, Multi-modal LSTM이나 Attention 메커니즘을 통해 융합하였다. 이후 Transformer 기반의 모델들이 등장하며 MDETR, LAVT, ReSTR 등이 제안되었고, 이들은 Transformer Decoder나 Encoder-Decoder 구조를 통해 시각-언어 특징을 포괄적으로 융합하며 성능을 높였다. 하지만 앞서 언급한 대로 Quadratic complexity로 인한 효율성 문제가 여전히 존재한다.

### State Space Models (SSMs) and Visual Applications

SSM은 제어 이론에서 유래하여 긴 시퀀스의 의존성을 모델링하는 데 강점이 있다. S4, S5, H3를 거쳐 최근의 Mamba는 선택 메커니즘(Selection mechanism)과 하드웨어 최적화 알고리즘을 통해 Transformer에 필적하는 성능과 선형 복잡도를 달성하였다. 현재 Vim, VMamba 등이 이미지 분류 및 세그멘테이션 등 단일 모달리티 작업에서 가능성을 보였으나, RIS와 같은 멀티모달 작업에 Mamba를 적용한 연구는 본 논문이 처음이다.

## 🛠️ Methodology

### 전체 시스템 구조

ReMamber는 여러 개의 **Mamba Twister Block**으로 구성된 인코더와, 최종 마스크를 생성하는 Flexible Decoder로 이루어져 있다. 각 Mamba Twister Block은 시각적 특징을 처리하는 **VSS (Visual State Space) Layer**들과 이미지-텍스트 융합을 담당하는 **Twisting Layer**로 구성된다.

### VSS Layer (Visual State Space Layer)

Mamba는 기본적으로 1차원 시퀀스 데이터를 처리하도록 설계되었으므로, 2차원 이미지 데이터를 처리하기 위해 VMamba의 Cross-Scan-Module (CSM)을 채택하였다. VSS Layer는 이미지 패치를 네 가지 서로 다른 방향으로 스캔하여 모든 픽셀의 정보가 통합되도록 하여 공간적 관계를 학습한다.

### Twisting Layer (멀티모달 융합 핵심)

Twisting Layer는 텍스트 조건을 이미지 특징에 주입하여 이미지 특징의 변환을 가이드한다. 과정은 다음과 같다.

**1. Vision-Language Interaction (상호작용 계산)**
이미지와 텍스트 간의 대응 관계를 명시적으로 구축하기 위해 두 가지 상호작용을 수행한다.

- **Global Interaction**: 텍스트 시퀀스 전체를 대표하는 글로벌 표현 $F_{CLS}^{t}$를 추출하고, 이를 이미지 크기와 동일하게 확장하여 $\tilde{F}_{t} \in \mathbb{R}^{h \times w \times C_{t}}$를 생성한다.
- **Local Interaction**: 이미지 특징 $F_{i}$와 텍스트 특징 $F_{t}$ 간의 행렬 곱을 통해 세밀한 상관관계 맵 $F_{c}$를 계산한다.
  $$F_{c} = F_{i} W_{i} \cdot (F_{t} W_{t})^{T} \in \mathbb{R}^{h \times w \times L}$$
  이후 컨볼루션 레이어를 통해 $\tilde{F}_{c} \in \mathbb{R}^{h \times w \times C_{c}}$로 변환한다.

**2. Hybrid Feature Cube 생성**
추출된 시각 특징($F_{i}$), 글로벌 텍스트 특징($\tilde{F}_{t}$), 로컬 상호작용 특징($\tilde{F}_{c}$)을 채널 차원으로 결합(Concatenation)하여 하이브리드 특징 큐브를 형성한다.
$$F_{cube} = [F_{i}, \tilde{F}_{t}, \tilde{F}_{c}] \in \mathbb{R}^{h \times w \times (C_{i} + C_{t} + C_{c})}$$

**3. Twisting Mechanism (큐브 비틀기)**
채널 차원에 배치된 서로 다른 모달리티 정보들이 서로 소통할 수 있도록 두 단계의 SSM 스캔을 수행한다.

- **Channel Scan**: 채널을 하나의 순서 있는 시퀀스로 취급하여 1D Selective Scan을 수행한다. 이를 통해 모달리티 간의 융합이 촉진된다.
- **Spatial Scan**: VSS Layer를 통해 공간 차원을 스캔하며, 각 모달리티 내의 패치 간 정보를 통합한다.
  $$F_{out} = \text{SSM}_{\text{spatial}}(\text{SSM}_{\text{channel}}(F_{cube}))$$

### 학습 절차

ReMamber는 일반적인 멀티모달 융합 프레임워크이므로 특정 디코더나 손실 함수에 종속되지 않는다. 본 논문에서는 단순한 Convolution-based decoder를 사용하였으며, 전체 네트워크를 End-to-End 방식으로 학습하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: RefCOCO, RefCOCO+, G-Ref 세 가지 벤치마크를 사용하였다.
- **평가 지표**: mIoU, oIoU 및 Precision@X ($X \in \{50, 60, 70, 90\}$)를 사용하였다.
- **구현 세부사항**: ImageNet으로 사전 학습된 가중치를 초기값으로 사용하였으며, 비교 실험을 위해 이미지 해상도를 480으로 설정하였다.

### 주요 결과

- **SOTA 비교**: ReMamber는 모든 데이터셋에서 기존의 SOTA 모델들을 능가하였다. 특히 Swin-Transformer 기반의 LAVT보다 월등한 성능을 보였는데, 이는 Mamba 기반 아키텍처가 세그멘테이션 작업에서 매우 강력하며 Mamba Twister의 융합 효율성이 높음을 입증한다.
- **융합 방식 변형 비교**: In-context Conditioning, Attention-based Conditioning, Norm Adaptation 등 다른 융합 방식과 비교했을 때 Mamba Twister가 모든 지표에서 가장 우수한 성능을 기록하였다.
- **효율성 분석**: 고해상도(예: 1,024) 설정에서 ReMamber는 LAVT보다 추론 속도(FPS)가 훨씬 빠르며 GPU 메모리 사용량 또한 현저히 낮음을 확인하였다.

## 🧠 Insights & Discussion

### Mamba와 Cross-Attention의 부조화

본 연구에서 흥미로운 점은 Transformer에서 널리 쓰이는 Cross-Attention 방식이 Mamba 아키텍처에서는 매우 낮은 성능을 보였다는 것이다. 저자들은 그 이유를 Mamba의 **순차적 의존성(Sequential dependencies)**과 Attention의 **병렬적 처리(Equal treatment)** 간의 근본적인 차이에서 찾는다. Mamba는 이전 상태가 다음 상태를 결정하는 엄격한 순서를 가지는 반면, Attention은 모든 토큰을 동일하게 처리하므로 Mamba가 구조적으로 시퀀스를 모델링하는 능력을 저해할 수 있다는 분석이다.

### Twisting Mechanism의 효과

PCA를 이용한 데이터 분포 시각화 결과, Channel Scan은 서로 다른 모달리티를 텍스트 분포 쪽으로 응집시키는 경향이 있고, Spatial Scan은 이를 다시 재분배하여 두 모달리티의 결합된 영향력이 반영된 분포로 만든다는 것을 확인하였다. 이는 채널과 공간 스캔의 순차적 적용이 단순한 합산보다 훨씬 정교한 융합을 가능케 함을 시사한다.

### 한계 및 비판적 해석

현재 모델의 디코더가 매우 단순한 컨볼루션 레이어로만 구성되어 있어, 더 정교한 Mamba 기반 디코더를 설계한다면 추가적인 성능 향상이 가능할 것으로 보인다. 또한, Cross-Attention과의 결합이 좋지 않다는 점은 Mamba 기반 멀티모달 모델 설계 시 기존 Transformer의 기법을 그대로 가져오는 것이 위험할 수 있음을 보여준다.

## 📌 TL;DR

본 논문은 Mamba의 선형 복잡도를 활용하면서도 멀티모달 융합 능력을 극대화한 **ReMamber** 아키텍처를 제안한다. 핵심인 **Mamba Twister** 블록은 글로벌/로컬 상호작용을 통해 하이브리드 특징 큐브를 만들고, 이를 **Channel Scan $\to$ Spatial Scan** 순으로 처리하는 'Twisting' 과정을 통해 효율적이고 강력한 시각-언어 융합을 달성한다. 실험적으로는 기존 SOTA 모델인 LAVT 등을 능가하는 성능과 압도적인 연산 효율성을 입증하였으며, 이는 향후 고해상도 멀티모달 이해 작업에서 Mamba 아키텍처가 중요한 역할을 할 가능성을 보여준다.
