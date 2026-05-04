# Mamba in Vision: A Comprehensive Survey of Techniques and Applications

Md Maklachur Rahman, Abdullah Aman Tutul, Ankur Nath, Lamyanba Laishram, Soon Ki Jung, Tracy Hammond (202X)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야에서 기존의 주류 아키텍처인 Convolutional Neural Networks (CNNs)와 Vision Transformers (ViTs)가 가진 고유한 한계점을 해결하고자 하는 Mamba 모델의 흐름을 분석한다.

CNN은 국소적 특징(local features) 추출에는 뛰어나지만, 수용 영역(receptive field)의 제한으로 인해 장거리 의존성(long-range dependencies)을 포착하기 위해서는 모델을 매우 깊게 설계해야 하며, 이는 계산 비용의 증가와 효율성 저하로 이어진다. 반면, ViT는 Self-attention 메커니즘을 통해 전역적 관계(global relationships)를 효과적으로 모델링할 수 있으나, 입력 데이터의 길이에 따라 계산 복잡도가 제곱으로 증가하는 $\mathcal{O}(n^2)$의 복잡도를 가지므로 고해상도 이미지나 실시간 애플리케이션에 적용하기에는 비용이 너무 크다는 문제가 있다.

따라서 본 연구의 목표는 선형 계산 복잡도 $\mathcal{O}(n)$를 유지하면서도 장거리 의존성을 효과적으로 포착할 수 있는 Selective Structured State Space Models (Mamba)의 기술적 특성을 분석하고, 이를 컴퓨터 비전의 다양한 태스크에 적용한 사례들을 체계적으로 정리하여 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

첫째, 컴퓨터 비전 분야에서 Mamba 모델의 전반적인 개요를 제공하고, CNN 및 Transformer와의 비교 분석을 통해 Mamba의 독특한 설계 이점을 규명하였다.

둘째, Mamba 기반 모델들을 응용 분야에 따라 분류한 새로운 Taxonomy(분류 체계)를 제시하여 연구자들이 목적에 맞는 모델을 선택할 수 있도록 가이드를 제공하였다.

셋째, Mamba의 핵심 구성 요소인 스캐닝 방법(Scanning methods)의 강점과 약점을 상세히 분석하고, 각 방법이 어떤 유스케이스에 적합한지를 논의하였다.

넷째, Mamba 모델이 현재 직면한 주요 도전 과제들을 정의하고, 이를 해결하기 위한 향후 연구 방향을 제시하였다.

## 📎 Related Works

논문은 기존에 발표된 Mamba 관련 서베이 논문들과의 차별점을 강조한다. 기존 연구들은 일반적인 State Space Models (SSMs)의 이론적 배경이나 의료 영상 분석, 원격 탐사(Remote Sensing)와 같은 특정 도메인에 국한된 분석을 제공하였다.

본 논문은 단순히 모델을 나열하는 것에 그치지 않고, 모델의 분류 체계(Taxonomy), 다양한 스캐닝 기법의 상세 분석, 그리고 CNN 및 Transformer와의 정량적인 비교 분석을 포함함으로써 보다 포괄적이고 실천적인 관점에서 Mamba를 분석하였다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

Mamba 기반 비전 모델의 일반적인 파이프라인은 다음과 같은 단계로 구성된다.

1. **Patching**: 입력 이미지를 작은 패치 단위로 분할한다.
2. **Scanning**: 2D 데이터를 Mamba가 처리할 수 있는 1D 시퀀스로 변환하는 스캐닝 연산을 수행한다.
3. **Mamba Block**: 선형 투영(Linear Projection), 합성곱 층(Convolutional layers), SiLU 활성화 함수, 그리고 SSM 연산을 통해 특징을 추출한다. 필요에 따라 CNN이나 Transformer 블록이 결합된 하이브리드 구조를 취하기도 한다.

### State Space Model (SSM) 기초

SSM은 1차원 입력 시퀀스 $x(t)$를 잠재 상태(latent state) $h(t)$를 거쳐 출력 시퀀스 $y(t)$로 매핑하는 모델이다. 이 과정은 다음과 같은 선형 방정식으로 정의된다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

여기서 $A, B, C$는 상태 전이, 입력 매핑, 출력 매핑을 결정하는 시스템 행렬이다. 디지털 시스템 구현을 위해 이를 이산화(discretization)하면 다음과 같은 식으로 변환된다.

$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

이산화된 시스템의 방정식은 다음과 같다.

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = \bar{C}h_t$$

또한, 전체 시퀀스에 대해 전역 합성곱(global convolution) 연산을 통해 효율적으로 계산할 수 있다.
$$y = \bar{K} \circledast x, \quad \bar{K} = (\bar{C}\bar{B}, \bar{C}\bar{A}\bar{B}, \dots, \bar{C}\bar{A}^{L-1}\bar{B})$$

### Selective State Space Model (Mamba)

Mamba는 기존 SSM의 고정된 파라미터 $A, B$ 대신, 입력 데이터에 따라 동적으로 변하는 **Selective mechanism**을 도입하였다. 즉, 파라미터 $B$와 $C$가 입력 $x$의 함수로 계산되어, 모델이 입력 시퀀스의 중요도에 따라 정보를 선택적으로 필터링할 수 있게 한다. 이를 통해 선형 복잡도를 유지하면서도 Transformer 수준의 문맥 이해 능력을 확보하였다.

### 스캐닝 방법 (Scanning Methods)

Mamba는 본래 1D 시퀀스 모델이므로, 2D 이미지 데이터를 처리하기 위해 이를 1D로 펼치는 스캐닝 전략이 필수적이다.

- **Bidirectional Scanning**: 이미지를 전방 및 후방(가로, 세로)으로 모두 스캔하여 전역 문맥을 포착한다.
- **Cross Scanning**: 수평 및 수직 축을 따라 스캔하여 다방향 공간 정보를 추출한다.
- **Zigzag Scanning**: 지그재그 패턴으로 이동하며 지역적 특징과 전역적 특징의 균형을 맞춘다.
- **Omnidirectional Selective Scanning**: 모든 방향에서 스캔하여 광범위한 공간 관계를 파악한다.
- **Atrous/Efficient Scanning**: 스킵 샘플링(skip-sampling)을 통해 계산 비용을 줄이면서 수용 영역을 유지한다.

## 📊 Results

### 실험 설정 및 지표

본 논문은 ImageNet-1K(분류), COCO(탐지 및 분할), ADE20K(시맨틱 분할), Kinetics-400(비디오 분류), SYSU-CD(원격 탐사 변화 탐지) 등의 벤치마크 데이터셋을 사용하여 Mamba, CNN, Transformer 모델들을 비교하였다. 주요 지표로는 Top-1 Accuracy, $AP_{50}$, mIoU, F1 Score, 파라미터 수, FLOPs 등이 사용되었다.

### 주요 정량적 결과

1. **이미지 분류 (ImageNet-1K)**:
   - 하이브리드 모델인 Heracles-C-L이 가장 높은 성능을 보였으며, 최상위 Transformer 모델인 SwinV2-B보다 Top-1 정확도가 1.3% 높으면서도 파라미터는 38.45%, FLOPs는 11.26% 적게 사용하였다.
2. **객체 탐지 (COCO)**:
   - VMamba-S, LocalVMamba-S 등 Mamba 기반 모델들이 상위권에 랭크되었다. 특히 VMamba-T는 InternImage-B 대비 $AP_{50}$ 차이는 0.8포인트에 불과하지만, 파라미터와 FLOPs를 각각 56.52%, 45.91%나 절감하였다.
3. **시맨틱 분할 (ADE20K)**:
   - VMamba-B는 기존 CNN SOTA인 InternImage-B보다 mIoU(SS)에서 0.2포인트, mIoU(MS)에서 0.3포인트 더 높은 성능을 기록하였다.
4. **비디오 분류 (Kinetics-400)**:
   - VideoMambaPro-M은 Transformer 기반의 TubeVit-H보다 정확도는 약간 낮지만, 파라미터는 89.08%, FLOPs는 73.92%나 적게 사용하여 극도로 높은 효율성을 입증하였다.
5. **원격 탐사 (SYSU-CD)**:
   - ChangeMamba-B가 매우 높은 F1 Score를 기록했으나, 다른 모델들에 비해 파라미터와 FLOPs 소모가 상당히 커서 이 태스크에서는 아직 계산 효율성이 부족함이 드러났다.

## 🧠 Insights & Discussion

### 강점 및 가능성

Mamba는 Transformer의 전역적 수용 영역이라는 장점과 CNN의 효율적인 계산 능력을 동시에 갖춘 대안으로 평가된다. 특히 선형 복잡도 $\mathcal{O}(n)$ 덕분에 초고해상도 이미지, 긴 비디오 시퀀스, 대규모 3D 포인트 클라우드와 같은 데이터셋에서 압도적인 확장성을 가진다. 정량적 분석 결과, 많은 태스크에서 Transformer 수준의 정확도를 유지하면서 계산 자원을 획기적으로 줄일 수 있음을 확인하였다.

### 한계 및 미해결 과제

1. **일반화 능력의 부족**: 숨겨진 상태(hidden states)에 도메인 특화된 정보가 누적되어 새로운 도메인에 대한 적응력이 떨어지는 경향이 있다.
2. **스캐닝 전략의 최적화**: 2D 데이터를 1D로 변환하는 과정에서 공간적 인접성이 훼손될 수 있으며, 최적의 스캐닝 경로를 찾는 표준화된 방법이 아직 부재하다.
3. **사전 학습 모델의 부재**: Transformer에 비해 가용한 대규모 사전 학습 모델이 매우 적어 다운스트림 태스크 적용 시 제약이 많다.
4. **해석 가능성(Interpretability)**: SSM의 복잡한 순차적 특성과 비선형 활성화 함수로 인해 모델의 결정 과정을 추적하기 어려운 '블랙박스' 문제가 존재한다.
5. **보안성**: 특히 파라미터 $B$와 $C$가 적대적 공격(adversarial attacks)에 취약하다는 점이 지적되었다.

## 📌 TL;DR

본 논문은 컴퓨터 비전 분야에서 $\mathcal{O}(n^2)$ 복잡도를 갖는 Transformer와 국소적 수용 영역을 갖는 CNN의 한계를 극복하기 위해 등장한 **Mamba(Selective SSM)** 모델들을 종합적으로 분석한 서베이 보고서이다. Mamba는 선형 복잡도를 통해 고해상도 및 장거리 의존성 데이터 처리에 탁월한 효율성을 보이며, 특히 하이브리드 구조를 통해 정확도와 효율성의 최적점을 찾고 있다. 향후 이 연구는 초고해상도 의료 영상, 실시간 비디오 분석, 대규모 원격 탐사 시스템 등 계산 자원이 제한적이면서 전역적 문맥 파악이 필수적인 분야에서 핵심적인 역할을 할 것으로 기대된다.
