# A Mamba-based Siamese Network for Remote Sensing Change Detection

Jay N. Paranjape, Celso de Melo, Vishal M. Patel (2024)

## 🧩 Problem to Solve

본 논문은 원격 탐사(Remote Sensing) 이미지에서의 변화 탐지(Change Detection, CD) 문제를 해결하고자 한다. 변화 탐지는 서로 다른 시점에 촬영된 동일 지역의 이미지 쌍을 분석하여 지표면의 유의미한 변화를 식별하는 작업으로, 환경 모니터링, 도시 계획, 재난 평가 및 군사적 용도 등 광범위한 분야에서 필수적인 도구로 활용된다.

그러나 실제 원격 탐사 환경에서는 조명 변화(illumination changes), 해상도 차이(varying resolutions), 노이즈(noise), 그리고 정렬 오류(registration errors)와 같은 데이터 불일치 문제가 빈번하게 발생한다. 이러한 요인들은 단순한 픽셀 차이 분석을 어렵게 만들며, 결과적으로 모델이 단순한 노이즈가 아닌 '유의미한 변화'만을 정확하게 세그멘테이션해야 하는 복잡한 도전 과제를 제기한다. 본 연구의 목표는 이러한 데이터 불일치에 강건하면서도, 전역적인 문맥(global context)을 효과적으로 포착하여 정밀한 변화 마스크를 생성하는 Mamba 기반의 새로운 네트워크인 M-CD를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최근 시퀀스 모델링에서 뛰어난 성능을 보이는 Selective State Space Model(Mamba)을 원격 탐사 변화 탐지 작업에 최적화하여 적용하는 것이다. 주요 기여 사항은 다음과 같다.

첫째, VMamba 아키텍처를 기반으로 한 Siamese Image Encoder(SIE)를 구축하여 두 입력 이미지로부터 풍부한 특징을 추출한다. 
둘째, 두 이미지의 특징을 다중 스케일에서 결합하여 시간적 차이를 효과적으로 학습하는 전용 Difference Module(DM)을 설계하였다.
셋째, 채널 간의 의존성을 학습할 수 있도록 개선된 Channel-Averaged VSS(CAVSS) 블록을 포함한 Mask Decoder(MD)를 제안하여, 공간적 정보와 채널 정보를 동시에 고려한 최종 변화 마스크를 생성한다.

## 📎 Related Works

### 기존 연구 및 한계
1. **전통적 방법**: 대수적 방법(Algebraic), 변환 기반 방법(Transformation-based), 분류 기반 방법(Classification-based) 등이 존재한다. 이들은 구현이 간단하지만, 복잡한 지표면 변화를 모델링하기에 부족하며 임계값(threshold) 설정에 지나치게 의존한다는 한계가 있다.
2. **딥러닝 기반 방법**: CNN 기반의 Siamese 네트워크(예: FC-Siam-Conc, SNUNet)는 특징 추출 능력을 향상시켰으나, 수용 영역(receptive field)의 한계로 인해 전역적인 문맥 파악이 어렵다. 이를 해결하기 위해 Transformer 기반 모델(예: BIT, ChangeFormer)이 도입되어 장거리 의존성(long-range dependencies)을 포착하기 시작했다.
3. **자기지도학습 및 확산 모델**: 대규모 데이터로 사전 학습된 모델이나 DDPM-CD와 같은 확산 모델(Diffusion Model) 기반 방식이 현재 SOTA 성능을 보이고 있으나, 이는 막대한 양의 사전 학습 데이터와 계산 자원을 필요로 한다는 단점이 있다.
4. **Mamba 기반 방법**: 최근 Mamba를 백본으로 사용하는 연구(예: ChangeMamba, RSMamba)가 등장했으나, 대부분은 기존 블록을 단순 교체하는 수준에 그쳤다.

### 차별점
M-CD는 단순한 블록 교체를 넘어, 변화 탐지라는 특수한 목적에 맞게 설계된 **Difference Module**과 **채널 인지형 Decoder**를 도입함으로써 시간적/공간적 관계를 더욱 정밀하게 모델링한다.

## 🛠️ Methodology

### 1. State Space Modeling (SSM) 기초
M-CD의 기반이 되는 SSM은 입력 $x(t)$에 대해 다음과 같은 연속 시스템 방정식으로 정의된다.

$$y(t) = Ch(t) + Dx(t)$$
$$\dot{h}(t) = Ah(t) + Bx(t)$$

여기서 $h(t)$는 은닉 상태(hidden state)이며, $A, B, C, D$는 학습 가능한 파라미터이다. 이산 시간 시퀀스(이미지 등)를 처리하기 위해 Zero Order Hold Discretization을 사용하여 다음과 같이 이산화한다.

$$y_k = Ch_k + Dx_k, \quad h_k = \bar{A}h_{k-1} + \bar{B}x_k$$

Mamba는 여기서 $\bar{B}, C, \Delta$를 입력 $x$에 의존하게 만들어(Selective Scan), 더 복잡한 관계를 효율적으로 학습할 수 있게 한다.

### 2. M-CD 전체 아키텍처
M-CD는 **Siamese Image Encoder (SIE) $\rightarrow$ Difference Module (DM) $\rightarrow$ Mask Decoder (MD)**의 세 단계로 구성된다.

#### 2.1 Siamese Image Encoder (SIE)
두 이미지(pre-change, post-change)를 동일한 가중치를 공유하는 두 개의 브랜치에 입력한다. 
- **구조**: Stem 모듈(2D Conv) 이후 4개의 Visual State Space(VSS) 블록과 다운샘플링 모듈이 배치된다.
- **SS2D Module**: VMamba의 전략을 따라 이미지를 4가지 방향(top-left $\rightarrow$ bottom-right 등)으로 스캔하여 전방향의 장거리 의존성을 추출한다.

#### 2.2 Difference Module (DM)
인코더에서 추출된 두 이미지의 다중 스케일 특징을 입력받아 결합된 특징 벡터를 생성한다.
- **Joint Selective Scan (JSS)**: 네트워크의 대칭성을 유지하고 학습을 돕기 위해, 특징 벡터를 두 가지 순서($P_{re};P_{ost}$ 및 $P_{ost};P_{re}$)로 연결(concatenation)하여 스캔 연산을 수행하고 그 결과를 합산한다.
- **역할**: 단순한 뺄셈이 아니라, 스캔 연산을 통해 두 이미지 간의 유의미한 시간적 차이를 학습한다.

#### 2.3 Mask Decoder (MD)
DM의 출력을 받아 최종 세그멘테이션 마스크를 생성한다.
- **CAVSS Block**: VSS 블록은 전역 문맥은 잘 잡지만 채널 간 의존성 학습이 부족하다. 이를 보완하기 위해 Average-pool 및 Max-pool 연산을 채널 차원에 적용하여 spatial context와 channel context를 동시에 학습한다.
- **구조**: UNet과 유사한 스킵 연결(skip connection) 구조를 가져, 저해상도의 전역 정보와 고해상도의 지역 정보를 결합하여 정밀한 경계를 복원한다.

## 📊 Results

### 실험 설정
- **데이터셋**: WHU-CD, DSIFN-CD, LEVIR-CD, CDD 등 4개의 공개 데이터셋을 사용하였다.
- **지표**: F1-score, IoU(Intersection-Over-Union), OA(Overall Accuracy)를 측정하였다.
- **비교 대상**: CNN 기반, Transformer 기반, 자기지도학습 기반, 확산 모델(DDPM-CD) 기반 및 기존 Mamba 기반 방법론들과 비교하였다.

### 정량적 결과
M-CD는 모든 데이터셋에서 기존 SOTA 모델들을 능가하는 성능을 보였다.
- **IoU 기준**: CNN 기반 모델 대비 약 $7\text{--}10\%$, Transformer 기반 모델 대비 약 $5\%$ 향상된 성능을 보였다.
- **SOTA 대비**: 현재 최고 성능인 DDPM-CD 및 다른 Mamba 기반 모델들보다 IoU 기준 약 $3\%$ 더 높은 성능을 기록하였다.
- 특히, DDPM-CD와 달리 방대한 양의 원격 탐사 이미지 사전 학습 없이 ImageNet-1k 가중치만으로 초기화하여 달성한 결과라는 점이 중요하다.

### 정성적 결과 및 분석
- **시각적 품질**: DDPM-CD의 결과물에서 나타나는 아티팩트(artefacts)가 M-CD에서는 관찰되지 않았으며, 더 깨끗하고 정밀한 경계선을 생성함을 확인하였다.
- **Effective Receptive Field (ERF)**: ERF 분석 결과, M-CD는 CNN보다 훨씬 전역적인 수용 영역을 가지며, Transformer의 균일한(uniform) 어텐션 맵보다 더 구조화된(structured) 형태의 수용 영역을 가져 중요한 영역을 더 잘 우선순위화함을 보였다.

## 🧠 Insights & Discussion

### 강점
M-CD는 Mamba 아키텍처의 선형 시간 복잡도와 넓은 수용 영역이라는 장점을 변화 탐지 작업에 성공적으로 이식하였다. 특히, 전용 Difference Module과 채널 인지형 Decoder를 설계함으로써, 단순한 백본 교체 이상의 성능 향상을 이끌어냈다. 또한, 확산 모델과 같은 무거운 사전 학습 없이도 SOTA 성능을 달성했다는 점에서 효율성이 높다.

### 한계 및 비판적 해석
- **연산 비용**: computational complexity 분석 결과, M-CD는 다른 SOTA 모델들에 비해 학습 가능한 파라미터 수가 많고 추론 시간(inference time)이 더 길다. (표 4 참조) 
- **Trade-off**: 비록 GFLOPS는 다른 모델들과 비슷하거나 DDPM-CD보다 낮지만, 절대적인 추론 속도가 느리다는 점은 실시간 응용 분야에서 제약이 될 수 있다. 하지만 저자들은 추가적인 사전 학습 데이터 없이 얻은 성능 향상이 이러한 계산 비용의 증가를 정당화한다고 주장한다.

## 📌 TL;DR

본 논문은 원격 탐사 이미지의 변화 탐지를 위해 Mamba(Selective State Space Model) 기반의 **M-CD** 네트워크를 제안한다. VMamba 기반의 Siamese 인코더, 시간적 차이를 학습하는 전용 Difference Module, 그리고 채널 정보를 통합하는 Mask Decoder를 통해 기존 CNN, Transformer, 심지어 최신 확산 모델(DDPM-CD)보다 우수한 세그멘테이션 성능을 달성하였다. 이 연구는 Mamba 아키텍처가 원격 탐사와 같은 고해상도 이미지 분석 작업에서 매우 강력한 전역 문맥 포착 능력을 제공함을 입증하였으며, 향후 효율적인 전역 모델링 기반의 원격 탐사 연구에 중요한 기준점이 될 것으로 보인다.