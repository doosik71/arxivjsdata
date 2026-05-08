# TP-UNet: Temporal Prompt Guided UNet for Medical Image Segmentation

Ranmin Wang, Limin Zhuang, Hongkun Chen, Boyan Xu, and Ruichu Cai (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 기존의 UNet 기반 의료 영상 분할(Medical Image Segmentation) 모델들이 스캔된 영상 내 장기들의 **시간적 순서(Temporal Order)** 정보를 무시한다는 점이다. CT나 MRI와 같은 의료 영상은 일반적으로 신체 상단에서 하단으로 순차적으로 촬영되며, 이에 따라 각 장기가 나타나는 위치와 순서가 어느 정도 정해져 있다.

예를 들어, 위(stomach), 소장(small intestine), 대장(large intestine)은 특정 시간 간격 내에서 정규 분포를 따르는 경향이 있으며, 일반적으로 $\mu_{stomach} \le \mu_{small} \le \mu_{large}$의 순서로 나타난다. 이러한 시간적 정보는 모델이 현재 슬라이스에서 어떤 장기에 더 집중해야 하는지를 알려주는 강력한 가이드가 될 수 있음에도 불구하고, 기존의 UNet 구조는 이러한 순차적 정보를 통합할 수 있는 직접적인 능력이 부족하다. 따라서 본 연구의 목표는 **Temporal Prompt(시간적 프롬프트)**를 도입하여 UNet이 영상의 시간적 정보를 학습하게 함으로써 분할 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 텍스트 기반의 프롬프트를 통해 의료 영상의 시간적 정보를 모델에 주입하는 것이다. 주요 기여 사항은 다음과 같다.

1. **TP-UNet 프레임워크 제안**: 의료 영상의 시간적 정보를 텍스트 프롬프트 형태로 변환하여 UNet의 학습을 가이드하는 단순하면서도 효과적인 구조를 제안하였다.
2. **시맨틱 정렬 및 모달 융합(Semantic Alignment & Modal Fusion)**: 서로 다른 도메인에서 오는 텍스트 임베딩과 이미지 특징 맵 사이의 간극(Semantic Gap)을 줄이기 위해 비지도 대조 학습(Unsupervised Contrastive Learning) 기반의 시맨틱 정렬 단계를 도입하고, 이후 Cross-Attention 메커니즘을 통해 두 정보를 효과적으로 통합하였다.
3. **SOTA 성능 입증**: LITS 2017 및 UW-Madison 데이터셋에 대한 광범위한 실험을 통해 기존의 최신 모델(Swin UNet 등)보다 우수한 성능을 달성함을 증명하였다.

## 📎 Related Works

### Prompt Learning

최근 NLP 분야에서 시작된 프롬프트 학습이 의료 영상 분할로 확장되고 있다. 기존 연구들에서는 분할 대상 장기의 이름이나 장기의 기능, 모양, 외형에 대한 텍스트 설명을 프롬프트로 사용하여 성능을 향상시켰다. 그러나 이러한 접근 방식들은 영상 내에서 장기가 나타나는 순서나 위치와 같은 '시간적 정보'를 활용하지 않았다는 한계가 있다.

### Multimodal Contrastive Learning

텍스트와 이미지 등 서로 다른 모달리티의 표현을 공유 임베딩 공간으로 정렬하는 대조 학습 기법(예: ConVIRT, GLoRIA)은 의료 도메인에서 데이터 효율성을 높이고 하위 작업의 성능을 개선하는 데 사용되어 왔다. 본 논문은 이러한 기법을 차용하여 시간적 프롬프트와 이미지 특징 간의 시맨틱 간극을 좁히는 데 활용하였다.

## 🛠️ Methodology

### 1. Temporal Prompt

시간적 정보는 전체 슬라이스 수 $N$에 대해 현재 슬라이스의 위치 $i$를 $[0, 1]$ 구간의 값($N^{th}_{i/N}$)으로 매핑하여 표현한다. 프롬프트 템플릿은 다음과 같이 정의된다:
`"This is {an MRI / a CT} of the {organ} with a segmentation period of {Nth i/N}."`
이 프롬프트는 의료 영상의 종류, 대상 장기, 그리고 촬영 순서 정보를 포함하며, 추론 시에는 의료진이 선택한 특정 범위의 슬라이스에 대해서만 자동으로 생성되어 효율성을 높인다.

### 2. Multimodal Encoder

- **Text Encoder**: 두 가지 아키텍처를 실험하였다.
  - **CLIP**: 일반 자연어에 강점이 있으나 의료 도메인 격차를 줄이기 위해 LoRA(Low-Rank Adaptation)를 이용한 PEFT(Parameter-Efficient Fine-Tuning)를 적용하였다.
  - **Electra**: 사전 학습된 모델에 SFT(Supervised Fine-Tuning)를 적용하였다.
- **Image Encoder**: 표준 UNet 구조를 사용하여 저수준 시맨틱 정보를 추출한다.

### 3. Semantic Align

이미지 특징 $F_m \in \mathbb{R}^{B \times C \times H \times W}$와 텍스트 특징 $F_t \in \mathbb{R}^{B \times L \times D}$는 서로 다른 세만틱 공간에 존재한다. 이를 정렬하기 위해 다음과 같은 대조 손실 함수를 사용한다.

이미지-텍스트 대조 손실:
$$\ell(F_m \to F_t)_i = -\log \frac{\exp(h_{F_{m_i}, F_{t_i}} / \tau)}{\sum_{k=1}^{N} \exp(h_{F_{m_i}, F_{t_k}} / \tau)}$$

텍스트-이미지 대조 손실:
$$\ell(F_t \to F_m)_i = -\log \frac{\exp(h_{F_{t_i}, F_{m_k}} / \tau)}{\sum_{k=1}^{N} \exp(h_{F_{t_i}, F_{m_k}} / \tau)}$$

여기서 코사인 유사도 $h$는 다음과 같이 정의된다:
$$h_{F_{m_i}, F_{t_i}} = \frac{F_{m_i}^\top F_{t_i}}{\|F_{m_i}\| \|F_{t_i}\|}$$

최종 최적화 대상인 대조 손실 함수는 다음과 같다:
$$L_{contrastive} = \frac{1}{N} \sum_{i=1}^{N} (\lambda \ell(F_m \to F_t)_i + (1-\lambda) \ell(F_t \to F_m)_i)$$

### 4. Modality Fusion

정렬된 특징들을 통합하기 위해 Cross-Attention 메커니즘을 사용한다:
$$F = \text{softmax} \left( \frac{([F'_m; F'_t]W_Q ([F'_m; F'_t]W_K)^\top)}{\sqrt{d_k}} \right) [F'_m; F'_t]W_V$$
여기서 $[;]$는 연결(concatenation) 연산을 의미하며, $F'_m$과 $F'_t$는 각각 이미지와 텍스트의 투영된 특징이다. 생성된 특징 맵 $F$는 UNet의 첫 번째 레벨 스킵 연결 특징 맵과 결합되어 최종 분할 마스크를 생성하는 디코더로 전달된다.

## 📊 Results

### 실험 설정

- **데이터셋**: UW-Madison(결장 및 위 MRI 영상), LITS 2017(간 CT 영상)
- **평가 지표**: Dice coefficient 및 Jaccard coefficient
- **비교 대상**: UNet, UNet++, Attention UNet, scSE UNet, Trans UNet, Swin UNet

### 주요 결과

- **UW-Madison 데이터셋**: TP-UNet(CLIP 기반)이 모든 장기 카테고리에서 최고 성능을 보였다. UNet 대비 Dice score가 평균 $4.44\%$ 향상되었으며, 특히 소장(Small Intestine)에서 $5.32\%$의 가장 큰 폭의 향상을 보였다. SOTA 모델인 Swin UNet보다도 Dice score 기준 $1.3\%$ 더 높은 성능을 기록하였다.
- **LITS 2017 데이터셋**: 간 분할 작업에서 UNet 대비 Dice score가 $6.08\%$ 향상되었으며, 기존 SOTA 모델 대비 Dice score가 $9.21\%$ 향상되는 압도적인 결과를 보여주었다.

### Ablation Study

- **시간 정보 제거**: 프롬프트에서 타임스탬프를 제거했을 때 UW-Madison 데이터셋의 mDice score가 $2.1\%$ 감소하였다.
- **전체 프롬프트 제거**: 텍스트 인코더와 프롬프트를 모두 제거했을 때 LITS 데이터셋의 mDice score가 $5.36\%$ 크게 감소하였다.
- **시맨틱 정렬 제거**: 정렬 단계 없이 직접 융합했을 때 mDice score가 $1.01\%$ 감소하여, 모달리티 간 간극을 줄이는 과정이 필수적임을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상의 물리적 특성인 '촬영 순서'를 텍스트 프롬프트라는 현대적인 딥러닝 기법과 결합하여 성공적으로 성능을 끌어올렸다. 특히 단순한 결합이 아니라 **대조 학습(Contrastive Learning)을 통한 시맨틱 정렬 $\rightarrow$ Cross-Attention을 통한 융합**이라는 2단계 파이프라인을 구축함으로써, 서로 다른 성격의 데이터(이미지-텍스트)가 가진 정보를 효율적으로 통합하였다.

다만, 논문에서 언급한 장기들의 출현 확률이 정규 분포를 따른다는 가정은 일종의 휴리스틱한 접근이며, 환자 개개인의 해부학적 변이에 따라 이 분포가 달라질 수 있다는 점에 대한 심층적인 분석은 부족하다. 그럼에도 불구하고, 텍스트 프롬프트를 통해 모델에 일종의 '위치 사전 지식(Spatial Prior)'을 주입하는 방식은 향후 다른 순차적 의료 영상 분석에도 응용될 가능성이 매우 높다.

## 📌 TL;DR

TP-UNet은 의료 영상의 슬라이스 순서(시간적 정보)를 텍스트 프롬프트로 변환하여 UNet에 주입하는 프레임워크이다. 대조 학습을 통한 시맨틱 정렬과 Cross-Attention 기반의 융합 과정을 통해 이미지와 텍스트 정보를 통합하며, 이를 통해 UW-Madison 및 LITS 2017 데이터셋에서 기존 SOTA 모델들을 뛰어넘는 성능을 달성하였다. 이 연구는 의료 영상 분할에서 해부학적 순서 정보가 매우 중요한 단서가 될 수 있음을 시사한다.
