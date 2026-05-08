# EgoSurgery-Phase: A Dataset of Surgical Phase Recognition from Egocentric Open Surgery Videos

Ryo Fujii, Masashi Hatano, Hideo Saito, and Hiroki Kajita (2024)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 개복 수술(Open Surgery) 환경에서의 수술 단계 인식(Surgical Phase Recognition)을 위한 데이터셋의 부족과 그로 인한 모델 성능의 한계이다. 수술 단계 인식은 실시간 수술 보조, 교육, 치료 평가 등 현대 수술실의 다양한 요구를 충족시킬 수 있는 중요한 기술이다. 하지만 기존의 연구들은 대부분 최소 침습 수술(Minimally Invasive Surgery, MIS)에 집중되어 있으며, 개복 수술에 대한 연구는 상대적으로 매우 부족한 실정이다.

이러한 불균형은 공개적으로 사용할 수 있는 대규모 개복 수술 비디오 데이터셋이 없다는 점에 기인한다. 또한, 개복 수술 영상은 수술용 조명의 강한 빛으로 인해 수술 영역 외곽이 검게 처리되는 '블랙 클리핑(Black Clipping)' 현상이 빈번하게 발생한다. 이로 인해 기존의 Masked Autoencoder(MAE) 방식에서 사용하는 무작위 마스킹(Random Masking) 전략을 적용할 경우, 정보량이 적은 영역이 마스킹될 가능성이 높아져 효율적인 표현 학습이 어렵다는 문제가 존재한다. 따라서 본 논문은 개복 수술 전용 데이터셋을 구축하고, 시선(Gaze) 정보를 활용해 효율적으로 학습할 수 있는 새로운 방법론을 제시하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 주요 기여 사항은 다음과 같다.

첫째, 최초로 공개 가능한 대규모 실측 개복 수술 egocentric 비디오 데이터셋인 **EgoSurgery-Phase**를 구축하였다. 이 데이터셋은 수술자의 머리에 장착된 카메라로 촬영되었으며, 단순 영상뿐만 아니라 수술자의 시선(Eye Gaze) 데이터를 함께 제공함으로써 향후 연구에 풍부한 정보를 제공한다.

둘째, 시선 정보를 활용한 마스킹 전략을 도입한 **Gaze-Guided Masked Autoencoder (GGMAE)**를 제안하였다. 이는 수술자가 응시하는 영역이 수술 단계 인식에 있어 핵심적인 의미를 가진다는 직관에서 출발하였다. 무작위로 토큰을 제거하는 대신, 시선 데이터가 집중된 '의미적으로 풍부한 영역'을 우선적으로 마스킹하고 이를 복원하도록 유도함으로써, 모델이 수술의 핵심 영역에 더 집중하여 유의미한 특징을 학습하도록 설계하였다.

## 📎 Related Works

기존의 수술 단계 인식 연구들은 주로 복강경 수술과 같은 MIS 데이터셋을 기반으로 발전해 왔으며, TeCNO, Trans-SVNet, NETE 등 다양한 딥러닝 아키텍처가 제안되었다. 하지만 이러한 모델들은 개복 수술 특유의 시각적 특성(예: 조명으로 인한 정보 손실)을 고려하지 않았다.

최근 비디오 이해 분야에서는 Masked Autoencoders (MAE) 기반의 자가 지도 학습(Self-supervised Learning)이 큰 성공을 거두었으며, 이를 수술 영상에 적용한 SurgMAE와 같은 시도가 있었다. 그러나 기존 MAE 기반 방식들은 대개 무작위 마스킹(Random Masking)이나 튜브 마스킹(Tube Masking)을 사용한다. 본 논문은 개복 수술 영상의 상당 부분이 비정보적(non-informative)이라는 점을 지적하며, 시선 정보를 통해 마스킹 위치를 결정하는 가이드 방식이 기존의 일반적인 마스킹 전략보다 더 효과적임을 강조하며 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

GGMAE는 입력 비디오 $V \in \mathbb{R}^{T \times C \times H \times W}$와 시선 히트맵 $G \in \mathbb{R}^{T \times H \times W}$를 입력으로 받는다. 전체 파이프라인은 Space-time cube embedding을 통해 비디오를 토큰으로 변환한 뒤, 제안된 Gaze-Guided Masking (GGM) 모듈을 통해 특정 토큰을 마스킹하고, Transformer Encoder-Decoder 구조를 통해 마스킹된 픽셀을 복원하는 방식으로 구성된다.

### Gaze-Guided Masking (GGM)

개복 수술 영상의 무의미한 영역을 배제하고 학습 효율을 높이기 위해, 시선 정보를 기반으로 비균등(non-uniform)하게 토큰을 샘플링한다. 상세 절차는 다음과 같다.

1. **토큰별 시선 값 계산**: 각 토큰 $x_i$에 해당하는 시선 히트맵의 픽셀 값들을 합산하여 누적 시선 값 $d_i$를 계산한다.
   $$d_i = \sum_{j \in \Omega_i} G_{i,j}$$
   여기서 $\Omega_i$는 $i$번째 토큰에 포함된 픽셀들의 집합이다.

2. **마스킹 확률 분포 생성**: 계산된 $d_i$를 Softmax 함수에 통과시켜 각 토큰이 마스킹될 확률 $\pi_t$를 구한다. 이때 하이퍼파라미터 $\tau$는 분포의 날카로운 정도(sharpness)를 조절한다.
   $$\pi_t = \text{Softmax}(d_t / \tau)$$

3. **토큰 샘플링**: 구해진 확률 $\pi_t$를 기반으로 다항 분포(Multinomial distribution)에서 샘플링하여 마스킹할 토큰을 결정한다. 결과적으로 수술자가 집중해서 보는 영역의 토큰이 더 높은 확률로 마스킹되며, 모델은 이 중요한 영역을 복원하기 위해 더 노력하게 된다.

### 학습 목표 및 손실 함수

모델의 학습 목표는 마스킹된 영역의 원래 픽셀 값을 정확하게 복원하는 것이다. 이를 위해 입력 픽셀 값 $I(p)$와 복원된 픽셀 값 $\hat{I}(p)$ 사이의 평균 제곱 오차(Mean Squared Error, MSE)를 손실 함수로 사용한다.
$$L = \frac{1}{|\Omega|} \sum_{p \in \Omega} |I(p) - \hat{I}(p)|^2$$
여기서 $\Omega$는 마스킹된 토큰들의 집합을 의미한다.

### 학습 및 추론 절차

- **Pre-training**: ViT-Small 백본을 사용하여 90%의 높은 마스킹 비율로 800 에포크 동안 사전 학습을 진행한다. $\tau$ 값은 0.5로 설정하였다.
- **Fine-tuning**: 사전 학습된 백본 위에 MLP 헤드를 추가하고, 교차 엔트로피(Cross-Entropy) 손실 함수를 사용하여 100 에포크 동안 전체 네트워크를 미세 조정한다. 클래스 불균형 문제를 해결하기 위해 리샘플링(Resampling) 전략을 적용하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: EgoSurgery-Phase (20개 비디오, 총 15시간, 9개 단계)
- **평가 지표**: Precision, Recall, Jaccard Index (클래스 불균형을 고려하여 Macro-average 사용)
- **비교 대상**:
  - 일반 단계 인식 모델: PhaseLSTM, PhaseNet, TeCNO, Trans-SVNet, NETE
  - MAE 기반 모델: Supervised ViT-S, VideoMAE, VideoMAE V2, SurgMAE

### 정량적 결과

실험 결과, GGMAE는 모든 지표에서 기존 모델들을 상회하는 성능을 보였다.

1. **기존 인식 모델 대비**: 가장 성능이 좋았던 NETE와 비교했을 때, Precision은 8.0%p, Recall은 10.4%p, Jaccard Index는 6.4%p 향상되었다.
2. **MAE 기반 모델 대비**: VideoMAE V2 대비 Jaccard Index 기준 3.1%p, SurgMAE 대비 6.1%p 더 높은 성능을 기록하였다.

### 절제 연구 (Ablation Study)

- **마스킹 전략**: Random 및 Tube 마스킹보다 Gaze-guided 마스킹을 사용했을 때 Jaccard Index가 3.3%p 향상되어, 시선 정보가 유의미한 가이드 역할을 함을 입증하였다.
- **마스킹 비율 ($\rho$)**: 90%일 때 최적의 성능을 보였으며, 너무 높거나 낮을 경우 성능이 하락하는 경향을 보였다.
- **온도 파라미터 ($\tau$)**: $\tau=0.5$일 때 가장 높은 성능을 기록하였으며, 이는 시선 집중 영역의 마스킹 강도와 무작위성의 적절한 균형이 중요함을 시사한다.

## 🧠 Insights & Discussion

본 논문의 강점은 개복 수술이라는 데이터 희소 영역에서 고품질의 데이터셋을 구축함과 동시에, 수술자의 시선이라는 도메인 특화 정보를 딥러닝의 마스킹 전략에 영리하게 결합했다는 점이다. 특히 단순히 시선 정보를 입력 피처로 사용하는 것이 아니라, 학습 과정의 '제약 조건(Masking)'으로 활용하여 모델이 강제로 중요한 영역을 학습하게 만든 점이 인상적이다.

다만, 몇 가지 한계점이 존재한다. 첫째, 데이터셋의 클래스 불균형이 매우 심하며(그림 3 참조), 이를 리샘플링으로 해결하려 했으나 특정 단계(예: Dissection)에 과하게 편향된 데이터 분포가 최종 성능에 영향을 주었을 가능성이 있다. 둘째, 시선 데이터 수집을 위해 특수 장비(Tobii 카메라)가 필요하므로, 데이터 수집의 확장성 면에서 제약이 있다. 셋째, 본 논문은 단일 시점(surgeon's head) 데이터만 사용하였으나, 실제 수술실에서는 다양한 각도의 시점이 존재하므로 이를 통합하는 연구가 필요하다.

## 📌 TL;DR

본 연구는 공개된 최초의 대규모 개복 수술 egocentric 비디오 데이터셋인 **EgoSurgery-Phase**를 구축하고, 수술자의 시선 정보를 활용해 의미 있는 영역을 집중적으로 학습하는 **GGMAE** 방법론을 제안하였다. GGMAE는 기존의 무작위 마스킹 방식보다 효율적으로 수술 영상의 특징을 추출하여, 수술 단계 인식 작업에서 SOTA(State-of-the-art) 성능을 달성하였다. 이 연구는 향후 개복 수술의 자동 분석 및 수술 보조 시스템 개발에 중요한 기반이 될 것으로 기대된다.
