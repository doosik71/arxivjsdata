# ‘Squeeze & Excite’ Guided Few-Shot Segmentation of Volumetric Images

Abhijit Guha Roy, Shayan Siddiqui, Sebastian Pölsterl, Nassir Navab, Christian Wachinger (2019)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석에서 수반되는 막대한 양의 수동 어노테이션(manual annotation) 비용 문제를 해결하기 위해, 매우 적은 수의 학습 데이터만으로 새로운 클래스를 분할(segmentation)하는 Few-shot segmentation 프레임워크를 제안한다. 

일반적인 딥러닝 기반 분할 모델은 수천 개의 정밀하게 라벨링된 데이터가 필요하지만, 의료 분야에서는 전문가의 의존도가 높아 이러한 데이터를 확보하기 어렵다. 기존의 컴퓨터 비전 분야 Few-shot segmentation 연구들은 주로 2D RGB 이미지를 대상으로 하며, ImageNet 등으로 사전 학습된(pre-trained) 네트워크를 사용하여 초기 가중치를 설정한다. 그러나 의료 영상 분야에서는 다음과 같은 두 가지 핵심적인 제약 사항이 존재한다:

1. **사전 학습된 모델의 부재**: 의료 영상의 특성상 일반 이미지 모델을 그대로 적용하기 어려우며, 의료 영상 전용의 범용 사전 학습 모델이 부족하여 모델을 처음부터(from scratch) 학습시켜야 하는 불안정성이 존재한다.
2. **볼륨 데이터의 특성**: 의료 영상은 3D 볼륨 데이터이지만, 모든 슬라이스를 라벨링하는 것은 매우 비효율적이다. 따라서 소수의 슬라이스만으로 전체 볼륨을 분할해야 하는 전략이 필요하다.

결과적으로 본 논문의 목표는 사전 학습된 모델 없이도 안정적으로 학습 가능하며, 소수의 어노테이션 슬라이스를 활용해 3D 의료 영상의 장기를 효과적으로 분할하는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Conditioner arm**과 **Segmenter arm** 사이의 강한 상호작용(strong interaction)을 구축하여 사전 학습된 모델 없이도 학습의 안정성을 확보하는 것이다.

구체적인 기여 사항은 다음과 같다:
- **sSE(channel squeeze & spatial excitation) 기반 상호작용**: 기존의 단순한 결합 방식 대신, 가벼우면서도 강력한 sSE 모듈을 네트워크의 여러 지점(Encoder, Bottleneck, Decoder)에 배치하여 Conditioner의 task representation이 Segmenter의 특징 맵을 효과적으로 재보정(re-calibration)하도록 설계하였다.
- **사전 학습 모델 없는 학습 가능성**: 강한 상호작용을 통해 그래디언트 흐름을 원활하게 하여, 의료 영상 분야의 제약인 '사전 학습 모델 부재' 문제를 해결하고 처음부터 안정적인 학습을 가능하게 하였다.
- **효율적인 볼륨 분할 전략**: 제한된 어노테이션 예산 $k$ 내에서, 서포트 볼륨의 대표 슬라이스를 선정하고 이를 쿼리 볼륨의 해당 구간 슬라이스들과 최적으로 매칭하여 3D 일관성을 유지하는 전략을 제안하였다.

## 📎 Related Works

논문에서는 Few-shot learning을 세 가지 그룹(기존 분류기 적응, 동적 파라미터 예측, Metric learning)으로 분류하며, 특히 분할 분야의 기존 연구들을 언급한다.

1. **Fine-tuning 기반 접근 (Caelles et al., 2017)**: 사전 학습된 모델을 소량의 데이터로 미세 조정하는 방식이다. 하지만 매우 적은 데이터 상황에서는 과적합(overfitting) 위험이 크고, 새로운 클래스가 추가될 때마다 재학습이 필요하다는 한계가 있다.
2. **2-arm 아키텍처 기반 접근 (Shaban et al., 2017; Rakelly et al., 2018)**: Conditioner가 태스크 표현을 생성하고 이를 Segmenter에 전달하는 구조이다. Shaban et al.은 분류기 가중치를 회귀(regress)하는 방식을, Rakelly et al.은 Feature fusion 방식을 사용하였다. 
3. **차별점**: 기존 방식들은 주로 2D 이미지에 국한되며, 무엇보다 사전 학습된 모델을 사용하여 강력한 초기 특징을 추출한다는 전제가 있다. 반면, 본 논문은 사전 학습 모델 없이도 동작하도록 sSE 모듈을 통한 다지점의 강한 상호작용을 도입하였으며, 3D 볼륨 데이터 처리를 위한 슬라이스 매칭 전략을 추가하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
전체 네트워크는 **Conditioner arm**, **Interaction blocks (sSE)**, **Segmenter arm**의 세 가지 구성 요소로 이루어진 대칭적 구조를 가진다.

- **Conditioner arm**: 서포트 세트 $(I_s, L_s(\alpha))$를 입력받아 해당 클래스의 특징을 추출하는 Task representation을 생성한다. 이미지와 바이너리 마스크를 채널 방향으로 쌓아(stacking) 2채널 입력을 사용하며, Encoder-Decoder 구조를 가진다. 특이사항으로 U-Net과 달리 skip connection을 제거하였다.
- **Segmenter arm**: 쿼리 이미지 $I_q$를 입력받아 최종 분할 맵 $M_q(\alpha)$를 생성한다. 구조는 Conditioner와 유사하나 채널 수가 더 많고(64개), 마지막에 Soft-max를 포함한 Classifier block이 추가되어 있다.
- **Interaction blocks (sSE)**: Conditioner의 특징 맵을 사용하여 Segmenter의 특징 맵을 공간적으로 재보정한다.

### 2. Squeeze & Excitation (sSE) 모듈
sSE 모듈은 다음과 같은 과정을 통해 Conditioner의 정보를 Segmenter에 전달한다.

1. **Channel Squeeze**: Conditioner의 특징 맵 $U_{con} \in \mathbb{R}^{H \times W \times C'}$를 $1 \times 1$ 컨볼루션을 통해 공간 맵 $q \in \mathbb{R}^{H \times W}$로 투영한다.
   $$q = W_{sq} * U_{con}$$
2. **Spatial Excitation**: 생성된 $q$를 시그모이드 함수 $\sigma(\cdot)$에 통과시켜 $[0, 1]$ 범위의 가중치 맵을 만들고, 이를 Segmenter의 특징 맵 $U_{seg}$에 곱하여 최종 특징 맵 $\hat{U}_{seg}$를 생성한다.
   $$\hat{U}_{seg} = \sigma(q) \cdot U_{seg}$$
이 과정은 모델 복잡도를 거의 높이지 않으면서 Segmenter가 서포트 세트의 공간적 정보에 집중하게 만든다.

### 3. 학습 절차 및 손실 함수
- **Batch Sampler**: 학습 시 무작위로 클래스 $\alpha$를 선택하고, 해당 클래스를 포함하는 두 개의 슬라이스 쌍을 무작위로 추출하여 하나는 서포트 세트로, 다른 하나는 쿼리 세트로 사용한다.
- **손실 함수**: 예측 맵 $M_q(x)$와 정답 맵 $L_q(x)$ 사이의 **Dice loss**를 사용하여 최적화한다.
  $$L_{Dice} = 1 - \frac{2 \sum_x M_q(x) L_q(x)}{\sum_x M_q(x) + \sum_x L_q(x)}$$
- **최적화**: Momentum이 적용된 SGD를 사용하여 가중치를 업데이트하며, 타겟 클래스가 매 반복마다 변경되므로 네트워크가 특정 클래스에 종속되지 않고 일반적인 분할 능력을 학습하게 된다.

### 4. 볼륨 분할 전략 (Volumetric Segmentation Strategy)
3D 볼륨을 분할하기 위해 제한된 예산 $k$개의 슬라이스만 라벨링하여 사용하는 전략을 제안한다.
1. 서포트 볼륨과 쿼리 볼륨에서 장기가 존재하는 슬라이스 범위 $[S_s, S_e]$와 $[Q_s, Q_e]$를 설정한다.
2. 두 범위를 각각 $k$개의 동일한 간격 그룹으로 나눈다.
3. 각 서포트 그룹의 중심 슬라이스 $S_{c_j}$를 라벨링하여 서포트 세트로 사용한다.
4. 서포트의 $j$번째 중심 슬라이스를 쿼리의 $j$번째 그룹에 속하는 모든 슬라이스와 매칭하여 분할을 수행한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Visceral dataset (ceCT scans) 사용.
- **대상 장기**: 간(Liver), 비장(Spleen), 좌/우 신장(Kidney), 좌/우 요근(Psoas Muscle) 총 6개 클래스.
- **평가 지표**: Dice score 및 Average Surface Distance (mm).

### 2. 주요 정량적 결과
- **sSE의 효과**: sSE 기반 상호작용이 cSE(channel excitation)보다 압도적으로 우수함을 확인하였다. (Mean Dice: sSE $\approx 0.567$ vs cSE $\approx 0.056$). 이는 의료 영상에서 타겟 장기가 차지하는 비율이 작아, 전역 평균 풀링을 사용하는 cSE는 배경 정보에 의해 클래스 정보가 소실되기 때문으로 분석된다.
- **상호작용 위치**: Encoder, Bottleneck, Decoder 모든 지점에서 상호작용이 일어날 때 가장 높은 성능을 보였다.
- **서포트 슬라이스 수 ($k$)**: $k=10$일 때 성능이 포화(saturation)되는 경향을 보였으며, 평균 Dice score는 0.56에 도달하였다.
- **기존 방법론 비교 (Test Set)**:
  - **Proposed**: Mean Dice $0.485$, Avg. Surface Distance $10.48\text{mm}$.
  - **Feature Fusion (Rakelly et al.)**: Mean Dice $0.270$, Avg. Surface Distance $20.00\text{mm}$.
  - **Fine-Tuning**: Mean Dice $0.092$, Avg. Surface Distance 측정 불가(대부분 실패).
  - 제안 방법이 기존 Few-shot 방식보다 Dice score 기준 약 21%p 높은 성능을 보였다.

### 3. 상한선(Upper Bound) 모델과의 비교
모든 데이터를 사용하여 학습한 Fully Supervised U-Net 모델과 비교했을 때, 제안 방법은 약 20-40% 낮은 Dice score를 기록하였다. 하지만 이는 수천 개의 슬라이스를 학습한 모델과 단 10개의 슬라이스로 학습한 모델의 차이임을 감안할 때, 매우 효율적인 결과라고 평가할 수 있다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰
- **Copy-over effect 발견**: 본 연구에서 흥미로운 점은 skip connection을 도입했을 때, 네트워크가 쿼리 이미지의 특징을 학습하는 대신 서포트 세트의 바이너리 마스크를 그대로 복사하여 출력하는 현상을 발견한 것이다. 이를 통해 Few-shot segmentation에서는 일반적인 U-Net 구조의 skip connection이 오히려 독이 될 수 있음을 시사하였다.
- **사전 지식의 활용**: 테스트 클래스가 학습 데이터에서 '배경'으로 이미 등장했다는 점이 모델이 새로운 클래스를 빠르게 학습하는 데 긍정적인 prior knowledge로 작용했을 가능성이 있다.

### 2. 한계 및 비판적 해석
- **슬라이스 범위 지정의 필요성**: 볼륨 분할을 위해 장기가 존재하는 시작과 끝 슬라이스 인덱스를 미리 알아야 한다는 제약이 있다. 이는 실무에서 수동 작업이나 별도의 자동화 도구가 필요함을 의미한다.
- **정밀도 한계**: 매우 적은 감독 데이터만 사용하므로, 정밀한 경계 분할(precise segmentation)에는 한계가 있으며, 높은 정확도가 필수적인 의료 현장에서는 여전히 Fully Supervised 학습이 권장된다.
- **서포트 볼륨 의존성**: 어떤 볼륨을 서포트 세트로 선택하느냐에 따라 성능이 약 4-8% 정도 변동하는 것을 확인하였다. 이는 대표성 있는 서포트 볼륨을 선정하는 전략이 향후 중요한 연구 과제임을 보여준다.

## 📌 TL;DR

본 논문은 사전 학습된 모델 없이 3D 의료 영상을 Few-shot으로 분할하기 위한 **sSE 기반의 두 팔(two-armed) 구조 네트워크**를 제안한다. Conditioner와 Segmenter 사이에 sSE 모듈을 다중 배치하여 강한 상호작용을 구축함으로써 학습의 안정성을 높였으며, 효율적인 슬라이스 매칭 전략을 통해 볼륨 데이터를 처리하였다. 실험 결과, 기존의 Feature Fusion이나 Fine-tuning 방식보다 월등한 성능을 보였으며, 특히 의료 영상 특유의 제약 사항을 극복하며 Few-shot learning의 의료 적용 가능성을 입증하였다. 이 연구는 데이터 확보가 어려운 희귀 질환이나 특수 장기 분할 연구에 중요한 기초가 될 수 있다.