# SAM2-3DMED: EMPOWERING SAM2 FOR 3D MEDICAL IMAGE SEGMENTATION

Yeqing Yang, Le Xu, Lixia Tian (2025)

## 🧩 Problem to Solve

3D 의료 영상 분할(Medical Image Segmentation)은 질병 진단, 치료 계획 수립 및 수술 내비게이션 등 정밀 의료의 핵심적인 요소이다. 하지만 고품질의 3D 의료 데이터에 대해 슬라이스별로 정밀한 어노테이션(Annotation)을 수행하는 것은 비용과 시간이 매우 많이 소요되는 '어노테이션 병목 현상(Annotation Bottleneck)' 문제를 야기한다.

최근 비디오 객체 분할(Video Object Segmentation)에서 뛰어난 성능을 보인 SAM2(Segment Anything Model 2)를 3D 의료 영상에 적용하려는 시도가 있으나, 저자들은 비디오와 3D 의료 영상 사이에 두 가지 근본적인 도메인 간극(Domain Gap)이 존재한다고 분석한다.

첫째, 비디오의 프레임 간 관계는 단방향의 시간적 흐름(Unidirectional Temporal Flow)을 가지는 반면, 3D 의료 영상의 슬라이스 간 관계는 해부학적 연속성에 기반한 양방향 공간적 문맥(Bidirectional Spatial Context)을 가진다. 둘째, 비디오 분할은 대상의 전체적인 일관성에 집중하는 경향이 있어 경계선 묘사의 정밀도가 상대적으로 낮지만, 의료 영상 분할은 진단 정확도와 치료 안전성을 위해 매우 정밀한 경계 묘사(Boundary Delineation)가 필수적이다.

따라서 본 논문의 목표는 SAM2를 3D 의료 영상 도메인에 맞게 최적화하여, 양방향 공간 의존성을 모델링하고 경계 분할의 정밀도를 높이는 SAM2-3dMed 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM2의 강력한 특징 표현 능력을 유지하면서, 자기지도 학습(Self-supervised Learning) 과제와 구조적 최적화를 통해 모델이 의료 영상의 공간적 구조 특성을 이해하고 경계 정보에 집중하도록 유도하는 것이다.

이를 위해 크게 두 가지 핵심 모듈을 도입하였다.
1. **Slice Relative Position Prediction (SRPP) 모듈**: 임의의 두 슬라이스 간의 상대적 위치를 예측하는 자기지도 학습 과제를 통해, SAM2가 시간적 의존성이 아닌 양방향 공간적 의존성과 해부학적 연속성을 학습하도록 강제한다.
2. **Boundary Detection (BD) 모듈**: 마스크 디코더와 병렬로 작동하는 전용 경계 검출 브랜치를 추가하여, 장기와 조직의 윤곽선을 명시적으로 학습함으로써 분할의 정밀도를 향상시킨다.

## 📎 Related Works

### 3D 의료 영상 분할의 기존 접근 방식
기존에는 3D U-Net, V-Net과 같은 Fully Convolutional Network(FCN) 기반 구조나 UNETR와 같은 Transformer 기반 모델들이 주로 사용되었다. 특히 nnU-Net은 네트워크 설계와 전처리를 자동화하여 매우 높은 성능을 보여주었다. 그러나 이러한 모델들은 대량의 정밀한 3D 라벨 데이터가 필요한 완전 지도 학습(Fully Supervised Learning)에 의존한다는 한계가 있다.

### 데이터 효율적 학습 (SSL 및 TL)
어노테이션 병목 현상을 해결하기 위해 준지도 학습(Semi-Supervised Learning, SSL)과 전이 학습(Transfer Learning, TL)이 연구되었다. SSL은 라벨이 없는 데이터를 활용하지만 데이터 분포에 민감하고 노이즈가 전파될 위험이 있다. 반면 TL은 대규모 소스 도메인(자연 이미지 등)에서 학습된 표현을 타겟 도메인으로 전이하여 데이터 부족 문제를 완화한다.

### 파운데이션 모델의 등장
SAM(Segment Anything Model)은 2D 이미지 분할에서 혁신을 일으켰으며, 이를 의료 영상에 적용한 MedSAM 등이 등장하였다. 하지만 SAM은 3D 볼륨을 독립적인 2D 슬라이스로 처리하여 볼륨 문맥을 무시하는 한계가 있다. SAM2는 비디오의 프레임 간 의존성을 모델링하는 메모리 메커니즘을 갖추고 있어 3D 의료 영상을 '공간적 비디오(Spatial Video)'로 간주하고 적용할 수 있는 가능성을 열었으나, 앞서 언급한 도메인 간극으로 인해 단순 적용 시 성능이 최적화되지 않는 문제가 발생한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
SAM2-3dMed는 입력 3D 의료 볼륨 $X \in \mathbb{R}^{3 \times D \times H \times W}$를 받아 정밀한 분할 마스크 $P^{seg}$를 생성한다. 전체 시스템은 다음과 같이 구성된다.

1. **SAM2 Backbone**: 사전 학습된 SAM2의 Image Encoder를 사용하여 각 슬라이스에서 특징을 추출한다. 이때 일반화 능력을 보존하기 위해 Encoder는 학습 과정에서 고정(Frozen) 상태를 유지하며, 결과물로 특징 텐서 $Z \in \mathbb{R}^{C \times D \times H' \times W'}$를 생성한다.
2. **SAM2-based Segmentation Module**: SAM2의 Memory Attention과 Mask Decoder를 통해 최종 분할 맵 $P^{seg}$를 생성한다.
3. **SRPP Module**: 슬라이스 간의 양방향 공간 의존성을 학습하는 보조 모듈이다.
4. **BD Module**: 경계선 특징을 추출하고 강화하여 분할 정밀도를 높이는 병렬 브랜치이다.

### 상세 모듈 설명 및 방정식

#### 1. Slice Relative Position Prediction (SRPP) 모듈
이 모듈은 두 슬라이스 $i$와 $j$ 사이의 상대적 오프셋을 예측함으로써 공간적 문맥을 학습한다.
- **예측 과정**: 특징 텐서 $Z$의 각 슬라이스 표현 $Z_i, Z_j$를 입력으로 하여 상대적 위치 $(P^{pos})_{i,j}$를 예측한다.
  $$(P^{pos})_{i,j} = \text{SRPP}(Z_i, Z_j)$$
- **정답(GT)**: 실제 상대 위치 $(GT^{pos})_{i,j} = j - i$로 정의된다.
- **손실 함수**: Mean Squared Error (MSE) 손실을 사용하여 학습하며, 모든 유일한 슬라이스 쌍에 대해 합산한다.
  $$L^{srpp} = \frac{\sum_{i=1}^{D} \sum_{j=1, j \neq i}^{D} ((P^{pos})_{i,j} - (GT^{pos})_{i,j})^2}{D(D-1)}$$

#### 2. Boundary Detection (BD) 모듈
경계 픽셀의 희소성으로 인한 클래스 불균형을 해결하고 정밀한 윤곽선을 추출한다.
- **특징 융합**: 경계 인식 특징 $(Z^{bd})_i$와 일반 슬라이스 특징 $Z_i$를 Cross-Attention 메커니즘을 통해 융합하여 경계 위치를 정교화한다.
  $$(Z'_{bd})_i = \text{softmax}\left(\frac{(Q^{bd})_i (K_i)^T}{\sqrt{d}}\right) V_i$$
- **손실 함수**: 경계 픽셀에 더 높은 가중치를 부여하는 Weighted Binary Cross-Entropy (BCE) 손실을 사용한다.
  $$L^{bd} = \frac{N_{non-bd}}{N} \sum_{j \in \Omega_{bd}} \text{BCE}((P^{bd})_j, (GT^{bd})_j) + \frac{N_{bd}}{N} \sum_{j \in \Omega_{non-bd}} \text{BCE}((P^{bd})_j, (GT^{bd})_j)$$
  여기서 $N_{bd}$와 $N_{non-bd}$는 각각 경계 및 비경계 픽셀의 수이다.

### 학습 절차 및 총 손실 함수
모델은 세 가지 손실 함수의 결합을 최소화하는 방향으로 학습된다. Image Encoder는 고정되며, Memory Attention과 Mask Decoder 블록들이 미세 조정(Fine-tuning)된다.
$$L^{total} = L^{seg} + \lambda_1 L^{srpp} + \lambda_2 L^{bd}$$
여기서 $L^{seg}$는 예측된 분할 맵과 실제 정답 간의 BCE 손실이며, $\lambda_1, \lambda_2$는 각 손실 항목의 균형을 맞추기 위한 하이퍼파라미터이다.

## 📊 Results

### 실험 설정
- **데이터셋**: Medical Segmentation Decathlon (MSD)의 세 가지 과제(Lung Tumor, Spleen, Pancreas)를 사용하였다.
- **비교 대상 (Baselines)**: nnU-Net, nnFormer, MedNeXt(CNN/Transformer 기반), Universal-Model, MedSAM-2, Fine-tuned SAM2.
- **평가 지표**: 영역 중첩도를 측정하는 Dice와 IoU, 경계 정밀도를 측정하는 HD95(Hausdorff Distance 95th percentile)와 NSD(Normalized Surface Distance)를 사용하였다. (Dice, IoU, NSD는 높을수록, HD95는 낮을수록 우수)

### 주요 결과
정량적 분석 결과, SAM2-3dMed는 모든 데이터셋에서 기존 SOTA 방법론들을 능가하였다. 특히 형태학적 변동성이 큰 **Pancreas(췌장)** 데이터셋에서 괄목할 만한 성능 향상을 보였으며, Dice 기준 두 번째로 좋은 모델 대비 $+2.98\%$ 향상된 $0.7039$를 기록하였다. 또한, HD95와 NSD 지표에서 우수한 성적을 거두어 경계 분할의 정밀도가 크게 개선되었음을 입증하였다.

정성적 분석(Fig. 3)에서도 SRPP 모듈 덕분에 슬라이스 간 구조적 일관성이 높아졌으며, BD 모듈을 통해 실제 정답(GT) 윤곽선에 훨씬 더 가깝게 분할되는 결과가 확인되었다.

### 절제 연구 (Ablation Study)
1. **전이 학습의 영향**: 사전 학습된 SAM2 가중치를 사용하지 않았을 때(w/o Pre-train) 성능이 급격히 저하되었다. 특히 Pancreas의 Dice score가 $0.4353$에서 $0.7039$로 크게 상승하여, 데이터가 부족한 의료 영상 분야에서 파운데이션 모델의 가중치 초기화가 필수적임을 보였다.
2. **SRPP 모듈의 역할**: SRPP 제거 시 특히 Pancreas와 같이 공간적 연속성이 복잡한 장기에서 성능 저하가 뚜렷하였다. 이는 양방향 공간 의존성 모델링의 중요성을 시사한다.
3. **BD 모듈의 역할**: BD 모듈 제거 시 HD95 지표가 크게 악화되었으며, 이는 경계 인식 학습이 정밀한 형태 분석에 결정적인 역할을 함을 의미한다.
4. **입력 슬라이스 수의 영향**: 입력 슬라이스 수를 3 $\rightarrow$ 6 $\rightarrow$ 12로 늘릴수록 성능이 전반적으로 향상되었다. 이는 더 많은 공간적 문맥 정보가 제공될수록 분할 정확도가 높아짐을 보여준다.

## 🧠 Insights & Discussion

### 강점 및 기여도
본 연구는 비디오 파운데이션 모델인 SAM2를 3D 의료 영상에 적응시키기 위한 체계적인 방법론을 제시하였다. 단순히 모델을 적용하는 것에 그치지 않고, 비디오의 '시간적 흐름'과 의료 영상의 '공간적 연속성'이라는 도메인 차이를 명확히 분석하고 이를 SRPP 모듈이라는 구체적인 학습 과제로 해결한 점이 매우 뛰어나다. 또한, 의료 영상에서 가장 중요한 요소 중 하나인 경계 정밀도를 BD 모듈을 통해 명시적으로 해결함으로써 임상적 유용성을 높였다.

### 한계 및 논의사항
- **Image Encoder의 고정**: 효율적인 학습과 과적합 방지를 위해 Encoder를 고정하였으나, 의료 영상 특유의 텍스처나 병리적 특징을 더 깊게 학습하기 위해서는 일부 레이어를 해제하여 미세 조정하는 전략이 필요할 수 있다.
- **슬라이스 수의 트레이드-오프**: 입력 슬라이스 수가 많을수록 성능은 향상되지만, 이는 메모리 사용량 증가와 계산 복잡도 상승으로 이어진다. 실시간 진단 환경에 적용하기 위해서는 최적의 슬라이스 수를 결정하는 효율적인 윈도우 전략이 필요할 것이다.
- **메모리 모듈의 역설**: SRPP 모듈에 메모리 메커니즘을 추가했을 때 오히려 성능이 하락한 점은 흥미롭다. 이는 상대적 위치 예측과 같은 단순한 회귀 과제에서는 복잡한 메모리 구조가 오히려 노이즈로 작용하거나 과적합을 유발할 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 비디오 분할 모델인 SAM2를 3D 의료 영상 분할에 최적화한 **SAM2-3dMed**를 제안한다. 비디오의 단방향 시간 흐름과 의료 영상의 양방향 공간 연속성 차이를 극복하기 위해 **SRPP(슬라이스 상대 위치 예측)** 모듈을 도입하고, 정밀한 윤곽선 추출을 위해 **BD(경계 검출)** 모듈을 추가하였다. 실험 결과, MSD 데이터셋의 여러 장기 분할 작업에서 SOTA 성능을 달성하였으며, 특히 데이터가 부족한 환경에서 파운데이션 모델의 전이 학습과 공간적 제약 조건 부여가 매우 효과적임을 입증하였다. 이 연구는 향후 비디오 기반 파운데이션 모델을 공간적 볼륨 데이터로 확장하는 일반적인 패러다임을 제공한다.