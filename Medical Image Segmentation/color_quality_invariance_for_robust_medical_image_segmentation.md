# Color-Quality Invariance for Robust Medical Image Segmentation

Ravi Shah, Atsushi Fukuda, and Quan Huu Cap (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 발생하는 **단일 소스 도메인 일반화(Single-source Domain Generalization, SDG)** 문제를 해결하고자 한다. 딥러닝 기반의 분할 모델은 학습 데이터와 테스트 데이터의 분포가 다른 '도메인 시프트(Domain Shift)' 현상에 매우 취약하다. 특히 의료 영상의 경우, 전문 의료 기기로 촬영한 고품질(High-Quality, HQ) 이미지로 학습된 모델이 저품질(Low-Quality, LQ) 이미지나 스마트폰(Smartphone, SP)으로 촬영된 이미지에 적용될 때, 색상 분포와 화질의 차이로 인해 성능이 급격히 저하되는 문제가 발생한다.

다양한 도메인의 데이터를 수집하여 학습시키는 것이 이상적이지만, 의료 데이터의 특성상 수집과 공유가 어렵고 라벨링 비용이 매우 높기 때문에, 단일 소스 도메인의 데이터만으로도 보지 못한(unseen) 다양한 타겟 도메인에서 강건하게 작동하는 모델을 개발하는 것이 본 연구의 핵심 목표이다.

## ✨ Key Contributions

본 논문은 색상과 화질의 변화에 무관한(Invariant) 분할 성능을 확보하기 위해 다음의 두 가지 핵심 아이디어를 제안한다.

1. **Dynamic Color Image Normalization (DCIN) 모듈**: 테스트 시점에 입력 이미지의 색상을 소스 도메인의 색상 분포로 동적으로 정렬하는 기법이다. 단순히 고정된 참조 이미지를 사용하는 것이 아니라, 전체 데이터셋을 대표하는 글로벌 참조 이미지와 개별 입력 이미지와 유사한 로컬 참조 이미지를 전략적으로 선택하여 색상을 전이(Color Transfer)한다.
2. **Color-Quality Generalization (CQG) Loss**: 학습 단계에서 동일한 입력 이미지에 대해 서로 다른 색상 및 화질 변형을 가하더라도 모델이 일관된 분할 결과(Segmentation Mask)를 생성하도록 강제하는 대조 학습(Contrastive-based) 기반의 손실 함수이다.

## 📎 Related Works

기존의 도메인 시프트 해결 방안으로 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)이 제시되었으나, 이는 타겟 도메인의 데이터를 미리 알고 있어야 하며 도메인이 바뀔 때마다 재학습이 필요하다는 단점이 있다. 또한, 다중 소스 도메인 일반화(Multi-source Domain Generalization, MDG)는 여러 개의 라벨링된 소스 데이터셋이 필요하여 실제 의료 환경에서 적용하기 어렵다.

이에 따라 단일 소스 데이터만 사용하는 SDG 연구들이 진행되어 왔으나, 대부분 도메인 간 분포 차이가 적다는 가정을 전제로 한다. 실제 환경에서는 색상과 화질의 차이가 매우 크기 때문에 기존 SDG 방식만으로는 한계가 있다. 기존의 색상 정규화(Color Normalization) 기법들은 전문가가 선택한 참조 이미지에 의존하는 경향이 있어 주관적이며 최적의 성능을 보장하지 못한다는 한계가 존재한다.

## 🛠️ Methodology

### 1. Dynamic Color Image Normalization (DCIN)

DCIN은 테스트 이미지의 색상 분포를 소스 도메인의 참조 이미지와 일치시켜 모델의 입력 분포를 정렬한다. 색상 전이는 지각 기반 색 공간인 $l\alpha\beta$ 공간에서 수행된다. 참조 이미지를 선택하는 두 가지 전략은 다음과 같다.

- **Global Reference Image Selection (GRIS)**: 소스 도메인의 모든 이미지를 $ab$-빈 정규화 색상 히스토그램 벡터로 변환한다. 모든 이미지 간의 평균 쌍별 유클리드 거리(Average Pairwise Distance)를 최소화하는 이미지를 전역 참조 이미지 $x_g$로 선택한다.
  $$D_{pairwise}(x_i) = \frac{1}{N} \sum_{j=1}^{N} \sqrt{\sum_{k=1}^{b} (H_k(x_i) - H_k(x_j))^2}$$
  여기서 $H_k(x)$는 이미지 $x$의 $k$번째 빈 값이며, $N$은 소스 도메인의 이미지 수이다.
- **Local Reference Image Selection (LRIS)**: 입력 테스트 이미지 $x_{test}$와 시맨틱하게 가장 유사한 이미지를 선택한다. 사전 학습된 Swin-V2-Large 모델을 통해 소스 이미지들의 특징 벡터를 추출하고, $x_{test}$의 특징 벡터와 코사인 유사도(Cosine Similarity)가 가장 높은 이미지를 로컬 참조 이미지 $x_l$로 선택한다.
- **Ensemble**: GRIS와 LRIS를 통해 각각 정규화된 두 개의 입력 이미지를 생성하고, 모델을 통해 나온 두 개의 예측 마스크를 픽셀 단위로 평균(Pixel-wise mean) 내어 최종 결과를 도출한다.

### 2. Color-Quality Generalization (CQG) Loss

CQG Loss는 색상과 화질이 변하더라도 모델이 동일한 분할 결과를 내도록 유도하는 훈련 목표 함수이다.

- **입력 생성**: 학습 이미지 $x$에 대해 두 가지 변형을 가한다.
  - $x_1$: 기하학적 변형(Random horizontal flip, shear, shift, scale, rotation, elastic transform)만 적용.
  - $x_2$: 기하학적 변형과 광도 변형(Random blur, sharpening, Gaussian noise, brightness contrast, RGB shifts)을 모두 적용.
- **손실 함수 정의**:
  $$\mathcal{L} = \lambda_1 DC(y, y_1) + \lambda_2 DC(y, y_2) + \lambda_3 MSE(y_1, y_2)$$
  - $y$: Ground-truth 마스크
  - $y_1 = S(x_1), y_2 = S(x_2)$: 모델 $S$의 예측 마스크
  - $DC(y, y')$: Dice loss와 Cross-entropy loss의 합
  - $MSE(y_1, y_2)$: 두 예측 마스크 간의 평균 제곱 오차(Mean Squared Error)
  - $\lambda_1, \lambda_2, \lambda_3$: 각 항의 가중치를 조절하는 하이퍼파라미터

## 📊 Results

### 실험 설정

- **데이터셋**: 일본 내 병원에서 수집한 고품질 인후 이미지(HQ, 16,000장, 그 중 2,000장 라벨링), 저품질 이미지(LQ, 255장), 스마트폰 촬영 이미지(SP, 125장)를 사용하였다.
- **모델 구조**: EfficientNet-B2를 백본으로 하는 U-Net 구조를 사용하였다.
- **비교 대상**:
  - $S_{base}$: 최소한의 전처리만 적용한 베이스라인.
  - $S_{aug}$: CQG Loss의 증강 기법만 적용하고 Loss는 사용하지 않은 모델.
  - $S_{CQG}$: CQG Loss를 적용하여 학습한 모델.
- **평가 지표**: Dice Score를 사용하여 예측 마스크와 정답 마스크의 겹침 정도를 측정하였다.

### 정량적 결과

- **소스 도메인(HQ)**: 모든 모델이 약 88% 내외의 높은 Dice Score를 기록하여 HQ 이미지에 대해서는 잘 작동함을 확인하였다.
- **타겟 도메인(LQ, SP)**:
  - $S_{base}$ 모델은 HQ $\to$ LQ 전이 시 Dice Score가 88.9에서 36.9로 급격히 하락하였다.
  - DCIN과 CQG Loss를 모두 적용했을 때 성능 향상이 가장 뚜렷하였다. 특히 LQ 데이터셋에서 베이스라인 대비 최대 **32.3 포인트**의 Dice Score 상승을 기록하였다.
  - DCIN의 참조 이미지 선택 전략 중, 전문가가 선택한 이미지(ExRI)보다 제안된 GRIS와 LRIS 전략이 더 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 학습 단계에서의 강건함(CQG Loss)과 추론 단계에서의 데이터 정렬(DCIN)을 동시에 적용함으로써, 극심한 도메인 시프트 상황에서도 실용적인 수준의 분할 성능을 확보하였다. 특히, 전문가의 주관적인 선택보다 데이터 기반의 객관적인 참조 이미지 선택 전략(GRIS, LRIS)이 더 효과적이라는 점을 입증한 것이 고무적이다.

### 한계 및 향후 과제

1. **효율성 문제**: 현재 DCIN의 색상 전이 연산이 CPU에서 수행되어 GPU 프로세싱과의 병목 현상이 발생한다. 이를 GPU로 이식하여 처리 속도를 높일 필요가 있다.
2. **구조적 차이**: 색상과 화질 외의 도메인 시프트(예: 스마트폰 이미지에 나타나는 치아와 같은 아티팩트)는 여전히 모델을 혼란스럽게 만든다. 이는 단순한 색상 정규화만으로는 해결할 수 없으며, 추가적인 후처리 기법이나 구조적 강건성을 높이는 연구가 필요하다.

## 📌 TL;DR

본 논문은 단일 소스 의료 영상 데이터만으로 학습하여 다양한 저품질/스마트폰 영상에서도 잘 작동하는 분할 모델을 제안한다. 테스트 시점에 색상을 동적으로 맞추는 **DCIN**과 학습 시 일관성을 강제하는 **CQG Loss**를 통해, 타겟 도메인에서의 Dice Score를 최대 32.3포인트 향상시켰다. 이 방법론은 모델에 구애받지 않는(Model-agnostic) 특성을 가져 다양한 의료 영상 분할 시스템에 확장 적용될 가능성이 높다.
