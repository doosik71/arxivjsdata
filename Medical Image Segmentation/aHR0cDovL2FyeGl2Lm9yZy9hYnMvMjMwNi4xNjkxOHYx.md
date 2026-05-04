# PCDAL: A Perturbation Consistency-Driven Active Learning Approach for Medical Image Segmentation and Classification

Tao Wang, Xinlin Zhang, Yuanbo Zhou, Junlin Lan, Tao Tan, Min Du, Qinquan Gao and Tong Tong (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석 분야에서 딥러닝 모델의 성능을 높이기 위해 필수적인 대규모 어노테이션 데이터(annotated data) 확보의 어려움을 해결하고자 한다. 의료 영상 데이터의 레이블링은 전문 지식을 갖춘 숙련된 방사선 전문의만이 수행할 수 있어 비용과 시간이 매우 많이 소요되며, 실제 임상 환경에서는 대규모 데이터를 확보하는 것이 사실상 불가능한 경우가 많다.

기존의 Active Learning(AL) 방법론들이 자연어 처리나 일반 이미지 분류 작업에서는 성과를 거두었으나, 의료 영상 분야, 특히 3D 의료 영상 분할(segmentation) 작업에 효과적이고 보편적으로 적용될 수 있는 AL 기반 방법론은 부족한 실정이다. 따라서 본 연구의 목표는 2D 의료 영상의 분류 및 분할, 그리고 3D 의료 영상 분할 작업 모두에 동시에 적용 가능한 범용적인 AL 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Perturbation Consistency(섭동 일관성)**를 이용하여 모델이 확신하지 못하는 샘플을 식별하는 것이다. 동일한 이미지에 대해 섭동(perturbation, 여기서는 Flip 연산)을 가했을 때, 모델의 예측 결과가 크게 변한다면 해당 데이터는 모델이 충분히 학습하지 못한 정보가 포함되어 있을 가능성이 높다는 직관에 기반한다.

이를 위해 저자들은 Perturbation Consistency Evaluation Module(PCEM)을 설계하여 섭동에 따른 예측 값의 변화량을 정량화하고, 변화가 큰 샘플(High Perturbation Impact, HPI)을 우선적으로 레이블링함으로써 최소한의 데이터로 모델 성능을 극대화하는 전략을 제안한다.

## 📎 Related Works

의료 영상 분석을 위해 U-Net, Res-UNet, 3D U-Net과 같은 CNN 기반 구조와 최근의 Vision Transformer(ViT), Swin-Transformer 등이 활용되고 있다. 이러한 지도 학습 기반 방법들은 높은 성능을 보이지만, 앞서 언급한 데이터 부족 문제에 취약하다.

이를 해결하기 위해 Semi-supervised Learning(SSL) 등이 제안되었으나, SSL은 학습 시 GPU 메모리 점유율이 높고 학습 시간이 길다는 하드웨어적 제약이 존재한다. 반면, Active Learning(AL)은 정보량이 많은 샘플을 전략적으로 선택하여 레이블링 비용을 줄이는 접근 방식을 취한다. 기존 AL의 샘플 선택 전략으로는 Maximum Entropy를 이용한 불확실성 샘플링(Uncertainty Sampling), 데이터 분포를 고려한 Diversity-based sampling, 그리고 VAE와 적대적 네트워크를 결합한 VAAL 등이 있다. 하지만 이러한 방법들은 의료 영상의 고유한 특성을 충분히 반영하지 못하거나, 3D 분할 작업으로의 확장성이 부족하다는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인
PCDAL 프레임워크는 다음과 같은 반복적인 루프 과정을 거친다.
1. 제한된 양의 어노테이션 데이터로 CNN 모델을 초기 학습시킨다.
2. 학습된 모델을 사용하여 레이블이 없는(unlabeled) 데이터와 여기에 섭동을 가한 데이터들을 추론한다.
3. PCEM을 통해 각 샘플의 섭동 일관성을 평가하여 섭동 에러(perturbation error)를 계산한다.
4. 에러 값이 높은 HPI 샘플을 우선적으로 선택하여 레이블링을 수행한다.
5. 새로 레이블링된 데이터를 학습 세트에 추가하고 다시 모델을 학습시킨다. 이 과정은 목표 성능에 도달할 때까지 반복된다.

### 섭동 방법 (Flip-based Perturbation)
본 논문에서는 계산 비용이 낮고 이미지의 의미론적 정보(semantic information)를 유지하는 Flip 기반 섭동을 사용한다. 각 이미지에 대해 다음 세 가지 변환을 적용한다.
- 수평 뒤집기 (Horizontal Flipping)
- 수직 뒤집기 (Vertical Flipping)
- 수평 및 수직 뒤집기 조합 (Combination of both)

### Perturbation Consistency Evaluation Module (PCEM)
예측 결과의 일관성을 측정하기 위해 Mean Square Error(MSE) 기반의 계산 방식을 사용한다.

먼저 원본 이미지와 3개의 섭동 이미지에 대한 예측 결과의 평균 $\text{P}_{\text{average}}$를 계산한다.
$$\text{P}_{\text{average}} = \frac{1}{4} \sum_{i=1}^{4} \text{P}_{\text{img}_i}$$

분류(Classification) 작업의 경우, 각 예측 값과 평균 값의 차이의 제곱을 평균 내어 섭동 에러를 정의한다.
$$\text{P}_{\text{turb cls}} = \frac{1}{4} \sum_{i=1}^{4} (\text{P}_{\text{img}_i} - \text{P}_{\text{average}})^2$$

분할(Segmentation) 작업의 경우, 이미지 내의 모든 픽셀 $N$개에 대해 위와 같은 연산을 수행하여 전체 픽셀의 평균 에러를 계산한다.
$$\text{P}_{\text{turb seg}} = \frac{1}{N} \sum_{j=1}^{N} \text{P}_{\text{turb cls}_j}$$

### 작업별 적용 상세
- **분류 작업**: 각 이미지의 $\text{P}_{\text{turb cls}}$ 값을 기준으로 내림차순 정렬하여 HPI 샘플을 선택한다.
- **분할 작업**: 섭동된 이미지의 예측 결과는 뒤집힌 상태이므로, 먼저 역변환(Inverse Transformation)을 통해 원본 방향으로 되돌린 후 $\text{P}_{\text{turb seg}}$를 계산하여 HPI 샘플을 선택한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Kvasir(내시경 이미지 분류), COVID-19 Infection Segmentation(흉부 X-ray 분할), BraTS2019(3D 뇌 MRI 분할)
- **모델 구조**: 분류는 ResNet34, 2D 분할은 Res34-UNet, 3D 분할은 3D U-Net을 백본으로 사용하였다.
- **평가 지표**: 분류는 Precision(Pre)과 Accuracy(Acc), 2D 분할은 Dice coefficient와 Pixel Accuracy(PA), 3D 분할은 Dice coefficient와 $HD_{95}$(95% Hausdorff Distance)를 사용하였다.
- **실험 절차**: 초기 데이터 10%를 사용하고, 매 반복마다 10%씩 데이터를 추가하며 성능을 측정하였다.

### 주요 결과
- **Kvasir 데이터셋**: PCDAL은 30%의 데이터만으로 랜덤 샘플링의 50% 수준 성능을 상회하는 Accuracy $93.76\%$를 달성하였다. 이는 기존의 Max-Entropy나 ALFA-Mix보다 우수한 성능이다.
- **COVID-19 데이터셋**: 30% 데이터 사용 시 PCDAL의 Dice score가 랜덤 샘플링보다 유의미하게 높았으며, 섭동 영향이 낮은 LPI(Low Perturbation Impact) 샘플을 선택했을 때보다 훨씬 효율적임을 확인하였다.
- **BraTS2019 데이터셋**: 3D 분할 작업에서도 PCDAL은 30%의 데이터만으로 랜덤 샘플링 50%와 유사한 Dice $86.12\%$ 및 $HD_{95} 7.95$를 기록하여, 레이블링 비용을 약 20% 절감하였다.

### 절제 연구 (Ablation Study)
- **섭동 방법**: 수평, 수직, 그리고 두 조합의 Flip을 모두 사용했을 때 가장 안정적이고 높은 성능을 보였다. 회전(Rotation)을 추가하는 것은 계산 비용만 증가시킬 뿐 성능 향상은 미미하였다.
- **손실 함수**: PCEM에서 에러를 계산할 때 KL divergence, L1, Huber Loss보다 MSE Loss를 사용했을 때 가장 높은 정확도를 보였다. 이는 제곱 연산이 섭동에 의한 큰 변화를 더 효과적으로 증폭시켜 정보량이 많은 샘플을 더 잘 식별하기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 매우 단순한 섭동 연산과 일관성 측정만으로도 복잡한 의료 영상 작업에서 효과적인 AL 전략을 구축할 수 있음을 보여주었다. 특히 2D 분류, 2D 분할, 3D 분할이라는 서로 다른 차원과 목적의 작업에 동일한 프레임워크를 적용할 수 있다는 범용성이 큰 강점이다.

결과적으로 섭동에 민감하게 반응하는 HPI 샘플이 모델의 결정 경계(decision boundary) 근처에 있거나 데이터셋 내에서 희소한 특성을 가졌을 가능성이 높다는 가설이 실험적으로 입증되었다. 반면, 섭동 영향이 적은 LPI 샘플을 레이블링하는 것은 모델 성능 향상에 거의 기여하지 못한다는 점은 레이블링 자원의 효율적 배분이 얼마나 중요한지를 시사한다.

다만, 본 연구에서는 Flip 기반의 단순 섭동만을 사용하였는데, 더 복잡한 의료 영상의 특성을 반영할 수 있는 다른 형태의 섭동(예: 강도 변화, 가우시안 노이즈 등)이 성능에 어떤 영향을 미칠지는 명시적으로 다루지 않았다. 또한, 초기 학습 세트(10%)의 구성에 따라 초기 모델의 편향이 발생할 수 있으나, 저자들은 이를 위해 층화 5-겹 교차 검증(Stratified 5-fold CV)을 통해 객관성을 확보하려 노력하였다.

## 📌 TL;DR

본 논문은 의료 영상의 레이블링 비용을 줄이기 위해 **섭동 일관성(Perturbation Consistency)** 기반의 Active Learning 방법론인 **PCDAL**을 제안한다. 이미지를 Flip 하여 예측 값의 변화량(MSE 기반)을 측정하고, 변화가 큰 샘플을 우선적으로 레이블링하는 단순하면서도 강력한 구조를 가진다. 이 방법은 2D 분류 및 분할, 3D 분할 작업 모두에서 랜덤 샘플링 및 기존 AL 방법론보다 적은 데이터로 더 높은 성능을 달성하였으며, 의료 영상 분석의 데이터 효율성을 크게 향상시킬 가능성을 보여주었다.