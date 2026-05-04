# Unsupervised bias discovery in medical image segmentation

Nicolás Gaggion, Rodrigo Echeveste, Lucas Mansilla, Diego H. Milone, and Enzo Ferrante (2023)

## 🧩 Problem to Solve

의료 영상의 해부학적 분할(anatomical segmentation)을 위한 딥러닝 모델은 성별이나 인종과 같은 보호 속성(protected attributes)에 따라 특정 하위 집단에 대해 편향(bias)을 보일 수 있다. 이러한 모델의 공정성(fairness)을 감사하는 것은 매우 중요하지만, 일반적으로는 대상 집단에 대한 정답 마스크(ground-truth segmentation masks)가 필요하다는 제약이 있다.

실제 배포 환경이나 새로운 대상 집단에 모델을 적용할 때는 전문가의 어노테이션 비용이 매우 높거나 개인정보 보호 문제로 인해 정답 데이터를 확보하기 어려운 경우가 많다. 따라서 본 논문은 정답 마스크가 없는 상황에서도 의료 영상 분할 모델의 편향을 사전에 발견할 수 있는 비지도 편향 발견(Unsupervised Bias Discovery, UBD) 방법을 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 정답지가 없는 데이터의 분할 품질을 추정하기 위해 **Reverse Classification Accuracy (RCA)** 프레임워크를 활용하는 것이다. 

RCA는 추론된 분할 결과물을 가상의 정답으로 간주하고 이를 역으로 활용하여 품질을 측정하는 방식이다. 저자들은 이를 의료 영상 분할 영역에 적용하여, 서로 다른 인구통계학적 그룹 간의 RCA 점수 차이를 계산함으로써 정답 데이터 없이도 모델의 성능 격차, 즉 편향의 존재 여부와 방향성을 감지할 수 있음을 입증하였다.

## 📎 Related Works

기존에는 정답 마스크 없이 분할 성능을 추정하기 위해 다음과 같은 방법들이 제안되었다.
- **Predictive Uncertainty**: 예측의 불확실성이 높은 픽셀이 오류일 가능성이 높다는 가설에 기반한다. 그러나 이는 모델 자체의 불확실성 추정 능력에 크게 의존한다는 한계가 있다.
- **Learning-based Approach**: 영상과 예측 마스크 쌍을 입력받아 Dice-Sørensen coefficient (DSC) 점수를 직접 예측하는 별도의 CNN을 학습시키는 방법이다. 이는 모델에 구애받지 않는 장점이 있지만, DSC 예측을 위한 추가적인 CNN 학습이 필요하다.

본 논문은 Valindria 등이 제안한 RCA 방식을 기반으로 한다. RCA는 단일 테스트 영상과 그 예측 마스크를 사용하여 역 분류기를 구성하고, 이를 정답이 존재하는 참조 데이터셋에서 평가하여 품질을 추정한다. 본 연구는 기존 RCA를 확장하여 딥러닝 기반의 등록(registration) 네트워크를 결합한 효율적인 변형 모델을 제안하며, 이를 UBD에 처음으로 적용하였다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 RCA 프레임워크
대상 이미지 $I$와 모델 $M$에 의해 예측된 분할 마스크 $S_I$가 있을 때, 정답 데이터가 없는 상태에서 $S_I$의 품질을 추정하는 과정은 다음과 같다.

1. **참조 데이터베이스 구축**: 정답 마스크 $S_{GT}^{J_i}$가 존재하는 참조 이미지 집합 $\{J, S_{GT}^J\}_k$를 준비한다.
2. **이미지 등록 및 라벨 전파(Label Propagation)**: 이미지 $I$를 '아틀라스(atlas)'로 설정하고, 이를 참조 이미지 $J_i$에 정렬시키기 위한 변형 필드(deformation field) $D_i$를 계산한다. 이 변형 필드를 $S_I$에 적용하여 참조 이미지 $J_i$에 대응하는 후보 마스크 $\hat{S}_{J_i} = S_I \circ D_i$를 생성한다.
3. **품질 추정**: 생성된 후보 마스크 $\hat{S}_{J_i}$와 참조 데이터의 실제 정답 마스크 $S_{GT}^{J_i}$ 간의 DSC를 계산하여 평균을 낸다.

이를 수식으로 표현하면 다음과 같다.
$$\text{DSC}_{\text{RCA}}(I, S_I, \{J, S_{GT}^J\}_k) = \frac{\sum_{k} \text{DSC}(\hat{S}_{J_k}, S_{GT}_{J_k})}{k}$$

### Deep Registration Networks를 이용한 구현
계산 시간을 단축하고 정확도를 높이기 위해 저자들은 다음과 같은 딥러닝 기반 등록 네트워크를 제안한다.
- **Top-k Selection**: 모든 참조 이미지와 등록을 수행하는 대신, 아틀라스와 가장 유사한 상위 $k$개의 이미지만 선택한다.
- **Affine Registration**: Siamese 아키텍처의 CNN 인코더와 완전 연결 계층(FC layers)을 통해 2D 아핀 변환 파라미터를 빠르게 학습하여 1차 정렬을 수행한다.
- **Dense Registration**: ACNN-RegNet(Anatomically Constrained Deformable Registration Network)을 사용하여 세밀한 비정형 변형 필드를 추정하고 최종적으로 마스크를 워핑(warp)한다.

### 비지도 편향 발견 (UBD) 절차
특정 인구통계학적 속성 $A$(예: 성별)에 대한 편향을 측정하기 위해, 각 그룹의 $\text{DSC}_{\text{RCA}}$ 평균값을 비교한다. 성별을 예로 들면, 남성 그룹($A=M$)과 여성 그룹($A=F$)의 점수 차이인 $\Delta \text{DSC}_{\text{RCA}}$를 계산한다.
$$\Delta \text{DSC}_{\text{RCA}} = \text{DSC}_{\text{RCA}}^{A=M} - \text{DSC}_{\text{RCA}}^{A=F}$$
이 값의 부호와 크기를 통해 어떤 집단에 대해 모델이 더 편향되어 있는지(성능이 더 낮은지)를 판단한다.

## 📊 Results

### 실험 설정
- **작업**: 흉부 X-ray 영상에서의 폐(lung) 및 심장(heart) 분할.
- **데이터셋**: JSRT, Montgomery, Shenzhen, Padchest 등 총 911장의 이미지.
- **모델**: UNet (Soft Dice 및 Cross-entropy 손실 함수 사용).
- **지표**: 실제 $\Delta \text{DSC}$와 추정된 $\Delta \text{DSC}_{\text{RCA}}$ 간의 상관관계.

### 정량적 결과
1. **합성 실험 (Synthetic Experiment)**: 
   다양한 학습 단계의 체크포인트 모델들을 조합하여 인위적으로 성능 차이를 만들어낸 후 검증하였다. 실험 결과, $\text{DSC}_{\text{RCA}}$는 실제 편향과 매우 높은 상관관계를 보였다. 특히 편향의 크기가 0.02 이상인 경우, 폐 분할에서는 100%, 심장 분할에서는 96%의 확률로 편향의 방향(부호)을 정확히 예측하였다.

2. **실제 실험 (Real Experiment)**: 
   남성 데이터로만 학습한 모델과 여성 데이터로만 학습한 모델을 구축하여 실제 편향을 측정하였다. 흥미롭게도 학습 데이터의 구성과 상관없이 모델들이 대체로 여성 환자에게서 더 높은 성능을 보이는 경향이 발견되었다. $\Delta \text{DSC}_{\text{RCA}}$는 이러한 실제 성능 격차를 강하게 반영하며 상관관계를 보였다.

## 🧠 Insights & Discussion

본 연구는 정답 데이터가 부족한 실제 임상 환경에서 모델의 공정성을 모니터링할 수 있는 실질적인 도구를 제시하였다. 특히 RCA 기반의 UBD가 정답 없이도 편향의 방향성을 정확히 짚어낼 수 있다는 점은 모델 배포 후의 유지보수 단계에서 매우 유용할 것으로 판단된다.

다만, 실제 실험에서 $\Delta \text{DSC}_{\text{RCA}}$와 실제 $\Delta \text{DSC}$ 사이의 상관관계 직선의 기울기가 1이 아니라는 점이 관찰되었다. 이는 $\text{DSC}_{\text{RCA}}$가 편향의 존재와 방향은 잘 알려주지만, 편향의 절대적인 크기를 정확히 측정하기 위해서는 추가적인 캘리브레이션(calibration) 과정이 필요함을 시사한다. 또한, 본 연구는 성별이라는 단일 속성에 집중하였으나, 실제로는 인종, 연령 등 여러 속성이 복합적으로 작용하는 교차 편향(intersectional bias)에 대한 분석이 추가로 필요할 것이다.

## 📌 TL;DR

본 논문은 정답 마스크가 없는 의료 영상 분할 모델의 편향을 탐지하기 위해 **Reverse Classification Accuracy (RCA)**를 이용한 비지도 편향 발견(UBD) 방법을 제안하였다. 딥러닝 기반 등록 네트워크를 통해 효율적으로 구현된 이 방법은 흉부 X-ray 분할 실험에서 실제 성능 격차와 매우 높은 상관관계를 보였으며, 정답 데이터 없이도 모델의 공정성 문제를 사전에 감지할 수 있음을 입증하였다. 이는 향후 의료 AI 모델의 안전한 배포 및 실시간 공정성 모니터링에 중요한 기여를 할 가능성이 높다.