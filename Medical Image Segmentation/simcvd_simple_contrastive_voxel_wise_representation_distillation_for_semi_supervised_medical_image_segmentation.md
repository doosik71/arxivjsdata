# SimCVD: Simple Contrastive Voxel-Wise Representation Distillation for Semi-Supervised Medical Image Segmentation

Chenyu You, Yuan Zhou, Ruihan Zhao, Lawrence Staib, James S. Duncan (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)에서 발생하는 **레이블링 데이터의 부족 문제**를 해결하고자 한다. 딥러닝 기반의 분할 모델은 높은 성능을 보이지만, 전문의의 수동 레이블링에 의존해야 하므로 대량의 데이터를 확보하는 데 막대한 비용과 시간이 소요된다.

기존의 준지도 학습(Semi-supervised learning) 접근 방식들은 다음과 같은 한계점을 가지고 있다. 첫째, 완전 지도 학습(Fully-supervised) 방식에 비해 강건성(Robustness)이 떨어진다. 둘째, 객체의 경계(Boundary)와 같은 기하학적 구조(Geometric structure)와 세만틱 정보에 대한 명시적인 모델링이 부족하여 분할 정확도가 제한된다. 마지막으로, 적은 양의 데이터로 깊은 모델을 학습시킬 때 발생하는 과적합(Over-fitting)과 공적응(Co-adapting) 문제로 인해 일반화 능력이 저하된다.

따라서 본 연구의 목표는 극소량의 레이블링 데이터만으로도 의료 영상의 복셀 단위 표현 학습(Voxel-wise representation learning)을 고도화하여, 기하학적 구조를 잘 반영하는 강건한 분할 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문은 **SimCVD**라는 단순한 대조적 증류(Contrastive Distillation) 프레임워크를 제안한다. 핵심 아이디어는 다음과 같다.

1. **경계 인식 표현 학습(Boundary-aware Representation Learning)**: 단순한 분류를 넘어 Signed Distance Map(SDM)을 함께 예측하는 멀티태스크 학습을 통해 객체의 기하학적 형상 정보를 명시적으로 학습한다.
2. **대조적 증류(Contrastive Distillation)**: Mean-Teacher 프레임워크 내에서 두 가지 서로 다른 Dropout 마스크를 적용한 뷰(View)를 생성하고, 이를 대조 학습(Contrastive Learning)시켜 경계 인식 지식을 증류한다.
3. **쌍별 구조 증류(Pair-wise Structural Distillation)**: 복셀 간의 상대적 유사성을 보존하는 쌍별 증류 방식을 도입하여, 단순한 픽셀 단위 일치를 넘어 구조적인 지식을 전수함으로써 일반화 성능을 높인다.
4. **Dropout의 재발견**: 복잡한 데이터 증강 기법보다 단순한 Dropout이 표현 붕괴(Representation collapse)를 방지하고 모델의 강건성을 높이는 최소한의 효율적인 데이터 증강 역할을 수행함을 입증하였다.

## 📎 Related Works

### 준지도 의료 영상 분할 (Semi-Supervised Medical Image Segmentation)

기존 연구들은 Mean-Teacher 프레임워크나 불확실성 맵(Uncertainty map), 혹은 경계 예측을 위한 Signed Distance Fields 등을 사용하여 레이블이 없는 데이터를 활용해 왔다. 그러나 SimCVD는 더 극단적인 적은 레이블 설정(Extreme few-annotation setting)에서도 높은 정확도를 달성하는 것을 목표로 하며, 더 명시적인 기하학적 모델링을 수행한다.

### 대조 학습 (Contrastive Learning)

대조 학습은 유사한 샘플(Positive)은 가깝게, 서로 다른 샘플(Negative)은 멀게 밀어내어 유용한 표현을 학습하는 방식이다. 기존 의료 영상 분야의 대조 학습은 수동 개입이 필요하거나 학습 시간이 매우 길다는 단점이 있었다. SimCVD는 이를 end-to-end 방식으로 통합하여 효율적으로 객체 경계를 인식하도록 설계되었다.

### 지식 증류 (Knowledge Distillation)

일반적인 지식 증류는 교사(Teacher) 모델의 출력 분포를 학생(Student) 모델이 따라하게 하여 과적합을 방지한다. 기존의 의료 영상 증류 방식은 주로 복셀 단위의 분류 문제로 접근했으나, SimCVD는 이를 구조적 예측 문제로 정의하고 인코더 특징 맵의 쌍별 관계(Relational similarity)를 매칭하는 방식을 취한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

SimCVD는 **Mean-Teacher** 구조를 기반으로 한다. 학생 네트워크($F_s$)와 교사 네트워크($F_t$)가 존재하며, 교사의 가중치는 학생 네트워크 가중치의 지수 이동 평균(Exponential Moving Average, EMA)으로 업데이트된다.

모델의 기본 아키텍처는 **V-Net**을 백본으로 사용하며, 두 가지 작업을 동시에 수행하는 멀티태스크 구조이다.

- **분류 브랜치**: 객체의 확률 맵(Probability map) $Q$를 예측한다.
- **회귀 브랜치**: 객체 경계로부터의 거리를 나타내는 Signed Distance Map(SDM) $Q_{sdm}$을 예측한다.

### 손실 함수 및 학습 절차

전체 학습 목적 함수는 다음과 같이 정의된다.
$$L = L^{sup} + \lambda L^{contrast} + \beta L^{pd} + \gamma L^{con}$$

#### 1. 지도 학습 손실 ($L^{sup}$)

레이블링된 데이터에 대해 분할 손실(Dice + Cross-entropy)과 SDM에 대한 평균 제곱 오차(MSE) 손실을 결합하여 사용한다.
$$L^{sup} = \frac{1}{N} \sum_{i=1}^{N} L^{seg}(Q^s_i, Y_i) + \frac{\alpha}{N} \sum_{i=1}^{N} L^{mse}(Q^{s,sdm}_i, Y^{sdm}_i)$$

#### 2. 경계 인식 대조 손실 ($L^{contrast}$)

레이블이 없는 데이터에 대해, 입력 볼륨 $X$와 예측된 SDM $Q_{sdm}$을 더해 경계 인식 특징($Q_{ba} = X + Q_{sdm}$)을 생성한다. 이후 두 개의 독립적인 Dropout 마스크를 적용하여 긍정 쌍(Positive pair, 동일 위치의 슬라이스)과 부정 쌍(Negative pair, 다른 위치나 다른 이미지의 슬라이스)을 구분하는 InfoNCE 손실을 적용한다.
$$L(h^t_{i,j}, h^s_{i,j}) = -\log \frac{\exp(h^t_{i,j} \cdot h^s_{i,j} / \tau)}{\sum_{k,l} \exp(h^t_{i,j} \cdot h^s_{k,l} / \tau)}$$
여기서 $h$는 Projection Head(MLP)를 통과한 특징 벡터이다.

#### 3. 쌍별 증류 손실 ($L^{pd}$)

인코더에서 추출된 특징 맵의 복셀 간 코사인 유사성을 비교하여 구조적 지식을 증류한다.
$$L^{pd} = -\frac{1}{M} \sum_{i=N+1}^{N+M} \sum_{j=1}^{H'W'D'} \log \frac{\exp(s(v^s_{i,j}, v^t_{i,j}))}{\sum_{k} \exp(s(v^s_{i,j}, v^t_{i,k}))}$$
여기서 $s(v_1, v_2)$는 두 벡터 간의 코사인 유사도이다.

#### 4. 일관성 손실 ($L^{con}$)

입력 데이터에 노이즈 $\eta$를 추가한 후, 학생과 교사 네트워크의 출력값 사이의 MSE를 최소화하여 학습의 안정성을 높인다.
$$L^{con} = \frac{1}{M} \sum_{i=N+1}^{N+M} L^{mse}(F_s(X^s_i + \eta^s_i), F_t(X^t_i + \eta^t_i))$$

## 📊 Results

### 실험 설정

- **데이터셋**: Left Atrium(LA) MR 데이터셋, NIH Pancreas CT 데이터셋.
- **평가 지표**: Dice coefficient, Jaccard Index, Average Symmetric Surface Distance(ASD), 95% Hausdorff Distance(95HD).
- **설정**: 레이블 비율을 20%와 10%로 설정하여 기존 SOTA 준지도 학습 모델들과 비교하였다.

### 정량적 결과

- **LA 데이터셋**: 20% 레이블 설정에서 Dice 점수 90.85%, Jaccard 점수 83.80%를 달성하여 이전 최고 기록을 각각 0.91%, 1.98% 경신하였다. 특히 10% 레이블 설정에서도 Dice 89.03%를 기록하며 매우 적은 데이터로도 높은 성능을 보였다.
- **Pancreas 데이터셋**: Dice 점수 기준 기존 방법들보다 최대 6.72%의 절대적 향상을 보이며 모든 지표에서 최상위 성능을 기록하였다.

### 정성적 결과

시각화 결과, SimCVD는 타 모델(UA-MT, ICT, DTC 등)에 비해 객체의 경계선(Border)과 전체적인 형상(Shape)을 훨씬 더 정확하게 예측하는 것으로 나타났다.

## 🧠 Insights & Discussion

### 강점 및 분석

1. **기하학적 제약의 효과**: SDM을 함께 학습시키는 것이 경계 인식 능력을 획기적으로 높임을 확인하였다. Ablation study 결과, SDM을 제거했을 때 Dice 점수가 하락하여 기하학적 정보가 좋은 Prior 역할을 함이 입증되었다.
2. **Dropout의 역할**: 본 논문은 Dropout이 단순한 정규화 도구를 넘어, '최소한의 데이터 증강(Minimal data augmentation)'으로 작용함을 발견하였다. In-painting이나 Local shuffle pixel 같은 복잡한 증강 기법은 오히려 노이즈를 유발하여 표현 붕괴를 일으키지만, Dropout 마스크는 효율적으로 긍정/부정 쌍을 구분하게 하여 강건성을 높였다.
3. **구조적 증류의 필요성**: 픽셀 단위의 일관성뿐만 아니라 인코더 레벨에서의 쌍별(Pair-wise) 유사성을 맞추는 것이 공간적 레이블링 일관성을 높이는 데 기여하였다.

### 한계 및 비판적 해석

논문에서는 제안 방법이 매우 효과적이라고 주장하지만, 대부분의 실험이 3D 볼륨 데이터의 슬라이스 단위 대조 학습으로 이루어졌다. 3D 전체 구조를 한 번에 고려하는 Contrastive Learning과의 비교 분석이 부족하며, 사용된 하이퍼파라미터($\lambda, \beta, \gamma$)의 민감도 분석이 상세히 제시되지 않았다. 또한, 다중 클래스(Multi-class) 분할 문제로의 확장 가능성에 대해서는 언급만 되었을 뿐 실제 검증은 이루어지지 않았다.

## 📌 TL;DR

본 논문은 적은 레이블 데이터로도 고성능 의료 영상 분할을 가능케 하는 **SimCVD** 프레임워크를 제안한다. **SDM(Signed Distance Map)**을 이용한 경계 인식 학습, **Dropout 마스크 기반의 대조적 증류**, 그리고 **복셀 간 쌍별 구조 증류**를 결합하여, 준지도 학습 환경에서도 완전 지도 학습에 근접하는 정밀한 분할 성능을 달성하였다. 특히 단순한 Dropout이 의료 영상의 표현 학습에서 매우 효율적인 증강 기법이 될 수 있음을 시사하며, 향후 의료 영상 합성 및 등록(Registration) 등 다양한 하위 작업으로의 확장 가능성을 보여주었다.
