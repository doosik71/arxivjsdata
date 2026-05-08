# DOMINO: Domain-aware Model Calibration in Medical Image Segmentation

Skylar E. Stolte et al. (2022)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델의 예측 확률(predicted probability)과 실제 정답 확률(true correctness likelihood) 사이의 일치 정도를 나타내는 모델 Calibration(교정) 문제를 다룬다. 현대의 심층 신경망(DNN)은 높은 정확도를 보임에도 불구하고 Calibration 성능이 떨어지는 경향이 있으며, 이는 모델이 자신의 예측에 대해 지나치게 확신하는 과잉 확신(overconfidence) 문제로 이어진다.

이러한 문제는 특히 의료 영상 분할(Medical Image Segmentation) 분야에서 치명적이다. 조직 경계의 자연스러운 불확실성과 다수 클래스에 편향된 손실 함수(loss function)의 특성으로 인해 모델이 잘못된 예측을 하면서도 높은 확신도를 갖게 된다. 이는 뇌 부피 측정 오류나 비침습적 뇌 자극(non-invasive brain stimulation) 파라미터 추정 오류와 같이 임상적으로 위험한 결과로 이어질 수 있다. 따라서 본 연구의 목표는 클래스 간의 의미적 혼동 가능성(semantic confusability)과 계층적 유사성(hierarchical similarity)이라는 도메인 지식을 활용하여, 신뢰할 수 있고 정확한 의료 영상 분할 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 클래스 레이블 간의 의미적 관계를 반영하는 **Domain-aware Regularization(도메인 인식 정규화)**을 손실 함수에 도입하는 것이다.

기존의 많은 연구들은 클래스 간의 평균을 서로 직교(orthogonal)하게 만들어 클래스들을 완전히 분리하려 노력했지만, 본 연구는 실제 의료 영상 데이터에서 특정 조직(클래스)들은 자연스럽게 유사한 특성을 가진다는 점에 주목하였다. 즉, 모델이 모든 클래스를 억지로 분리하려 하기보다, 유사한 클래스 간의 혼동은 어느 정도 허용하고 서로 다른 클래스 간의 혼동은 강하게 처벌하는 방식의 정규화를 통해 더 나은 클래스 표현(representation)을 학습하고 Calibration 성능을 높일 수 있다는 직관을 제시한다.

## 📎 Related Works

논문에서는 기존의 의료 영상 분할 모델들이 주로 표준 손실 함수(예: Cross-Entropy)에 의존하고 있음을 지적한다. Cross-Entropy 손실 함수는 정답 레이블의 출력값을 최대화하는 특성이 있어, 모델이 정답 클래스의 로짓(logit)을 오답 클래스보다 훨씬 크게 높이게 만들어 결국 과잉 확신(overconfidence) 문제를 야기한다.

또한, 클래스 간의 독립성을 강조하며 클래스 평균을 직교하게 만드는 기존 접근 방식들과 달리, 본 연구는 도메인 지식을 활용하여 클래스 간의 '의미적 유대'를 인정하는 것이 더 신뢰할 수 있는 모델을 만드는 길이라고 주장하며 기존 방식과의 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 및 아키텍처

본 연구는 기본 모델로 **UNETR (U-Net Transformers)**를 사용한다. UNETR은 Transformer 인코더를 통해 이미지의 전역적 문맥(global context)을 학습하고, 이를 전통적인 FCN(Fully Convolutional Network) 디코더의 로컬 정보와 결합하여 3D 의료 영상 분할을 수행하는 구조이다.

### Domain-aware Loss Regularization

모델의 Calibration을 개선하기 위해 기존의 손실 함수에 도메인 인식 정규화 항을 추가한다. 전체 손실 함수는 다음과 같이 정의된다.

$$L_{total} = L(y, \hat{y}) + \beta(y')(W)(\hat{y})$$

여기서 $L$은 기본 손실 함수로 Dice Loss와 Cross-Entropy가 결합된 $\text{DiceCE}$를 사용하며, $y$는 원-핫 인코딩된 정답 레이블, $\hat{y}$는 Softmax를 통과한 예측 확률이다. $\beta$는 정규화의 강도를 조절하는 하이퍼파라미터이다. 핵심은 $N \times N$ 크기의 정규화 행렬 $W$로, 대각 성분은 0이며 비대각 성분은 클래스 $i$를 $j$로 혼동했을 때의 페널티 값을 의미한다.

본 논문은 $W$를 설계하기 위한 두 가지 접근 방식을 제안한다.

**1. Confusion Matrix 기반 정규화 (UNETR-CM)**
데이터 기반의 자동 정규화 방식이다. 먼저 정규화가 없는 기본 모델(UNETR-Base)을 학습시킨 후, 검증 데이터셋(validation set)에서 생성된 혼동 행렬(Confusion Matrix)을 활용하여 $W$를 계산한다.

$$W_{ij} = S \cdot \frac{I_i - C_{ij}}{I_i}$$

여기서 $C$는 클래스 빈도로 정규화된 혼동 행렬, $I_i$는 단위 행렬의 $i$번째 행이며, $S$는 정규화 가중치를 조절하는 스케일링 팩터이다. 즉, 모델이 원래 자주 혼동하던 클래스 쌍에는 낮은 페널티를, 거의 혼동하지 않던 쌍에는 높은 페널티를 부여하여 자연스러운 혼동을 유도한다.

**2. Hierarchical Classes 기반 정규화 (UNETR-HC)**
전문가의 도메인 지식을 활용한 수동 정규화 방식이다. 조직의 생물학적 특성에 따라 클래스들을 계층적으로 그룹화한다. 예를 들어, 'Bone' 그룹에는 'Cancellous bone'과 'Cortical bone'을 포함시키고, 'Soft tissue' 그룹에는 'Skin', 'Fat', 'Muscle' 등을 포함시킨다. 동일한 상위 그룹에 속한 클래스 간의 혼동에는 낮은 페널티를 부여하고, 다른 그룹 간의 혼동에는 높은 페널티를 부여하는 행렬 $W$를 직접 설계하여 적용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 65-89세 성인 113명의 T1-weighted MRI 영상.
- **작업**: 11개 조직 분할(Full task) 및 6개 조직 분할(Reduced task).
- **비교 대상**: UNETR-Base, UNETR-CM, UNETR-HC, 그리고 기존 표준 소프트웨어인 Headreco.
- **평가 지표**: Dice Score (중첩도), Hausdorff Distance (경계 거리), Top-N Accuracy (예측 확률 상위 N개 내 정답 포함 여부), Calibration Curves (신뢰도 도표).

### 주요 결과

**1. 11-클래스 분할 성능**

- **정량적 결과**: UNETR-CM이 Dice Score와 Top-N 정확도에서 가장 우수한 성능을 보였다. Hausdorff Distance의 경우 UNETR-CM과 UNETR-HC 모두 기본 모델(UNETR-Base)보다 뛰어난 성능을 기록했다.
- **정성적 결과**: UNETR-HC가 회백질(GM)과 뇌척수액(CSF) 사이의 미세한 경계를 가장 잘 포착하는 것으로 나타났다.

**2. 6-클래스 분할 및 Headreco 비교**

- **정확도**: DOMINO 방법론(CM, HC)은 모든 조직 유형에서 Headreco와 비슷하거나 더 우수한 성능을 보였으며, 특히 Air, Bone, Soft tissue에서 더 높은 성능을 기록했다.
- **Calibration**: Calibration Curve 분석 결과, DOMINO 모델들이 UNETR-Base보다 훨씬 더 잘 교정되었으며, 백질(WM), 뇌척수액(CSF), 뼈, 연부 조직에서 Headreco보다 뛰어난 Calibration 성능을 보였다.

## 🧠 Insights & Discussion

본 연구는 모델의 성능(Accuracy)과 Calibration 사이의 트레이드오프 관계를 극복하고, 두 지표를 동시에 향상시킬 수 있음을 보여주었다. 특히 Top-N 정확도의 상승은 모델이 단순히 정답을 맞히는 것을 넘어, 틀리더라도 도메인 관점에서 '그럴듯한(reasonable)' 오답을 내놓고 있음을 시사한다.

**강점 및 의의**:

- 도메인 지식을 손실 함수에 직접 통합함으로써 모델의 신뢰성을 높였다.
- UNETR-HC의 경우, 6-클래스 작업으로 전환 시 재학습 없이도 경계 분할 성능이 개선됨을 보여주어 방법론의 유연성을 입증했다.

**한계 및 논의**:

- $W$ 행렬의 하이퍼파라미터($\beta, S$) 설정이 모델과 데이터셋에 따라 달라질 수 있으며, 이에 대한 최적화 과정이 필요하다.
- UNETR-HC의 경우 페널티 행렬을 수동으로 설계하므로 주관성이 개입될 수 있다는 한계가 있다.

## 📌 TL;DR

본 논문은 의료 영상 분할 모델의 과잉 확신 문제를 해결하기 위해 클래스 간의 의미적 유사성을 활용한 **DOMINO(Domain-aware Model Calibration)** 프레임워크를 제안한다. 혼동 행렬(CM)과 계층적 구조(HC)를 이용한 정규화 항을 손실 함수에 추가함으로써, 모델의 분할 정확도와 Calibration 성능을 동시에 향상시켰다. 이 연구는 의료 AI의 신뢰성과 안전성을 높이는 데 기여할 수 있으며, 향후 다양한 의료 진단 및 분할 작업에 적용될 가능성이 높다.
