# One-Shot Semantic Segmentation

Amirreza Shaban, Shray Bansal, Zhen Liu, Irfan Essa, Byron Boots (2017)

## 🧩 Problem to Solve

본 논문은 매우 제한된 데이터, 특히 단 하나의 레이블링된 이미지(support set)만을 사용하여 새로운 시맨틱 클래스에 대한 픽셀 단위의 세그멘테이션 마스크를 생성하는 **One-Shot Semantic Segmentation** 문제를 해결하고자 한다.

일반적인 딥러닝 기반의 시맨틱 세그멘테이션 모델은 방대한 양의 픽셀 단위 주석(pixel-level annotation)이 필요하며, 새로운 클래스에 적응하기 위해 기존 모델을 미세 조정(fine-tuning)하는 방식은 파라미터 수가 너무 많아 단일 이미지에 대해 심각한 과적합(overfitting) 문제를 일으킨다. 또한, 이미지 분류(classification)에서 사용되는 One-shot learning 기법을 세그멘테이션에 그대로 적용하기에는, 단일 이미지에서 생성되는 고밀도 특징(dense features)의 수가 너무 많아 계산 효율성과 확장성 문제가 발생한다. 따라서 본 연구의 목표는 적은 데이터로도 효율적으로 새로운 클래스를 세그멘테이션할 수 있는 메타 학습(meta-learning) 기반의 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **조건부 파라미터 생성(Conditional Parameter Generation)**을 위한 **두 개의 분기(two-branched) 구조**를 설계한 것이다.

기존의 미세 조정 방식이 경사 하강법(SGD)을 통해 반복적으로 파라미터를 업데이트하는 것과 달리, 본 모델은 하나의 분기가 서포트 세트를 입력받아 세그멘테이션에 필요한 파라미터를 직접 예측하는 함수 역할을 수행한다. 즉, 네트워크가 "어떻게 학습해야 하는지"를 학습하는 메타 학습 방식을 채택하여, 단 한 번의 순전파(forward pass)만으로 새로운 클래스에 최적화된 분류기 파라미터를 생성하고 이를 쿼리 이미지에 적용함으로써 속도와 일반화 성능을 동시에 확보하였다.

## 📎 Related Works

### 기존 연구 및 한계

1. **FCN (Fully Convolutional Networks):** 픽셀 단위 분류를 통해 세그멘테이션을 수행하지만, 테스트 클래스에 대한 대규모 학습 데이터가 필요하다는 전제가 있다.
2. **약지도 학습 (Weak Supervision):** 바운딩 박스나 이미지 레벨 레이블을 사용하지만, 여전히 클래스당 많은 수의 약한 레이블(weak labels)을 요구하는 경우가 많다.
3. **Co-segmentation:** 동일 클래스의 이미지들 사이에서 공통 객체를 찾아내지만, 주로 비지도 학습 기반이거나 특정 시각적 유사성에 의존하며 정밀한 픽셀 주석의 이점을 충분히 활용하지 못한다.
4. **Few-Shot Learning (Classification):** 이미지 분류에서는 성공적이었으나, 이를 세그멘테이션에 적용할 경우 고차원의 밀집 특징(dense features)으로 인해 계산 복잡도가 급증하며, Siamese Network 같은 거리 기반 매칭 방식은 메모리 및 계산 비용 문제가 발생한다.

### 차별점

본 연구는 단순한 특징 매칭이나 반복적인 미세 조정 대신, 서포트 세트로부터 **동적 파라미터(dynamic parameters)**를 생성하여 쿼리 이미지의 특징 공간에 즉각적으로 적용하는 방식을 제안함으로써 과적합을 방지하고 추론 속도를 획기적으로 높였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 크게 **Conditioning Branch**와 **Segmentation Branch** 두 개의 네트워크로 구성된다.

1. **Conditioning Branch (파라미터 생성 분기):**
   - 입력: 서포트 세트 $S = (I_s, Y_s(l))$ (이미지와 해당 클래스의 이진 마스크).
   - 처리: 입력 이미지를 마스크와 곱하여 타겟 객체만 남긴 뒤, VGG-16 네트워크를 통과시킨다.
   - 출력: 픽셀 단위 로지스틱 회귀에 사용될 가중치 $w$와 편향 $b$를 생성한다.
   - **Weight Hashing:** VGG의 출력(1000차원)을 로지스틱 회귀 파라미터(4097차원)로 확장하기 위해 Weight Hashing 레이어를 사용한다. 이는 완전 연결 계층(fully connected layer)을 사용할 때 발생하는 과적합을 방지하고 파라미터 분산을 줄이기 위함이다.

2. **Segmentation Branch (특징 추출 분기):**
   - 입력: 쿼리 이미지 $I_q$.
   - 처리: FCN-32s (또는 Dilated-FCN) 아키텍처를 사용하여 고밀도 특징 볼륨 $F_q = \phi_{\zeta}(I_q)$를 추출한다.
   - 여기서 $F_{mn}^q$는 공간 위치 $(m, n)$에서의 특징 벡터이다.

### 예측 절차 및 방정식

추출된 쿼리 이미지의 특징 $F_{mn}^q$와 Conditioning Branch에서 생성된 파라미터 $\{w, b\}$를 이용하여 다음과 같이 픽셀 단위 분류를 수행한다.

$$\hat{M}_{mn}^q = \sigma(w^T F_{mn}^q + b)$$

여기서 $\sigma(\cdot)$는 시그모이드 함수이며, $\hat{M}_{mn}^q$는 해당 위치의 픽셀이 타겟 클래스일 확률이다. 최종 결과는 bilinear interpolation으로 업샘플링한 후 0.5 임계값을 적용하여 이진 마스크로 변환한다.

### 학습 절차

학습 시에는 훈련 데이터셋 $D_{train}$에서 서포트 세트 $S$, 쿼리 이미지 $I_q$, 정답 마스크 $M_q$를 무작위로 샘플링하여 다음의 로그 가능도(log-likelihood)를 최대화하도록 학습한다.

$$L(\eta, \zeta) = \mathbb{E}_{S, I_q, M_q \sim D_{train}} \left[ \sum_{m,n} \log p_{\eta, \zeta}(M_{mn}^q | I_q, S) \right]$$

- $\eta$는 Conditioning Branch의 파라미터, $\zeta$는 Segmentation Branch의 파라미터이다.
- VGG 네트워크가 더 빠르게 과적합되는 경향이 있어, $\eta$에 대한 학습률(learning rate)에 0.1의 승수를 적용하여 조절하였다.

### k-shot 확장

$k$개의 서포트 이미지가 주어진 경우, 각 이미지에 대해 독립적으로 $\{w_i, b_i\}$ 파라미터를 생성하여 $k$개의 예측 마스크를 만든다. 이후 이 마스크들을 **논리적 OR 연산(logical-OR operation)**으로 결합한다. 이는 각 단일 이미지 기반 분류기가 정밀도(precision)는 높으나 재현율(recall)이 낮다는 특성을 보완하기 위함이며, 추가적인 재학습 없이 $k$값에 유연하게 대응할 수 있다.

## 📊 Results

### 실험 설정

- **데이터셋:** PASCAL VOC 2012를 기반으로 한 $\text{PASCAL-5}^i$ 벤치마크를 사용한다. 20개 클래스를 5개씩 4개의 폴드(fold)로 나누어, 훈련에 사용되지 않은 클래스를 테스트 클래스로 설정한다.
- **평가 지표:** 클래스별 Intersection over Union (IoU)의 평균인 **meanIoU**를 측정한다.
- **비교 대상:** 1-NN, Logistic Regression, Fine-tuning, Co-segmentation, Siamese Network.

### 정량적 결과

1. **세그멘테이션 성능:** 1-shot 설정에서 제안 방법은 baseline 대비 **상대적 meanIoU를 약 25% 향상**시켰다. 구체적으로 1-shot meanIoU $40.8\%$를 기록하며, Fine-tuning($32.6\%$)이나 Siamese($31.4\%$)보다 우수한 성능을 보였다.
2. **k-shot 성능:** 5-shot 설정에서는 $43.9\%$의 meanIoU를 달성하여, 비지도 방식인 Co-segmentation($27.1\%$)을 크게 상회하였다.
3. **추론 속도:** 제안 방법은 1-shot에서 두 번째로 빠른 Logistic Regression보다 약 **$3\times$ 빠르며**, 5-shot에서는 약 **$10\times$ 더 빠른** 속도를 보였다. (1-shot 기준 추론 시간: 0.19s)

### 정성적 결과

제안 방법은 단 하나의 서포트 이미지만으로도 쿼리 이미지 내의 객체를 효과적으로 분리해내며, 서포트 세트의 클래스를 변경함에 따라 예측 마스크가 동적으로 변하는 조건부(conditioning) 특성이 잘 작동함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **메타 학습의 효율성:** 반복적인 최적화 과정 없이 단일 forward pass만으로 파라미터를 생성함으로써 추론 속도를 획기적으로 개선하였다.
- **사전 학습의 영향:** ImageNet으로 사전 학습된 가중치를 사용한 것이 성능 향상에 기여하였다. 특히, PASCAL 클래스와 겹치는 ImageNet 클래스를 모두 제거한 데이터셋($\text{AlexNet-771}$)으로 실험했을 때도 수렴 후 성능이 유사했다는 점은, 본 모델이 특정 클래스의 지식이 아니라 "새로운 클래스를 어떻게 학습할 것인가"에 대한 일반적인 능력을 습득했음을 시사한다.
- **메타 학습 vs 단순 분류기:** 단순 Logistic Regression은 약지도 레이블(weak labels) 없이는 성능이 저조했으나, 제안 모델은 메타 학습을 통해 레이블이 없는 클래스에 대해서도 높은 일반화 성능을 보였다.

### 한계 및 논의

- **학습 데이터의 제약:** 본 연구는 PASCAL-5라는 비교적 작은 규모의 벤치마크에서 검증되었다. 더 방대한 클래스를 가진 데이터셋에서도 동일한 효율성이 유지될지는 추가 검증이 필요하다.
- **가정:** 서포트 세트의 이미지가 타겟 객체의 대표성을 충분히 가지고 있다는 가정 하에 작동한다. $k$-shot 확장 시 OR 연산을 사용하는 단순한 방식이 $k$가 매우 커질 경우 노이즈를 증가시킬 가능성이 있다.

## 📌 TL;DR

본 논문은 단 하나의 레이블링된 이미지로 새로운 객체를 세그멘테이션하는 **One-Shot Semantic Segmentation**을 위한 메타 학습 프레임워크를 제안한다. 핵심은 **Conditioning Branch**가 서포트 세트로부터 분류기 파라미터를 직접 생성하고, 이를 **Segmentation Branch**의 고밀도 특징에 적용하는 구조이다. 이 방식은 기존의 Fine-tuning이나 거리 기반 매칭 방식보다 **과적합에 강하고 추론 속도가 훨씬 빠르며(최대 10배)**, PASCAL 데이터셋에서 기존 베이스라인 대비 **meanIoU를 약 25% 향상**시키는 성과를 거두었다. 이 연구는 데이터 효율적인 세그멘테이션 모델 설계에 중요한 방향성을 제시한다.
