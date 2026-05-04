# Robust Instance Segmentation through Reasoning about Multi-Object Occlusion

Xiaoding Yuan, Adam Kortylewski, Yihong Sun, Alan Yuille (2021)

## 🧩 Problem to Solve

본 논문은 복잡한 장면에서 여러 객체가 서로를 부분적으로 가리는 **Multi-Object Occlusion(다중 객체 폐색)** 상황에서의 인스턴스 세그멘테이션(Instance Segmentation) 문제를 해결하고자 한다.

일반적인 딥러닝 기반의 이미지 분석 방법론들은 이미지 내의 객체들을 독립적으로 처리하는 경향이 있으며, 인접한 객체 간의 상대적인 폐색 관계를 고려하지 않는다. 이로 인해 신경망은 사람이 인지하는 것보다 폐색된 객체를 인식하는 능력이 현저히 떨어지며, 특히 객체 간의 순서나 위치의 조합적 다양성으로 인해 성능 저하가 발생한다.

본 연구의 목표는 **Bounding Box 수준의 약한 지도 학습(Weakly-supervised learning)**만으로도 폐색에 강건한 다중 객체 인스턴스 세그멘테이션을 수행하는 딥러닝 네트워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Compositional Networks(CompNets)**의 생성 모델(Generative Model)을 다중 객체로 확장하고, 객체 간의 폐색 순서를 추론하여 세그멘테이션 오류를 스스로 교정하는 **Occlusion Reasoning Module (ORM)**을 도입한 것이다.

중심적인 직관은 동일한 픽셀에 대해 여러 객체가 자신이 전경(Foreground)이라고 주장하는 '충돌'이 발생했을 때, 픽셀 단위의 경쟁과 객체 단위의 폐색 순서 추론을 통해 실제 가려진 객체(Occludee)와 가리는 객체(Occluder)를 구분하여 마스크를 정밀하게 수정할 수 있다는 점이다.

## 📎 Related Works

### 관련 연구 및 한계
1. **Occlusion Reasoning**: 기존의 연구들은 바운딩 박스를 이용한 3D 관계 추론이나, Markov Random Field(MRF)를 이용한 확률 모델, 혹은 엣지 정보를 이용한 방법들을 제안하였다. 그러나 이러한 방법들은 세만틱 정보가 부족하거나, 알려지지 않은 클래스의 폐색(Unknown Occlusion)을 처리하는 능력이 부족하다는 한계가 있다.
2. **Weakly-supervised Instance Segmentation**: 픽셀 단위의 정밀한 어노테이션 비용을 줄이기 위해 이미지 레벨이나 바운딩 박스 레벨의 지도 학습을 사용하는 방법들이 제안되었다. 특히 **CompositionalNets**는 생성 모델을 통해 객체와 배경을 분리하여 폐색에 강건한 인식을 보여주었으나, 단일 객체 처리 위주로 설계되어 다중 객체 간의 상호작용을 무시한다는 단점이 있다.

### 차별점
본 논문은 픽셀 레벨의 폐색 추론과 객체 레벨의 폐색 순서 복원(Order Recovery)을 결합함으로써, 알려지지 않은 객체에 의한 폐색과 다중 객체 간의 복잡한 폐색 관계를 동시에 해결한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. Compositional Networks (Prior Work)
본 논문의 기반이 되는 CompNets는 분류 헤드를 미분 가능한 생성 모델로 대체한다. 특정 클래스 $y$에 대해 특징 맵 $F$의 생성 확률 $p(F|y)$를 다음과 같이 정의한다.

$$p(F|\Theta^y) = \sum_{m} \nu_m p(F|\theta^m_y)$$

여기서 $\nu_m$은 활성화된 혼합 성분을 나타내는 이진 변수이다. 각 픽셀 $i$의 특징 벡터 $f_i$에 대한 가능도(Likelihood)는 전경(Foreground)과 컨텍스트(Context)의 합으로 정의되며, von-Mises-Fisher(vMF) 분포를 사용하여 모델링한다.

$$p(f_i|A^m_{i,y}, \chi^m_{i,y}, \Lambda) = p(i|m,y)p(f_i|A^m_{i,y}, \Lambda) + (1-p(i|m,y))p(f_i|\chi^m_{i,y}, \Lambda)$$

또한, 폐색에 대응하기 위해 공간적 구조가 없는 **Outlier Model** $p(f_i|\beta, \Lambda)$를 도입하여, 객체 모델과 아웃라이어 모델 중 하나가 활성화되도록 설계한다.

### 2. Multi-Object Generative Model 확장
단일 객체 모델을 다중 객체로 확장하기 위해, 이미지 내 $N$개의 객체와 하나의 아웃라이어 모델을 포함하는 결합 가능도를 다음과 같이 정의한다.

$$p(F|\theta^m_{y_1}, \dots, \theta^m_{y_N}, \beta) = \prod_{i} \prod_{n=1}^{N+1} p^n(f_i)^{z_{i,n}}$$

단, $\sum_{n} z_{i,n} = 1$이며 $z_{i,n} \in \{0,1\}$이다. 즉, 특정 픽셀 $i$에서는 오직 하나의 객체 모델 혹은 아웃라이어 모델만이 활성화될 수 있음을 강제한다.

### 3. Occlusion Reasoning Module (ORM)
다중 객체 모델의 최적화는 복잡하므로, 본 논문은 다음과 같은 단계적 추론 절차를 제안한다.

**Step 1: Feed-forward extraction**
각 객체 바운딩 박스를 독립적으로 처리하여 클래스 예측 $\hat{y}$와 전경($F$), 컨텍스트($C$), 폐색($O$)에 대한 가능도 맵을 추출한다.

**Step 2: Pixel-level competition**
두 객체의 바운딩 박스가 겹치는 영역에서 두 모델 모두 전경이라고 예측하는 '충돌 세트 $C$'를 정의한다. 이후 픽셀별로 전경 가능도를 비교하여 더 높은 값을 가진 객체에 픽셀을 할당한다.

$$z_{i,1} = \begin{cases} 1, & \text{if } p(f_i=F, \hat{y}_1) > \max\{p(f_i=F, \hat{y}_2), p(f_i=O)\} \\ 0, & \text{otherwise} \end{cases}$$

**Step 3: Order recovery**
픽셀 단위의 할당 결과를 바탕으로 객체 간의 폐색 순서 $R(I_1, I_2)$를 추론한다. 충돌 영역 $C$에서 더 많은 픽셀을 확보한 객체가 전경(앞쪽)에 있다고 판단한다.

$$R(I_1, I_2) = \begin{cases} 1, & \sum_{i \in C} z_{i,1} > \sum_{i \in C} z_{i,2} \\ -1, & \text{otherwise} \end{cases}$$

이 순서 정보가 결정되면, "All or Nothing" 방식으로 전경 객체에게 해당 영역의 픽셀을 우선 할당하여 세그멘테이션 마스크를 수정한다.

**Step 4: Self-correction (Top-down refinement)**
수정된 가시성 변수 $z_{i,n}$을 다시 모델에 입력하여 객체의 가능도를 재계산한다. 이를 통해 잘못된 세그멘테이션으로 인해 틀렸던 클래스 분류 결과까지 교정하는 반복적(Iterative)인 상향식-하향식 구조를 갖는다.

## 📊 Results

### 실험 설정
- **데이터셋**: KITTI INStance (KINS) 데이터셋 및 저자들이 직접 구축한 **Synthetic Occlusion Challenge** 데이터셋(2객체, 4객체, 미지의 폐색체 포함 시나리오)을 사용하였다.
- **비교 대상**: Fully-supervised 방식인 Mask R-CNN, self-supervised 방식인 PCNet-M, weakly-supervised 방식인 BBTP 및 기본 CompNet을 비교군으로 설정하였다.
- **평가 지표**: mIoU를 사용하였으며, 폐색 수준을 $L0(0\text{-}1\%)$, $L1(1\text{-}30\%)$, $L2(30\text{-}60\%)$, $L3(60\text{-}90\%)$로 나누어 분석하였다.

### 주요 결과
1. **Modal Segmentation (가시 부분 세그멘테이션)**:
   - KINS 데이터셋에서 제안 방법은 약한 지도 학습 모델들 중 가장 높은 성능을 보였으며, 특히 폐색 수준이 높은 $L2, L3$ 영역에서 기본 CompNet 대비 mIoU가 각각 $9.6\%, 11.3\%$ 향상되었다.
   - Fully-supervised 모델인 Mask R-CNN과의 성능 격차를 유의미하게 줄였다.

2. **Amodal Segmentation (전체 형태 세그멘테이션)**:
   - 제안 모델은 모든 폐색 수준에서 다른 약한 지도 학습 방법들을 압도하였다.
   - 놀랍게도 모달 마스크를 직접 지도 학습으로 사용한 PCNet-M보다 전체 mIoU 기준 6.5% 더 높은 성능을 기록하였다.

3. **Order Recovery의 효과**:
   - Ablation study를 통해 픽셀 단위 경쟁(NOD)만 수행했을 때보다 폐색 순서 복원(OD)을 함께 수행했을 때 모달/아모달 세그멘테이션 성능이 모두 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점
본 논문은 딥러닝의 특징 추출 능력과 전통적인 생성 모델의 구조적 추론 능력을 효과적으로 결합하였다. 특히, 데이터 증강(Data Augmentation)만으로는 해결하기 어려운 '폐색 관계의 논리적 추론'을 ORM이라는 모듈을 통해 구현함으로써, 바운딩 박스라는 최소한의 정보만으로도 정교한 인스턴스 세그멘테이션이 가능함을 입증하였다.

### 한계 및 가정
- **바운딩 박스 의존성**: 실험 과정에서 정확한 평가를 위해 Ground Truth 바운딩 박스를 제공하였다. 실제 환경에서 바운딩 박스 검출기(Detector)의 오차가 발생할 경우, ORM의 추론 결과에 영향을 줄 가능성이 크다.
- **계산 복잡도**: 다중 객체에 대해 순차적으로 네트워크를 처리하고 ORM을 반복 적용하는 과정에서 연산 시간이 증가할 수 있다.

### 비판적 해석
본 연구는 '알려지지 않은 폐색체'에 대해서도 강건함을 주장하는데, 이는 Outlier Model을 통해 정체불명의 특징들을 처리하기 때문이다. 하지만 이는 Outlier Model이 충분히 일반화된 특징들을 학습했다는 가정 하에 가능하며, 매우 특이한 형태의 폐색체가 나타날 경우 여전히 취약할 수 있다. 그럼에도 불구하고, 픽셀 단위의 확률적 경쟁과 객체 단위의 순서 추론을 결합한 접근 방식은 매우 타당하며 효율적인 해결책으로 판단된다.

## 📌 TL;DR

본 논문은 다중 객체가 서로 가려진 상황에서 바운딩 박스만으로 정밀한 인스턴스 세그멘테이션을 수행하는 모델을 제안한다. **Compositional Networks**를 확장하여 다중 객체를 처리하고, **Occlusion Reasoning Module (ORM)**을 통해 픽셀 간 경쟁과 폐색 순서를 추론함으로써 세그멘테이션 오류를 스스로 교정한다. 실험 결과, 특히 폐색이 심한 상황에서 기존 약한 지도 학습 방법론들을 크게 상회하는 성능을 보였으며, 아모달 세그멘테이션에서 탁월한 성과를 거두었다. 이 연구는 복잡한 환경의 자율주행 및 로봇 비전 시스템에서 객체 인식의 강건성을 높이는 데 기여할 가능성이 크다.