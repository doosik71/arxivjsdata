# Bottom-up Instance Segmentation using Deep Higher-Order CRFs

Anurag Arnab and Philip H.S. Torr (2016)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Instance Segmentation**이다. Instance Segmentation은 이미지 내의 객체를 픽셀 단위로 인식하고 국지화하는 작업으로, Object Detection과 Semantic Segmentation의 교차점에 위치한다.

- **Object Detection의 한계**: 객체를 Bounding Box 수준에서 국지화할 수는 있지만, 픽셀 단위의 정밀한 마스크를 생성하지 못한다.
- **Semantic Segmentation의 한계**: 각 픽셀의 클래스 레이블은 결정할 수 있으나, 동일한 클래스에 속하는 서로 다른 객체 인스턴스(Instance)를 구분하는 개념이 없다.

따라서 본 연구의 목표는 객체 검출(Object Detection)의 인스턴스 구분 능력과 시맨틱 분할(Semantic Segmentation)의 픽셀 수준 정밀도를 결합하여, 개별 객체를 픽셀 단위로 분리해내는 Bottom-up 방식의 인스턴스 분할 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Deep Higher-Order Conditional Random Fields (CRF)**를 이용하여 시맨틱 분할 결과와 객체 검출 결과를 통합하는 것이다.

중심적인 설계 직관은 다음과 같다:
1. **Bottom-up 접근 방식**: 먼저 이미지 전체에 대해 카테고리 수준의 시맨틱 분할을 수행한 후, 이를 인스턴스 수준으로 정제한다.
2. **Higher-Order Detection Potentials**: 객체 검출기(Object Detector)의 출력을 CRF의 고차 포텐셜(Higher-order potential)로 삽입하여, 시맨틱 분할의 정확도를 높이는 동시에 검출기의 신뢰도 점수를 재보정(Recalibrate)한다.
3. **Differentiable Pipeline**: 전체 시스템(CNN $\rightarrow$ CRF $\rightarrow$ Instance Identification $\rightarrow$ Instance CRF)을 완전히 미분 가능한 형태로 설계하여 end-to-end 학습이 가능하도록 구현하였다.

## 📎 Related Works

논문에서는 기존의 인스턴스 분할 접근 방식을 크게 두 가지로 분류하여 설명한다.

- **SDS (Simultaneous Detection and Segmentation)**: Hariharan et al. [13] 등이 제안한 방식으로, 객체 제안(Object Proposal)을 먼저 생성하고 각 제안 내에서 마스크를 생성하는 방식이다. 하지만 이 방식은 초기 Proposal의 품질에 의존하며, Proposal 생성 단계에서 많은 시간이 소요된다는 한계가 있다.
- **Proposal-free 및 기타 방식**: PFN [24]과 같이 시맨틱 분할을 먼저 수행하고 인스턴스 바운딩 박스를 예측하는 방식이 존재한다. 하지만 이러한 방식은 인스턴스 개수 예측이나 복잡한 클러스터링 단계가 필요하다.

**차별점**: 본 논문의 방법은 객체 검출기의 결과를 CRF의 포텐셜로 직접 활용하여 시맨틱 분할 단계에서 이미 인스턴스에 대한 힌트를 얻으며, 별도의 복잡한 인스턴스 개수 예측 단계 없이 재보정된 검출 점수를 통해 인스턴스를 식별한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

전체 시스템은 크게 두 단계의 서브 네트워크로 구성된다: **End-to-end Object Segmentation Subnetwork**와 **End-to-end Instance Segmentation Subnetwork**이다.

### 1. Semantic Segmentation with Higher Order CRF
먼저 Pixelwise CNN을 통해 초기 유니러리(Unary) 예측을 수행하고, 이를 Higher-Order CRF를 통해 정제한다. 이때 CRF의 에너지 함수 $E(x)$는 다음과 같이 정의된다:

$$E(x) = \sum_{i} \psi_{U}^{i}(x_{i}) + \sum_{i<j} \psi_{P}^{i,j}(x_{i}, x_{j}) + \sum_{d} \psi_{Det}^{d}(x_{d}, y_{d}) + \sum_{d} \psi_{U}^{d}(y_{d})$$

여기서 각 항의 역할은 다음과 같다:
- $\psi_{U}^{i}$: CNN에서 예측된 픽셀 $i$의 클래스 확률(Unary potential).
- $\psi_{P}^{i,j}$: 픽셀 간의 외관 및 공간적 일관성을 강제하는 Pairwise potential.
- $\psi_{Det}^{d}$: 객체 검출 결과 기반의 고차 포텐셜. 검출된 객체 $d$의 전경(Foreground) 픽셀들이 해당 클래스 레이블 $l_d$를 갖도록 유도한다.
- $\psi_{U}^{d}$: 검출기의 초기 신뢰도 점수(Confidence score)에 기반한 유니러리 포텐셜.

특히, 각 검출 결과에 대해 잠재 이진 변수 $Y_d$를 도입한다. $Y_d=1$이면 해당 검출이 유효함을, $Y_d=0$이면 무효함을 의미한다. $\psi_{Det}^{d}$는 다음과 같이 정의된다:

$$\psi_{Det}^{d}(X_{d}=x_{d}, Y_{d}=y_{d}) = 
\begin{cases} 
w_{l} s_{d} |F_{d}| \sum_{i=1}^{|F_{d}|} [x_{d}^{(i)} = l_{d}] & \text{if } y_{d}=0 \\
w_{l} s_{d} |F_{d}| \sum_{i=1}^{|F_{d}|} [x_{d}^{(i)} \neq l_{d}] & \text{if } y_{d}=1 
\end{cases}$$

이 과정을 통해 시맨틱 분할 결과가 개선될 뿐만 아니라, CRF 추론 후의 $Y_d$ 확률값을 통해 **재보정된 검출 점수(Recalibrated detection score)**를 얻을 수 있다.

### 2. Instance Identification and Refinement
시맨틱 분할 결과와 재보정된 검출 점수를 이용하여 픽셀을 특정 인스턴스 $k$에 할당한다. 픽셀 $i$가 인스턴스 $k$에 속할 확률 $\Pr(v_i = k)$는 다음과 같다:

$$\Pr(v_{i}=k) = 
\begin{cases} 
\frac{1}{Z(Y,Q)} Q_{i}(l_{k}) \Pr(Y_{k}=1) & \text{if } i \in B_{k} \\
0 & \text{otherwise} 
\end{cases}$$

여기서 $B_k$는 $k$번째 검출기의 Bounding Box이며, $Q_i(l_k)$는 이전 단계에서 얻은 시맨틱 분할 확률이다. 이렇게 얻은 확률값은 다시 한번 **Instance CRF**의 유니러리 포텐셜로 사용되어, 최종적인 인스턴스 분할 맵을 생성한다. 이 Instance CRF는 이미지마다 검출된 객체 수 $D$에 따라 레이블 수가 변하는 **Dynamic CRF** 구조를 가진다.

## 📊 Results

### 실험 설정
- **데이터셋**: PASCAL VOC 2012 검증 세트 (1,449장).
- **평가 지표**: $\text{AP}_r$ (Intersection over Union, IoU 임계값에 따른 평균 정밀도) 및 $\text{AP}_{vol}$ (IoU 0.1부터 0.9까지의 평균 $\text{AP}_r$).
- **구현**: VGG-16 기반의 Fully Convolutional Network (FCN)를 백본으로 사용하고 Faster R-CNN 검출기를 활용하였다.

### 주요 결과
1. **Detection Potentials의 효과**: 
   - Detection potentials를 제거했을 때보다 $\text{AP}_r$ (IoU=0.5) 기준 약 3.7% 성능이 하락하였다.
   - 단순히 시맨틱 분할 성능만 높이는 것이 아니라, $Y$ 변수를 통한 점수 재보정이 인스턴스 식별 단계에서 중요한 역할을 함을 확인하였다.

2. **타 방법론과의 비교**:
   - SDS [13] 및 Chen et al. [4]와 같은 Detection-and-Refine 방식보다 월등히 높은 성능을 보였다.
   - PFN [24]과 비교했을 때, 전반적인 $\text{AP}_{vol}$은 비슷하지만, **높은 IoU 임계값(예: 0.9)에서 훨씬 높은 정밀도**를 기록하였다 ($\text{AP}_r$ at 0.9: PFN 15.7% vs Ours 20.1%). 이는 본 제안 방법이 생성하는 마스크의 경계가 훨씬 더 정확함을 시사한다.

## 🧠 Insights & Discussion

**강점**: 
본 연구는 시맨틱 분할과 객체 검출이라는 두 가지 독립적인 태스크의 강점을 CRF라는 구조 속에 효과적으로 통합하였다. 특히 Bottom-up 방식을 채택함으로써 초기 Proposal의 품질에 종속되는 문제를 해결하였고, 높은 IoU 임계값에서 우수한 성능을 보임으로써 픽셀 수준의 정밀도를 확보하였다.

**한계 및 비판적 해석**:
실험 결과에서 나타나듯, 시각적으로 매우 유사한 객체들이 서로 겹쳐 있거나(Occlusion) 강하게 중첩된 경우, 인스턴스를 개별적으로 분리하는 데 어려움을 겪는 모습이 관찰되었다. 이는 현재의 방식이 객체 간의 기하학적 관계나 깊이 정보보다는 픽셀의 외관과 바운딩 박스 정보에 주로 의존하기 때문으로 해석된다.

**결론**: 
본 논문은 Dynamic CRF를 신경망의 레이어로 삽입하여 end-to-end로 학습 가능하게 함으로써, 인스턴스 분할 문제를 효율적으로 해결할 수 있는 가능성을 제시하였다.

## 📌 TL;DR

본 논문은 시맨틱 분할 결과와 객체 검출 결과를 **Deep Higher-Order CRF**로 통합하여 정밀한 인스턴스 분할을 수행하는 Bottom-up 프레임워크를 제안한다. 객체 검출기의 출력을 CRF 포텐셜로 사용하여 시맨틱 분할을 정제하고 검출 점수를 재보정하며, 이를 통해 기존 방법론 대비 특히 **높은 IoU 임계값에서 매우 정밀한 객체 마스크를 생성**한다. 이 연구는 서로 다른 컴퓨터 비전 태스크를 미분 가능한 CRF 구조로 통합하여 시너지를 낼 수 있음을 보여주었다.