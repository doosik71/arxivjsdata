# Salient Instance Segmentation via Subitizing and Clustering

Jialun Pei, He Tang, Chao Liu, and Chuanbo Chen (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Salient Instance Segmentation (SIS)**이다. 기존의 Salient Region Detection(돌출 영역 검출)은 이미지에서 가장 시선을 끄는 영역을 찾는 것에 집중했지만, SIS는 해당 영역 내에서 개별 객체 인스턴스(individual instances)를 구분하여 분할하는 더 어려운 과업을 수행한다.

이 문제의 중요성은 타겟 인식, 운전자 보조 시스템, 이미지 캡셔닝 등 더 구체적이고 심도 있는 응용 분야에서 개별 객체 단위의 정밀한 분석이 필요하기 때문이다. 기존의 접근 방식들은 주로 Region Proposal Network(RPN)와 같은 객체 제안(object proposals) 방식에 의존했다. 그러나 이러한 방식은 수백 개의 후보 박스를 생성하고 필터링하는 과정이 비효율적이며, 강하게 가려진(occluded) 객체의 경계를 정확히 찾아내지 못하는 한계가 있다. 따라서 본 논문의 목표는 객체 제안 과정이 필요 없는 **Proposal-free**이며, 특정 클래스 정의가 필요 없는 **Category-independent**한 방식의 Salient Instance Segmentation 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 객체 제안 방식 대신 **Subitizing(순식간에 개수를 세는 능력)**과 **Spectral Clustering(스펙트럼 클러스터링)**을 결합하여 인스턴스를 분할하는 것이다.

주요 기여 사항은 다음과 같다:
1. **Multitask Densely Connected Neural Network (MDNN)** 제안: 인스턴스의 개수를 예측하는 DSN과 돌출 영역을 검출하는 DFCN이라는 두 개의 병렬 브랜치를 통해 제안 과정 없이 효율적으로 정보를 추출한다.
2. **Subitizing 기반의 인스턴스 수 예측**: DSN을 통해 이미지 내 돌출 객체의 수를 직접 예측함으로써, 기존의 복잡한 Proposal 최적화 과정을 대체한다.
3. **적응형 스펙트럼 클러스터링(Adaptive Spectral Clustering)**: DFCN에서 추출한 다중 스케일 딥 특징(deep features)을 활용하여 인스턴스를 분할한다. 특히, 무작위 선택 대신 **적응형 분위수 전략(Adaptive Quantile Strategy)**을 사용하여 k-means 클러스터링의 초기 중심점을 설정함으로써 지역 최적해(local minima) 문제를 해결하였다.

## 📎 Related Works

논문에서는 세 가지 관련 연구 분야를 다룬다.

1. **Salient Object Detection**: 전통적인 머신러닝 방식에서 최근의 CNN 기반 모델로 발전하며 정밀도가 향상되었다. 그러나 대부분의 연구가 개별 인스턴스의 구분보다는 전체적인 돌출 영역 검출에만 집중했다는 한계가 있다.
2. **Instance-Aware Semantic Segmentation**: Mask R-CNN과 같이 객체 검출과 세그멘테이션을 동시에 수행하는 방식이다. 하지만 이러한 프레임워크는 미리 정의된 카테고리(Category-dependent)에 의존하므로, 카테고리 구분 없이 돌출된 객체만을 분리해야 하는 SIS 과업에는 적합하지 않다.
3. **Salient Instance Segmentation**: MSRNet과 S4Net 등의 초기 연구가 존재한다. 이들은 MSRNet의 경우 다중 스케일 정제 네트워크와 CRF(Conditional Random Field)에 크게 의존하며, S4Net은 단일 단계 검출기를 사용하여 대략적인 위치를 찾는다. 두 방법 모두 Region Proposal에 의존하기 때문에 연산 효율성이 떨어지고 포스트 프로세싱 과정이 복잡하다는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조
제안된 MDNN 프레임워크는 크게 세 단계의 파이프라인으로 구성된다: **(1) DSN을 통한 인스턴스 수 예측 $\rightarrow$ (2) DFCN을 통한 돌출 영역 검출 $\rightarrow$ (3) 적응형 스펙트럼 클러스터링을 통한 최종 인스턴스 분할**.

### 1. Densely Connected Subitizing Network (DSN)
DSN은 이미지 내 돌출 객체의 수를 예측하는 분류 모델이다. DenseNet을 기반으로 하며, 채널 간의 상호 의존성을 모델링하여 표현력을 높이기 위해 **SE (Squeeze-and-Excitation) block**을 내장하였다.
- **SE Block 동작**: Global Average Pooling으로 채널 벡터 $v$를 얻고, 두 개의 FC 레이어와 ReLU, Sigmoid 함수를 거쳐 채널별 가중치를 재조정한다.
$$X' = (\phi(fc_2(\eta(fc_1(v, W_1)), W_2))) \cdot X$$
여기서 $\eta$는 ReLU, $\phi$는 Sigmoid 함수이다.
- **구조**: 4개의 Dense Block(각 6, 12, 48, 32 레이어)으로 구성되며, 최종 출력층은 4차원 FC 레이어로 1, 2, 3, 4+ 개의 인스턴스를 예측한다.

### 2. Densely Connected Fully Convolutional Network (DFCN)
DFCN은 픽셀 단위의 정밀한 돌출 영역 맵(saliency map)을 생성한다. 
- **구조**: U-shape 구조를 가지며, Downsampling 경로와 Upsampling 경로 사이에 **Skip Connection**을 구축하여 해상도 손실을 줄이고 특징을 보존한다.
- **CRF Refinement**: FCN 결과물의 거친 경계를 정밀하게 다듬기 위해 Fully Connected CRF를 적용한다. 에너지 함수 $E(S)$는 다음과 같이 정의된다.
$$E(S) = -\sum_i \log P(s_i) + \sum_{i,j} \phi^p(s_i, s_j)$$
여기서 $\phi^p$는 픽셀 간의 외관(appearance)과 매끄러움(smoothness)을 고려한 페어와이즈 비용(pairwise cost)이다.

### 3. Adaptive Spectral Clustering
예측된 인스턴스 수 $k$를 클러스터 개수로 사용하여 돌출 영역을 개별 인스턴스로 분할한다.
- **특징 추출**: SLIC 알고리즘을 통해 이미지를 슈퍼픽셀(superpixels)로 나누고, DFCN의 후반부 레이어에서 추출한 딥 특징을 활용한다.
- **친밀도 행렬(Affinity Matrix)**: 두 노드 $i, j$ 사이의 친밀도 $\omega_{ij}$는 다음과 같이 계산된다.
$$\omega_{ij} = \frac{e^{-\|c_i - c_j\|^2 / \sigma^2}}{1 + \lambda \cdot \|d_i - d_j\|}$$
여기서 $c$는 딥 특징의 평균, $d$는 유클리드 거리이다.
- **개선된 k-means (Fractile-based)**: 무작위 초기화 대신, 고유벡터 $U$의 값을 오름차순으로 정렬한 후 $k$개의 등분점으로 나눈 분위수(fractile) 지점을 초기 중심점으로 설정한다.
$$Q_i = \left[ \frac{50}{k} + (i-1)\frac{100}{k} \right] \cdot U, \quad i=1, 2, \dots, k$$

### 학습 절차 및 손실 함수
두 네트워크는 서로 다른 학습 세트를 사용하므로 파라미터를 공유하지 않는다. 손실 함수로는 Weighted Cross-Entropy를 사용한다.
$$L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_i^c \log \hat{y}_i^c$$
SGD(Stochastic Gradient Descent)를 사용하여 학습하며, DSN과 DFCN의 학습률 및 배치 사이즈는 각각 다르게 설정하여 최적화하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: dataset1K (인스턴스 분할), SOS, MSRA-B, DUT-OMRON, ECSSD, SOD, SED2 (영역 검출).
- **지표**:
    - 영역 검출: $\text{maxF}$, $\text{MAE}$, $\text{S-measure}$, $\text{E-measure}$.
    - 인스턴스 분할: $\text{AP}_r$ (Average Precision for instances).
- **비교 대상**: MSRNet, S4Net 및 기타 최신 돌출 영역 검출 모델들.

### 주요 결과
1. **Subitizing 성능**: DSN은 기존 CNN-Syn-FT 모델보다 평균 AP가 5%p 높았으며, 특히 3개의 객체를 예측하는 정확도가 약 11% 향상되었다.
2. **돌출 영역 검출 성능**: DFCN은 MAE 지표에서 최첨단 모델들보다 우수한 성능을 보였으며, CRF 적용 후 모든 지표에서 정밀도가 향상되었다.
3. **인스턴스 분할 성능**: dataset1K 테스트 세트에서 $\text{AP}_r@0.7$ 기준 **60.14%**를 달성하여 S4Net(55.34%)과 MSRNet보다 약 5% 정도 높은 성능을 기록하였다.
4. **효율성**: 이미지 한 장당 처리 시간이 약 **1.3초**로, MSRNet(20초 이상)에 비해 획기적으로 빨라졌다.

## 🧠 Insights & Discussion

### 강점
본 연구는 Proposal-free 방식을 통해 연산 효율성을 극대화하면서도, Subitizing이라는 직관적인 접근법으로 인스턴스의 개수를 정확히 파악하여 분할 성능을 높였다. 특히 클래스 정보 없이도 돌출된 객체를 분리해낼 수 있다는 점이 강력한 범용성을 제공한다. 또한, 딥 특징 기반의 스펙트럼 클러스터링과 분위수 기반 초기화 전략을 통해 클러스터링의 안정성을 확보하였다.

### 한계 및 미해결 질문
1. **데이터셋 부족**: SIS 분야의 가용 데이터셋이 매우 제한적이어서 모델의 일반화 능력을 완전히 검증하기 어렵다.
2. **예측 오류의 전이**: DSN이 인스턴스 개수 $k$를 잘못 예측할 경우, 후속 단계인 클러스터링 결과가 필연적으로 실패하게 된다.
3. **소형 객체 검출**: 매우 작은 인스턴스의 경우 스펙트럼 클러스터링 과정에서 무시되는 경향이 있어 분할에 실패하는 사례가 관찰되었다.

### 비판적 해석
본 논문은 제안 기반 방식의 비효율성을 성공적으로 해결했으나, 시스템의 전체 성능이 'Subitizing'이라는 단일 단계의 정확도에 과도하게 의존하는 구조이다. 만약 Subitizing 단계에서 오류가 발생했을 때 이를 보정할 수 있는 피드백 루프나 정제 메커니즘이 추가된다면 더욱 견고한 시스템이 될 수 있을 것이다.

## 📌 TL;DR

본 논문은 객체 제안(Proposal) 과정 없이 돌출 인스턴스를 분할하는 **MDNN** 프레임워크를 제안한다. **DSN**으로 객체 수를 빠르게 세고, **DFCN**으로 돌출 영역을 찾은 뒤, **적응형 스펙트럼 클러스터링**을 통해 개별 인스턴스를 분리한다. 이 방식은 기존 Proposal 기반 방식보다 훨씬 빠르고($\sim 1.3\text{s/img}$), $\text{AP}_r@0.7$ 기준 60.14%의 높은 정밀도를 달성하여 SIS 분야의 새로운 기준을 제시하였다. 향후 데이터셋 확충과 소형 객체 처리 능력을 개선한다면 실시간 타겟 인식 및 분석 시스템에 핵심적인 역할을 할 가능성이 높다.