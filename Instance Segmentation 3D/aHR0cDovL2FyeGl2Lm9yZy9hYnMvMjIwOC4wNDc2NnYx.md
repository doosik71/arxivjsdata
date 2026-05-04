# Semantic Segmentation-Assisted Instance Feature Fusion for Multi-Level 3D Part Instance Segmentation

Chun-Yu Sun, Xin Tong, Yang Liu (2022)

## 🧩 Problem to Solve

본 논문은 3D 포인트 클라우드(point cloud)에서 **3D Part Instance Segmentation**을 수행하는 것을 목표로 한다. 3D 인스턴스 분할은 객체 수준을 넘어 더 세밀한 부품(part) 수준에서 인스턴스를 구분하고 동시에 의미론적 라벨(semantic label)을 추출해야 하는 어려운 과제이다.

특히, 의자 바퀴나 책상 다리와 같은 부품 수준의 인스턴스는 구조와 기하학적 형태의 변동성이 매우 크고, 학습을 위한 주석 데이터(annotated data)가 부족하여 기존 방법론으로는 정확한 분할이 어렵다. 기존의 딥러닝 기반 접근 방식들은 주로 Semantic Segmentation과 Instance Center Prediction을 개별적인 학습 작업으로 처리하거나, 두 특징 간의 관계를 단순한 포인트 단위(pointwise)로만 융합하여, 형태의 의미론적 정보와 인스턴스 간의 내재적 관계를 충분히 활용하지 못하는 한계가 있다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Semantic Segmentation의 결과를 활용하여 비국소적(non-local) 방식으로 인스턴스 특징을 융합**하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Semantic Segmentation-Assisted Instance Feature Fusion**: Semantic Segmentation의 확률 벡터를 가이드로 삼아 인스턴스 특징을 집계(aggregate)함으로써, 인스턴스 중심 예측(instance center prediction)의 강건성을 높이는 경량화된 융합 전략을 제안한다.
2. **Multi- and Cross-level Fusion**: 인간이 만든 3D 형상의 계층적 구조를 활용하여, 여러 수준(coarse, middle, fine)의 의미론적 정보를 교차적으로 융합함으로써 세밀한 부품 인스턴스 분할 성능을 향상시킨다.
3. **Semantic Region Center Prediction**: 매우 인접해 있어 구분이 어려운 인스턴스들을 분리하기 위해 '의미론적 영역 중심(Semantic Region Center)'을 예측하고, 이를 통해 인스턴스 중심들을 서로 밀어내어 클러스터링 성능을 개선한다.

## 📎 Related Works

### 기존 연구 및 한계

3D 인스턴스 분할은 크게 제안 기반(Proposal-based), 탐지 기반(Detection-based), 그리고 클러스터링 기반(Clustering-based) 방법으로 나뉜다. 최근의 클러스터링 기반 방법들은 포인트별 의미론적 라벨과 인스턴스 특징(중심점 오프셋 등)을 동시에 예측하는 Encoder-Decoder 구조를 주로 사용한다.

### 차별점

ASIS나 JSNet과 같은 기존의 특징 융합 연구들은 의미론적 특징과 인스턴스 특징을 포인트 단위로 융합하거나 중간 네트워크 단계에서 융합하였다. 반면, 본 논문은 **최종 출력 단계의 Semantic Probability를 활용하여 비국소적인 방식으로 특징을 융합**함으로써 더 강력한 가이드를 제공한다. 또한, 단순한 객체 분할을 넘어 부품의 계층적 구조를 고려한 **Cross-level fusion**과 **Semantic Region Center** 개념을 도입하여 차별성을 갖는다.

## 🛠️ Methodology

### 1. Baseline Network

기본 구조는 **O-CNN 기반의 U-Net**을 사용하며, 하나의 Shared Encoder와 두 개의 병렬 디코더($D_{sem}$, $D_{ins}$)로 구성된다.

- **Semantic Decoder ($D_{sem}$)**: 포인트별 의미론적 확률 $P_{sem} \in \mathbb{R}^{N \times c}$를 예측하며, Cross-Entropy 손실 함수를 사용한다.
- **Instance Decoder ($D_{ins}$)**: 포인트에서 인스턴스 중심까지의 오프셋 $O^I \in \mathbb{R}^{N \times 3}$를 예측하며, $L_2$ 손실 함수를 사용한다.
- **Clustering**: 추론 단계에서는 예측된 오프셋으로 포인트를 이동시킨 후, Mean-shift 알고리즘을 통해 동일 의미론을 가진 포인트들을 인스턴스로 그룹화한다.

### 2. Semantic Segmentation-Assisted Instance Feature Fusion

의미론적 정보를 이용해 인스턴스 특징을 비국소적으로 융합하는 과정은 두 단계로 이루어진다.

**단계 1: 부품 인스턴스 특징($Z_m$) 계산**
각 의미론적 클래스 $m$에 대해, 해당 클래스에 속할 확률이 높은 포인트들의 인스턴스 특징 $F_{ins}$를 가중 평균하여 집계한다.
$$Z_m := \frac{\sum_{p \in S} P_{sem}(p)|_m \cdot F_{ins}(p)}{\sum_{p \in S} P_{sem}(p)|_m}$$

**단계 2: 집계된 인스턴스 특징($\hat{F}$) 계산**
각 포인트 $p$에 대해, 모든 클래스의 $Z_m$을 해당 포인트의 의미론적 확률로 다시 가중 합산한다.
$$\hat{F}(p) = \sum_{m=1}^{c} P_{sem}(p)|_m \cdot Z_m$$

최종적으로 융합된 특징 $F_{fusion}(p) = [\hat{F}(p), F_{ins}(p), p]$를 사용하여 인스턴스 중심 오프셋을 더 정확하게 예측한다.

### 3. Multi-level & Cross-level Fusion

부품의 계층 구조(Coarse $\rightarrow$ Middle $\rightarrow$ Fine)를 반영하기 위해, $k$번째 레벨의 인스턴스 특징을 융합할 때 $r$번째 레벨의 의미론적 확률을 사용하여 특징을 집계한다.
$$Z_{m}^{(k,r)} := \frac{\sum_{q \in S} P_{sem}^{(r)}(q)|_m \cdot F_{ins}^{(k)}(q)}{\sum_{q \in S} P_{sem}^{(r)}(q)|_m}$$
이를 통해 하위 레벨의 인스턴스가 상위 레벨의 의미론적 맥락을 상속받을 수 있게 한다.

### 4. Semantic Region Center

가위 날과 같이 매우 인접한 인스턴스들을 구분하기 위해, 동일 의미론을 가진 인스턴스 중심들의 중심점인 **Semantic Region Center** ($O^S$)를 예측한다. 클러스터링 전 포인트를 다음과 같이 이동시켜 인스턴스 간 거리를 벌린다.
$$\hat{p} := p + O^I(p) + \lambda \cdot \frac{O^I(p) - O^S(p)}{\|O^I(p) - O^S(p)\|}$$
여기서 $\lambda$는 밀어내는 강도를 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: PartNet (부품 수준), ScanNet 및 S3DIS (실내 장면 수준)
- **지표**: mAP (IoU 임계값 0.25, 0.5, 0.75), s-AP50, mIoU
- **비교 대상**: SGPN, PE, DyCo3D 등

### 주요 결과

1. **PartNet 성능**: 제안 방법은 24개 카테고리에 대해 기존 최우수 모델인 PE 대비 $\text{AP}_{50}$ 기준 평균 **+6.6%의 큰 폭으로 성능을 향상**시켰다.
2. **범용성 검증**: 제안한 Feature Fusion 모듈을 PointGroup, DyCo3D, HAIS 등 기존의 다른 인스턴스 분할 프레임워크에 플러그인 형태로 추가했을 때, ScanNet과 S3DIS 데이터셋 모두에서 일관된 성능 향상이 관찰되었다.
3. **효율성**: 추가적인 연산 시간이 매우 적어(수 msec 수준), 경량화된 모듈임을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **비국소적 융합의 효과**: 포인트 단위의 융합보다 의미론적 확률을 이용한 비국소적 융합이 인스턴스 중심을 훨씬 더 조밀하게 예측하게 함을 시각적으로 확인하였다.
- **계층 구조 활용**: Cross-level fusion을 통해 세밀한(Fine) 레벨의 부품 분할 시 발생하는 데이터 부족 문제를 상위 레벨의 정보로 보완할 수 있었다.
- **Semantic Region Center의 필요성**: 대칭적으로 배치된 부품들의 경우 중심점이 매우 가깝게 예측되어 오분류가 잦은데, 영역 중심을 이용해 이들을 밀어냄으로써 클러스터링 정확도를 획기적으로 높였다.

### 한계 및 논의

- **하이퍼파라미터 의존성**: Mean-shift의 대역폭(bandwidth)과 이동 파라미터 $\lambda$를 경험적으로 설정하였다는 점이 한계로 지적되었다.
- **미해결 과제**: 저자들은 향후 대역폭과 $\lambda$를 학습 가능하게 만드는 미분 가능한 클러스터링 알고리즘을 도입하는 것이 성능을 더 높일 수 있는 방향이라고 제시한다.
- **Gradient Stopping**: 융합 모듈에서 의미론적 디코더로 흐르는 그래디언트를 차단하는 것이 의미론적 분할 정확도를 유지하고 최종 성능을 높이는 데 도움이 된다는 점을 발견하였다.

## 📌 TL;DR

본 논문은 3D 부품 인스턴스 분할을 위해 **의미론적 확률 기반의 비국소적 특징 융합(Non-local Feature Fusion)**과 **의미론적 영역 중심(Semantic Region Center)** 예측 기법을 제안하였다. 특히 부품의 계층 구조를 활용한 **Cross-level fusion**을 통해 세밀한 부품 분할 성능을 크게 높였으며, 이 모듈은 경량 구조로서 다른 기존 3D 인스턴스 분할 모델에도 쉽게 적용 가능하여 범용적인 성능 향상을 가져온다.
