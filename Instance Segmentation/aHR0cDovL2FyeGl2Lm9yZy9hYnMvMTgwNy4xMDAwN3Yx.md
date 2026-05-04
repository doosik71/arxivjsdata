# Instance Segmentation by Deep Coloring

Victor Kulikov, Victor Yurchenko, and Victor Lempitsky (2018)

## 🧩 Problem to Solve

본 논문은 이미지 내에서 동일한 클래스에 속하는 개별 객체들을 구분해내는 Instance Segmentation 문제를 해결하고자 한다. 일반적으로 Instance Segmentation은 Semantic Segmentation보다 훨씬 어려운 과제로 간주되는데, 그 이유는 이미지 내 객체의 수가 사전에 정해져 있지 않고, 서로 다른 객체들이 시각적으로 매우 유사한 외형을 가질 수 있기 때문이다.

기존의 Feed-forward 방식 Instance Segmentation 아키텍처들은 대개 매우 복잡한 구조를 가지고 있어 학습 및 추론 시간이 오래 걸리며, 파라미터 튜닝과 구현이 어렵다는 단점이 있다. 이러한 복잡성의 주된 원인은 Convolutional 아키텍처가 자연스럽게 활용할 수 있는 객체 인스턴스 간의 일관된 순서(Ordering)가 존재하지 않기 때문이다. 따라서 본 논문의 목표는 Instance Segmentation 문제를 단순한 Semantic Segmentation 문제로 환원(Reduction)시켜, 기존의 효율적인 Semantic Segmentation 아키텍처를 그대로 활용하면서도 End-to-End 학습이 가능한 단순한 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 고정된 수의 라벨(이하 '색상', Colors)을 도입하고, 학습 과정에서 각 객체 인스턴스를 이러한 색상들에 동적으로 할당하는 **Deep Coloring** 방식이다.

핵심 직관은 다음과 같다. 제한된 수의 색상만으로도, 공간적으로 서로 떨어져 있는(Spatially separated) 객체들이라면 동일한 색상을 공유해서 사용해도 무방하다는 점이다. 즉, 네트워크가 이미지 내의 다수 인스턴스들을 고정된 색상 세트를 이용해 '색칠'하도록 학습시킨 뒤, 추론 단계에서 간단한 Connected Component Analysis(CCA)를 통해 개별 인스턴스를 분리해내는 방식이다. 이 접근법은 인스턴스의 수가 가변적이더라도 고정된 출력 채널 수를 가진 네트워크로 처리할 수 있게 한다.

## 📎 Related Works

논문에서는 Instance Segmentation의 기존 접근 방식을 크게 네 가지로 분류하여 설명한다.

1. **Proposal-based methods**: Mask R-CNN과 같이 객체 제안(Proposal)을 생성하고 이를 정교화하는 방식이다. 성능은 뛰어나나, Object Detection 루틴의 품질에 의존하며 경계 상자(Bounding box)로 근사하기 어려운 객체에는 취약하다.
2. **Recurrent methods**: RNN/LSTM을 사용하여 인스턴스를 하나씩 순차적으로 생성하는 방식이다. 추론 시간이 객체 수에 비례하여 증가하며 구조가 복잡하다.
3. **Proposal-free methods**: Semantic Segmentation 결과를 기반으로 인스턴스를 분리하는 방식이다. Deep Watershed Transform이나 픽셀 임베딩을 이용한 Metric Learning 방식(De Brabandere et al. [23])이 이에 해당한다.
4. **Weakly-supervised semantic segmentation**: 타겟 라벨을 동적으로 수정하며 학습하는 방식으로, 본 논문의 동적 색상 할당 과정과 철학적으로 유사하다.

본 연구는 특히 Metric Learning과 Clustering 기반의 후처리를 사용하는 [23]의 방식과 유사하지만, 이를 **Classification Learning**과 **Connected Component Analysis**로 대체함으로써 파이프라인을 획기적으로 단순화하고 속도를 높였다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인 및 추론 절차 (Inference)

추론 단계는 크게 두 단계로 구성된다.

1. **Coloring Stage**: 입력 이미지 $x$를 Coloring Network $\Psi$ (주로 U-Net 아키텍처 사용)에 통과시켜 $C$개의 채널을 가진 확률 맵 $y \in \mathbb{R}^{C \times W \times H}$를 생성한다. 마지막 층의 Softmax 함수를 통해 각 픽셀 $p$는 $C$개의 색상 중 하나를 가질 확률을 부여받는다. 이때 첫 번째 채널($c=1$)은 배경(Background)으로 예약된다.
2. **Post-processing Stage**:
    * 각 픽셀에서 가장 확률이 높은 색상을 선택하여 맵 $z$를 생성한다.
    * $z$에서 Connected Component Analysis(CCA)를 수행하여 연결된 성분들을 찾는다.
    * 크기 임계값 $\tau$보다 작은 성분은 배경으로 처리하여 노이즈를 제거한다.
    * (옵션) 동일한 색상을 가진 두 성분 사이의 Hausdorff 거리가 임계값 $\rho$보다 작으면, 폐색(Occlusion)으로 인해 분리된 동일 객체로 판단하여 병합한다.

### 학습 과정 (Learning)

학습의 목표는 각 객체 인스턴스가 동일한 색상으로 칠해지게 하되, 인접한 서로 다른 인스턴스는 서로 다른 색상을 갖게 하는 것이다. 이를 위해 본 논문은 고정된 정답 라벨 대신 **동적 색상 할당(Dynamic Coloring)** 방식을 제안한다.

**1. Halo Region 정의**
객체 $M_k$ 주변의 마진 거리 $m$ 내에 있는 픽셀 집합을 Halo region $M_k^{halo}$라고 정의한다. 이는 형태학적 팽창(Morphological dilation)을 통해 다음과 같이 계산된다.
$$M_k^{halo} = \text{dilate}(M_k, m) \setminus M_k$$

**2. 동적 색상 선택 (Coloring Rule)**
각 학습 에폭마다, 네트워크의 현재 예측값 $y$를 기반으로 각 인스턴스 $k$에 대해 다음 목적 함수를 최대화하는 색상 $c_k$를 동적으로 선택한다.
$$c_k = \arg \max_{c=2}^C \left( \frac{1}{|M_k|} \sum_{p \in M_k} \log y[c,p] + \mu \frac{1}{|M_k^{halo}|} \sum_{p \in M_k^{halo}} \log(1 - y[c,p]) \right)$$
이 식의 의미는 객체 내부($M_k$)에서는 해당 색상의 확률을 높이고, 주변 Halo 영역($M_k^{halo}$)에서는 해당 색상의 확률을 낮추는 색상을 선택하겠다는 것이다. $\mu$는 Halo 영역의 영향력을 조절하는 가중치이다.

**3. 손실 함수 (Loss Function)**
선택된 $c_k$를 가짜 정답(Pseudo ground-truth)으로 사용하여 표준 픽셀 단위 로그 손실(Log-loss)을 계산하고 역전파한다.
$$L(x, \theta) = -\sum_{k=1}^K \frac{1}{|M_k|} \sum_{p \in M_k} \log y[c_k, p] - \sum_{p \in \text{Background}} \log y[1, p]$$

## 📊 Results

### 실험 설정

* **데이터셋**: CVPPP (식물 잎), E.Coli (미생물 현미경 이미지), Cityscapes (자율주행 도로 장면).
* **아키텍처**: 기본적으로 U-Net 스타일의 Encoder-Decoder 구조를 사용하였으며, Cityscapes의 경우 성능 향상을 위해 PSP-module과 Batch Normalization을 추가하였다.
* **지표**: CVPPP에서는 SBD(Symmetric Best Dice coefficient)와 $|DiC|$(개수 차이)를, Cityscapes에서는 AP(Average Precision)를 사용하였다.

### 주요 결과

1. **CVPPP 데이터셋**: 제안 방법이 대부분의 기존 방법보다 우수한 성능(SBD 0.87)을 보였다. 특히 임베딩 기반 방식([23])과 비교했을 때, 후처리 속도가 수십 배 빠르며(0.05s vs 30s), 하이퍼파라미터 튜닝이 훨씬 용이함을 확인하였다.
2. **E.Coli 데이터셋**: 매우 많은 수의 인스턴스가 밀집된 환경에서도 효과적으로 작동하였다. Clustering 기반 방식([23])은 인스턴스 수가 많아질수록 성능이 급격히 떨어졌으나, 제안된 CCA 기반 방식은 안정적인 성능을 유지하였다.
3. **Cityscapes 데이터셋**:
    * 단순히 클래스별 색상을 할당하는 방식보다는, 별도의 Semantic Segmentation 헤드를 두어 결과를 퓨전하는 방식이 더 높은 성능을 보였다.
    * 특히 PSP-Net의 Semantic Segmentation 결과와 Deep Coloring의 인스턴스 분리 능력을 결합했을 때 가장 좋은 성능(AP 25.2)을 달성하여, 제안 방법이 경쟁력 있는 Proposal-free 방식임을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 통찰

* **단순성 및 효율성**: Instance Segmentation을 Semantic Segmentation의 형태로 환원함으로써, 기존의 수많은 Semantic Segmentation 최적화 기법(U-Net, PSP-Net 등)을 그대로 적용할 수 있다.
* **확장성**: 고정된 색상 수만으로도 공간적 분리를 통해 무한한 수의 인스턴스를 처리할 수 있으며, 추론 속도가 객체 수에 영향을 받지 않는다.
* **동적 학습의 효과**: 정적인 라벨링 대신 네트워크의 예측 상태에 따라 정답을 동적으로 생성하는 방식이 학습의 유연성을 높였다.

### 한계 및 비판적 해석

* **하이퍼파라미터 민감도**: 마진 거리 $m$과 가중치 $\mu$의 설정에 따라 Undersegmentation(인접 객체를 하나로 인식) 또는 Fragmentation(하나의 객체를 여러 개로 인식) 현상이 발생한다. 비록 실험적으로 둔감하다고 주장하지만, 데이터셋의 객체 크기에 따라 최적값이 달라지므로 사전 분석이 필요하다.
* **Semantic 성능 의존성**: Cityscapes 실험에서 드러났듯, 복잡한 장면에서는 정확한 Semantic Segmentation 결과가 뒷받침되어야 인스턴스 분리 성능이 극대화된다. 즉, Coloring Network 단독으로는 고수준의 시맨틱 이해를 수행하는 데 한계가 있을 수 있다.

## 📌 TL;DR

본 논문은 Instance Segmentation 문제를 '고정된 색상을 이용한 동적 색칠' 문제로 정의하여 Semantic Segmentation 아키텍처로 해결하는 **Deep Coloring** 방법을 제안한다. 학습 시에는 인접 객체가 서로 다른 색상을 갖도록 동적으로 라벨을 할당하고, 추론 시에는 CCA를 통해 객체를 분리한다. 이 방식은 구현이 매우 단순하고 추론 속도가 빠르며, 특히 객체 수가 매우 많은 데이터셋에서 Clustering 기반 방식보다 강점을 보인다. 향후 다양한 도메인의 인스턴스 분리 작업에 효율적인 베이스라인으로 활용될 가능성이 높다.
