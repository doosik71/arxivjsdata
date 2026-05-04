# Instance Embedding Transfer to Unsupervised Video Object Segmentation

Siyang Li, Bryan Seybold, Alexey Vorobyov, Alireza Fathi, Qin Huang, and C.-C. Jay Kuo (2018)

## 🧩 Problem to Solve

본 논문은 비디오 내에서 사용자의 가이드나 사전 학습된 비디오 데이터셋 없이 주요 객체를 자동으로 분리해내는 **Unsupervised Video Object Segmentation (UVOS)** 문제를 해결하고자 한다.

일반적인 비디오 객체 분할 작업은 주석(Annotation) 비용이 매우 높기 때문에 대규모 데이터셋을 구축하기 어렵다. 기존의 많은 방법론들은 정적 이미지 분할 네트워크를 전이 학습시키거나 비디오 도메인에 맞게 파인튜닝(Fine-tuning)하는 방식을 취했다. 그러나 직접적인 Foreground/Background(FG/BG) 분류 모델을 사용하면, 동일한 객체(예: 자동차)가 어떤 비디오에서는 전경(FG)이 되고 어떤 비디오에서는 배경(BG)이 되는 상황에서 모델이 혼란을 겪으며, 이를 해결하기 위해 매 시퀀스마다 온라인 파인튜닝이 필요하다는 한계가 있다.

따라서 본 논문의 목표는 정적 이미지에서 학습된 **Instance Embedding** 지식을 비디오로 전이하여, 별도의 재학습이나 온라인 파인튜닝 없이도 안정적으로 비디오 객체를 분할하는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 픽셀을 단순히 FG/BG로 분류하는 대신, 동일한 객체에 속한 픽셀들이 유사한 벡터 값을 갖도록 하는 **Instance Embedding** 공간을 활용하는 것이다.

1. **Instance Embedding 전이 전략**: 정적 이미지 데이터셋에서 학습된 인스턴스 임베딩 네트워크를 비디오에 적용하여, 모델의 재학습 없이도 객체의 정체성을 유지하는 특징량을 추출한다.
2. **비지도 기반 전경 객체 선택 기준**: Semantic Objectness score와 Optical Flow를 이용한 Motion Saliency를 결합하여, 어떤 임베딩이 전경 객체에 해당하는지를 결정하는 새로운 기준을 제시한다.
3. **임베딩 안정성 분석**: 비디오 프레임 간 임베딩의 일관성을 분석하고, 시간 경과에 따라 발생하는 Embedding Drift 문제를 해결하기 위한 효율적인 온라인 적응(Online Adaptation) 방법을 제안한다.

## 📎 Related Works

### Unsupervised Video Object Segmentation
기존의 비지도 방식은 픽셀의 계층적 클러스터링이나 Gaussian Mixture Model, Graph Cut 등을 사용하여 전경과 배경을 분리했다. 최근에는 CNN을 이용해 Saliency, Edge, Motion 특징을 결합하는 방식(예: LMP, LVO, FSEG)이 주를 이루었으나, 여전히 전경 객체를 정확히 식별하는 데 어려움이 있었다.

### Semi-supervised Video Object Segmentation
첫 번째 프레임의 마스크 정보를 제공받는 반지도 학습 방식(예: OSVOS, OnAVOS)은 매우 높은 정확도를 보이지만, 첫 프레임에 대한 사용자 주석이 필수적이며 많은 경우 온라인 파인튜닝에 상당한 시간이 소요된다는 단점이 있다.

### Image Segmentation
Semantic Segmentation은 픽셀의 카테고리를 예측하지만 동일 카테고리의 서로 다른 인스턴스를 구분하지 못한다. 반면, Instance Embedding 방식은 동일 인스턴스 내 픽셀 간의 거리를 좁히는 Metric Learning을 통해 각 객체를 고유하게 식별할 수 있게 하며, 본 논문은 이 특성이 비디오 내 객체 추적 및 분할에 유용할 것이라는 가설을 세웠다.

## 🛠️ Methodology

### 1. Feature Extraction (특징 추출)
본 시스템은 세 가지 입력 특징을 독립적으로 추출하며, 비디오 데이터셋에 대한 추가 학습을 진행하지 않는다.

- **Instance Embedding**: 정적 이미지로 학습된 CNN의 첫 번째 헤드에서 출력되며, 동일 객체 픽셀 간의 유클리드 거리를 최소화한다. 두 픽셀 $i, j$ 사이의 유사도 $R(i,j)$는 다음과 같이 정의된다.
$$R(i,j) = \frac{2}{1 + \exp(||f(i)-f(j)||_2^2)}$$
학습 시에는 정답지 $g(i,j)$(동일 객체 여부)와 유사도 간의 Cross Entropy Loss $\mathcal{L}_s$를 최소화한다.
- **Objectness Score**: 네트워크의 두 번째 헤드에서 출력되는 Semantic Segmentation 결과 중, 배경(Background)일 확률 $P_{BG}(i)$를 이용하여 다음과 같이 계산한다.
$$O(i) = 1 - P_{BG}(i)$$
- **Optical Flow**: FlowNet 2.0을 사용하여 프레임 간의 픽셀 움직임을 측정한다.

### 2. Generating Proposal Seeds (제안 시드 생성)
전경과 배경을 대표할 수 있는 소수의 대표점(Seed)을 선정한다.
- **Candidate Points**: 임베딩 공간에서 국소적으로 일관된(Locally Consistent) 영역만을 후보군 $C$로 선정하여 객체의 경계선을 제외한다.
- **Diverse Seeds**: KMeans++ 초기화 방식을 채택하여, Objectness score가 가장 높은 점을 먼저 선택한 후, 기존 선택된 시드들과의 유사도가 가장 낮은 점들을 반복적으로 추가하여 다양성을 확보한 시드 집합 $S$를 구성한다.

### 3. Ranking Proposed Seeds (시드 랭킹)
추출된 시드 중 어떤 것이 실제 전경 객체인지 판별하기 위해 다음 과정을 거친다.
- **Motion Saliency**: 객체성 점수가 가장 낮은 시드들을 통해 배경 모션 모델 $V_{BG}$를 구축하고, 각 시드의 평균 광학 흐름 $v_s$와 배경 모션 간의 거리를 통해 Motion Saliency $M(s)$를 계산한다.
- **Seed Tracks**: 프레임 간 임베딩 유사도가 높은 시드들을 연결하여 궤적(Track) $T_j$를 형성한다.
- **Foreground Score**: 각 궤적에 대해 Objectness와 Motion Saliency의 곱의 평균을 구해 전경 점수 $F(T_j)$를 산출한다.
$$F(T_j) = \frac{1}{|T_j|} \sum_{s \in T_j} O(s)M(s)$$

### 4. Final Segmentation (최종 분할)
- **Initial FG/BG**: 가장 높은 $F(T_j)$를 가진 시드를 전경 시드 $s_{FG}$로, Objectness가 낮은 시드들을 배경 시드 $S_{BG}$로 설정한다.
- **Seed Expansion**: 전경 마스크가 객체 전체를 덮지 못하는 문제를 해결하기 위해, 초기 분할 영역과 겹치는 비율이 높은 주변 시드들을 전경 집합 $S_{FG}$에 추가한다. 배경 역시 Objectness 또는 Motion Saliency 임계값 이하인 시드들을 추가하여 확장한다.
- **Pixel Classification**: 각 픽셀 $i_l$에 대해 전경/배경 시드 집합과의 최대 유사도를 기반으로 전경 확률 $P_{FG}(i_l)$을 계산한다.
$$P_{FG}(i_l) = \frac{R_{FG}(i_l)}{R_{FG}(i_l) + R_{BG}(i_l)}$$
최종적으로 **Dense CRF**를 적용하여 마스크의 경계선을 정교하게 다듬는다.

## 📊 Results

### 실험 설정
- **데이터셋**: DAVIS 2016, FBMS, SegTrack-v2.
- **평가 지표**: Region Similarity ($\mathcal{J}$, IoU) 및 Boundary Accuracy ($\mathcal{F}$).
- **구현 세부사항**: PASCAL VOC 2012로 학습된 DeepLab-v2(ResNet backbone) 기반의 인스턴스 임베딩 네트워크를 사용하였으며, 임베딩 차원 $E=64$로 설정하였다.

### 정량적 결과
- **DAVIS**: $\mathcal{J}$-mean 78.5%, $\mathcal{F}$-mean 75.5%를 기록하며 기존 비지도 학습 방법론들보다 높은 성능을 보였다. 특히, 첫 프레임 주석을 사용하는 일부 반지도 학습 방법(VPN, SegFlow)보다 우수한 성능을 나타냈다.
- **FBMS**: F-score 82.8% 및 $\mathcal{J}$-mean 71.9%를 달성하여 SOTA 성능을 기록하였다.
- **SegTrack-v2**: $\mathcal{J}$ 59.3%를 달성하였다.

### 분석 및 소결
- **Instance Embedding의 효용성**: DeepLab-v2의 fc7 특징량을 사용했을 때보다 인스턴스 임베딩을 사용했을 때 $\mathcal{J}$ 값이 약 10% 이상 향상되어, Metric Learning 기반의 임베딩이 객체 추적에 훨씬 유리함을 입증하였다.
- **Online Adaptation**: 시간이 흐를수록 임베딩이 변하는 'Embedding Drift' 현상이 발견되었으며, 매 프레임 시드를 업데이트하는 온라인 적응을 통해 $\mathcal{J}$ 성능을 7.0% 포인트 향상시켰다.
- **Ranking 전략**: Motion Saliency와 Objectness를 모두 사용했을 때 시드 선정 정확도가 가장 높았으며, 이는 두 지표가 서로 보완적인 역할을 하기 때문이다.

## 🧠 Insights & Discussion

본 논문은 단순한 분류기가 아닌 **임베딩 공간의 거리 기반 전이 학습**이라는 관점에서 비디오 객체 분할에 접근하였다. 이는 특정 객체 클래스에 종속되지 않고 임베딩의 상대적 유사성만을 이용하기 때문에, 학습 데이터에 없던 새로운 카테고리의 객체(예: 염소 등)에 대해서도 강건하게 작동한다는 강점이 있다.

다만, 몇 가지 한계점이 존재한다. 첫째, 광학 흐름(Optical Flow)의 부정확성으로 인해 전경 객체가 배경과 비슷하게 움직이거나, 반대로 배경이 급격히 움직일 때 오작동할 가능성이 있다. 둘째, 객체가 완전히 가려지는(Occlusion) 상황에서는 임베딩 정보만으로는 전경을 유지하기 어려워 오분류가 발생한다. 

비판적으로 해석하자면, 본 방법론은 임베딩의 안정성에 크게 의존하고 있으며, 온라인 적응 과정이 사실상 모델의 파라미터를 업데이트하는 파인튜닝의 역할을 시드 선택 단계에서 간접적으로 수행하고 있는 것으로 볼 수 있다.

## 📌 TL;DR

- **핵심**: 정적 이미지에서 학습된 Instance Embedding 네트워크를 비디오에 적용하여, 재학습 없이 전경 객체를 분리하는 비지도 학습 방법론을 제안함.
- **방법**: 임베딩 특징, Objectness, Optical Flow를 결합하여 전경/배경 대표 시드를 선정하고, 픽셀 간 임베딩 유사도를 통해 최종 마스크를 생성함.
- **성과**: DAVIS 및 FBMS 데이터셋에서 기존 비지도 SOTA 모델들을 능가하는 성능을 보였으며, 일부 반지도 학습 모델보다 우수한 결과를 냄.
- **의의**: 비디오 데이터셋에 대한 직접적인 학습 없이도 이미지 레벨의 지식을 효과적으로 전이하여 비디오 객체 분할에 활용할 수 있음을 보여줌.