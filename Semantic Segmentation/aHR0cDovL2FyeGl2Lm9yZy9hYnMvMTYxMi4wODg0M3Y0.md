# FastMask: Segment Multi-scale Object Candidates in One Shot

Hexiang Hu, Shiyi Lan, Yuning Jiang, Zhimin Cao, Fei Sha (2017)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 이미지 내 객체의 크기 변화(scale variance)에 따른 Segment Proposal의 성능 저하 문제이다. 일반적인 객체 탐지에서 Bounding Box 기반의 Proposal은 대략적인 위치 추정만으로도 어느 정도 유연하게 대응할 수 있으나, 픽셀 단위의 영역을 구분해야 하는 Segment Proposal은 객체 크기와 네트워크의 수용 영역(Receptive Field)이 매우 정밀하게 일치해야 한다.

수용 영역이 객체보다 너무 작을 경우 객체의 전체 윤곽을 파악하지 못해 불완전한 세그먼트를 생성하게 되며, 반대로 너무 클 경우 주변 배경의 노이즈가 유입되어 인접한 다른 객체까지 하나의 마스크로 묶어버리는 문제가 발생한다. 기존의 연구들은 이를 해결하기 위해 입력 이미지를 여러 크기로 조정하여 반복적으로 처리하는 Image Pyramid 전략(Multi-shot inference)을 사용하였으나, 이는 막대한 연산 비용을 초래하여 실시간 적용에 큰 제약이 된다. 따라서 본 논문의 목표는 단 한 번의 추론(One-shot inference)만으로 다양한 크기의 객체를 효율적이고 정확하게 세그먼트할 수 있는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 CNN의 계층적 특징 맵(Hierarchical Feature Map)을 활용하여 이미지 피라미드를 대체하는 것이다. 이를 위해 네트워크 구조를 Body, Neck, Head라는 세 가지 기능적 컴포넌트로 분리하여 설계하였다.

가장 중점적인 기여는 첫째, 가중치를 공유하는 Residual Neck 모듈을 제안하여 특징 세만틱을 보존하면서 효율적으로 특징 피라미드(Feature Pyramid)를 생성한 점이다. 둘째, Visual Attention 메커니즘을 도입한 Scale-tolerant Attentional Head 모듈을 통해 수용 영역과 객체 크기의 불일치로 발생하는 배경 노이즈를 억제하고 강건한 세그멘테이션 마스크를 생성한 점이다. 최종적으로 이러한 구조를 통해 연산 효율성을 극대화하면서도 기존의 Multi-shot 방식보다 우수한 성능을 내는 One-shot Segment Proposal 프레임워크를 구현하였다.

## 📎 Related Works

기존의 객체 Proposal 연구는 크게 Bounding Box 기반과 Segment 기반으로 나뉜다. EdgeBox나 Bing, 그리고 최근의 RPN(Region Proposal Network)과 같은 Bounding Box 기반 방식들은 슬라이딩 윈도우나 앵커 박스를 통해 효율적으로 후보 영역을 찾는다. 반면 Segment Proposal은 SelectiveSearch나 MCG와 같은 전통적인 Bottom-up 방식에서 시작하여, 최근에는 DeepMask나 SharpMask와 같이 CNN을 이용해 마스크를 직접 예측하는 방식으로 발전하였다.

특히 DeepMask와 SharpMask는 CNN 특징 맵으로부터 마스크를 디코딩하는 Body-Head 구조를 사용하지만, 추론 과정에서 Image Pyramid를 필수적으로 사용한다. 이는 동일한 연산을 다양한 스케일의 이미지에 대해 반복 수행하게 만들며, 이로 인해 연산 병목 현상이 발생한다. 본 연구는 이러한 Image Pyramid 기반의 Multi-shot 패러다임을 특징 맵 수준에서 해결하는 One-shot 패러다임으로 전환하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

FastMask는 입력 이미지에서 최종 세그먼트 마스크를 추출하기까지 **Body $\rightarrow$ Neck $\rightarrow$ Head** 순의 파이프라인을 거친다.

1. **Body**: VGGNet이나 ResNet과 같은 기본 CNN을 사용하여 입력 이미지로부터 고차원 세만틱 특징 맵을 추출한다.
2. **Neck**: 추출된 특징 맵을 재귀적으로 다운샘플링하여 다양한 스케일의 특징 피라미드를 생성한다.
3. **Head**: 각 스케일의 특징 맵에서 슬라이딩 윈도우를 통해 윈도우 특징을 추출하고, 이를 통해 객체의 존재 확률(Confidence Score)과 세그먼트 마스크를 디코딩한다.

### Residual Neck

단순한 Max Pooling은 특징 맵의 평균값을 상승시켜 스케일 간 보정이 어렵게 만들고, Average Pooling은 변별력 있는 특징을 뭉개뜨리는 경향이 있다. 이를 해결하기 위해 본 논문은 Average Pooling에 학습 가능한 잔차 연결(Residual Component)을 추가한 **Residual Neck**을 제안한다.
Residual Neck은 $3 \times 3$ Convolution 레이어와 $1 \times 1$ Convolution 레이어로 구성된 파라미터 모듈을 Average Pooling과 병렬로 배치하여, 특징 세만틱을 보존하면서도 효율적으로 특징 맵의 크기를 줄인다. 또한, 이 Neck 모듈은 가중치를 공유(Weight-shared)하므로, 추론 시 Neck의 개수를 조절하여 속도와 정확도 간의 Trade-off를 쉽게 조정할 수 있다.

### Attentional Head

수용 영역과 객체 크기의 불일치 문제를 해결하기 위해, 본 논문은 단순 디코딩 대신 **Visual Attention**을 적용한 Head를 제안한다.
슬라이딩 윈도우로 추출된 특징 맵이 입력되면, 먼저 Fully Connected 레이어를 통해 공간적 어텐션 맵(Spatial Attention Map)을 생성한다. 이후 이 어텐션 맵을 원래 특징 맵에 요소별 곱셈(Element-wise multiplication)하여 중요한 영역을 강조하고 배경 노이즈를 제거한다. 이렇게 정제된 특징 맵이 최종적으로 마스크 디코더로 전달되어 세그먼트 마스크를 생성한다.

### Two-stream Network

특징 피라미드의 스케일 밀도를 높이기 위해 Body 네트워크 중간에 서로 다른 스트라이드(예: 2와 3)를 가진 풀링 레이어를 두어 두 개의 스트림으로 분기하는 구조를 채택하였다. 이를 통해 단순 2의 배수 형태가 아닌 더 다양한 스케일의 특징 맵을 생성하여 스케일 변화에 대한 강건성을 더욱 높였다.

### 학습 목표 및 손실 함수

FastMask는 신뢰도 손실($L_{conf}$), 세그멘테이션 손실($L_{seg}$), 영역 어텐션 손실($L_{att}$)의 가중 합으로 구성된 전체 손실 함수를 최적화한다.

$$L(c,a,s) = \frac{1}{N} \sum_{k}^{N} [ L^{conf}(c^k, \hat{c}^k) + \mathbb{1}(c^k) \cdot ( L^{seg}(s^k, \hat{s}^k) + L^{att}(a^k, \hat{a}^k) ) ]$$

여기서 $\mathbb{1}(c^k)$는 해당 윈도우가 양성 샘플(객체 포함)일 때만 1을 반환하는 지시 함수이다. 즉, 세그멘테이션과 어텐션 손실은 객체가 존재하는 양성 샘플에 대해서만 역전파를 수행한다. 각 손실 함수는 다음과 같이 Binary Cross Entropy(BCE)를 기반으로 계산된다.

$$E(y, \hat{y}) = y \cdot \log(\sigma(\hat{y})) + (1-y) \cdot \log(1-\sigma(\hat{y}))$$

신뢰도 손실은 다음과 같으며:
$$L^{conf}(c, \hat{c}) = -E(s_{i,j}, \hat{s}_{i,j})$$

세그멘테이션과 어텐션 손실은 윈도우 크기($w, h$)로 정규화하여 계산한다:
$$L^{seg}(s, \hat{s}) = -\left[ \frac{1}{w \cdot h} \sum_{i,j}^{h,w} ( E(s_{i,j}, \hat{s}_{i,j}) ) \right]$$
$$L^{att}(a, \hat{a}) = -\left[ \frac{1}{w \cdot h} \sum_{i,j}^{h,w} ( E(a_{i,j}, \hat{a}_{i,j}) ) \right]$$

## 📊 Results

### 실험 설정

- **데이터셋**: MS COCO 벤치마크 (학습 80k, 검증 5k 이미지)
- **평가 지표**: IoU 0.5에서 0.95 사이의 Average Recall (AR@10, AR@100, AR@1k)
- **비교 대상**: DeepMask, SharpMask, InstanceFCN 등 최신 Segment Proposal 방법론들

### 정량적 결과

실험 결과, FastMask는 Bounding Box Proposal 성능에서 기존 방법들을 압도적인 차이로 앞섰으며, Segment Proposal에서도 매우 경쟁력 있는 성능을 보였다. 특히 ResNet-39를 Body로 사용한 Two-stream FastMask는 SharpMask 대비 AR@10에서 약 18%, AR@100에서 11%, AR@1k에서 8%의 상대적 성능 향상을 기록하였다.

### 효율성 분석

연산 속도 면에서 FastMask는 Image Pyramid를 사용하는 Multi-shot 방식보다 훨씬 빠르다. NVIDIA Titan X GPU 기준, 성능 중심 모델(FastMask-acc)뿐만 아니라 경량화된 PvaNet 기반의 모델(FastMask-fast)은 800x600 해상도 이미지에 대해 약 13 FPS의 속도로 거의 실시간에 가까운 추론 성능을 보여주었다. 이는 기존 Multi-shot 방식들이 수 초의 시간이 걸리는 것과 대비하여 수 배에서 수십 배 빠른 속도이다.

## 🧠 Insights & Discussion

본 논문은 Segment Proposal의 고질적인 문제였던 수용 영역-객체 크기 불일치 문제를 '특징 피라미드 생성(Residual Neck)'과 '공간적 주의 집중(Attentional Head)'이라는 두 가지 전략으로 해결하였다.

특히 주목할 점은 단순한 구조 변경만으로 성능을 높인 것이 아니라, 왜 기존의 Max/Avg Pooling이 부적절한지를 분석하고 이를 보완할 Residual 구조를 제안했다는 점이다. 또한, Attentional Head가 배경 노이즈를 효과적으로 제거하여 마스크의 정밀도를 높이는 것을 시각화 결과(Figure 6)를 통해 입증하였다.

다만, 본 논문에서는 마스크 정교화(Mask Refinement) 단계를 별도로 수행하지 않았음에도 높은 성능을 보였는데, 이는 추후 SharpMask와 같은 정교화 모듈을 결합한다면 더 높은 정확도를 얻을 수 있을 가능성을 시사한다. 또한, Two-stream 구조가 스케일 밀도를 높여 성능을 향상시키지만, 이는 연산량의 소폭 증가를 동반하므로 응용 분야에 따라 적절한 스트림 수와 Neck 개수를 설정하는 것이 중요할 것이다.

## 📌 TL;DR

FastMask는 이미지 피라미드를 생성하는 대신, CNN 내부에서 **Weight-shared Residual Neck**을 통해 특징 피라미드를 구축하고 **Attentional Head**로 배경 노이즈를 제거하는 One-shot Segment Proposal 프레임워크이다. 이를 통해 MS COCO 벤치마크에서 기존 SOTA 모델들보다 높은 Average Recall을 달성함과 동시에 추론 속도를 획기적으로 개선하여 실시간 적용 가능성을 입증하였다. 이 연구는 효율적인 객체 후보 영역 추출이 필요한 실시간 비디오 분석이나 모바일 환경의 객체 탐지 시스템에 매우 중요한 역할을 할 것으로 기대된다.
