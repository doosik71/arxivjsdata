# PWISEG: Point-Based Weakly-Supervised Instance Segmentation for Surgical Instruments

Zhen Sun, Huan Xu, Jinlin Wu, Zhen Chen, Zhen Lei, Hongbin Liu (2023)

## 🧩 Problem to Solve

수술 과정에서 사용되는 기구들을 정확하게 계수하는 것은 환자의 안전을 위해 매우 중요하다. 수술 도구가 체내에 잔류할 경우 감염이나 심각한 신체적 손상을 초래할 수 있기 때문이다. 하지만 수술실 환경에서는 기구들이 밀집되어 있거나 서로 겹쳐 있는 occlusion 현상이 빈번하게 발생하며, 이는 기존의 단순한 object detection 방식만으로는 정확한 위치 파악과 계수를 어렵게 만든다.

이러한 문제를 해결하기 위해 bounding box뿐만 아니라 픽셀 단위의 상세 정보를 제공하는 Instance Segmentation이 효과적인 대안이 될 수 있다. 그러나 Instance Segmentation을 학습시키기 위해 필요한 mask-level annotation은 막대한 비용과 노동력이 소모된다는 치명적인 단점이 있다. 따라서 본 논문의 목표는 mask-level annotation 없이, 상대적으로 획득하기 쉬운 bounding box와 소수의 key point만을 활용하여 수술 기구를 정확하게 분할하는 Weakly-supervised Instance Segmentation 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 FCN(Fully Convolutional Network) 기반의 구조를 사용하여 point-to-box 및 point-to-mask의 두 가지 브랜치를 통해 특징점(feature points)과 bounding box, 그리고 segmentation mask 간의 관계를 모델링하는 것이다.

특히, mask annotation의 부재를 극복하기 위해 세 가지의 핵심 손실 함수를 도입하였다. 첫째, 예측된 mask와 bounding box 간의 투영 관계를 이용하는 Unsupervised Projection Loss를 통해 기본적인 형태를 잡는다. 둘째, 소수의 key pixel을 활용하여 픽셀 간의 유사도(affinity)를 계산하고 이를 전파하여 pseudo-label을 생성하는 Key Pixel Association Loss를 제안하였다. 셋째, key pixel의 분포를 Gaussian kernel로 확장하여 예측된 probability map과의 거리를 최적화하는 Key Pixel Distribution Loss를 도입하여 세밀한 분할 성능을 높였다. 또한, 수술 기구 인식을 위한 새로운 고해상도 데이터셋을 구축하여 벤치마크를 제시하였다.

## 📎 Related Works

본 논문에서는 BoxInst, Discobox, BoxLevelSet과 같이 bounding box만을 이용하여 instance segmentation을 수행하는 기존의 weakly-supervised 방식들을 비교 대상으로 삼고 있다. 이러한 기존 방식들은 mask-level annotation의 비용 문제를 해결하려 노력하였으나, 수술 기구와 같이 형태가 정형화되어 있으면서도 occlusion이 심한 특수한 도메인에서의 성능 최적화에 한계가 있을 수 있다. PWISeg은 단순히 box 정보에만 의존하지 않고, 매우 적은 비용의 key point 정보를 추가로 활용함으로써 기존의 box-supervised 방법론들보다 더 정교한 mask 생성이 가능함을 보여준다.

## 🛠️ Methodology

### 전체 시스템 구조

PWISeg은 FCN 기반의 아키텍처를 채택하며, FPN(Feature Pyramid Network) 상에서 point-to-box 브랜치와 point-to-mask 브랜치가 병렬적으로 작동한다. 이를 통해 단일 모델 내에서 기구의 검출(detection)과 분할(segmentation)을 동시에 수행한다.

### Supervised Point to Box

Bounding box를 이용한 학습 단계에서는 각 위치의 카테고리 예측과 박스 회귀를 수행한다. Ground-truth bounding box $B_i$는 좌상단 좌표, 우하단 좌표, 그리고 클래스 $c^{(i)}$로 정의된다. 이때 사용하는 손실 함수는 다음과 같다.

$$L(c(x,y), t(x,y)) = \frac{1}{N_{pos}} \sum_{(x,y)} L_{cls}(\hat{c}(x,y), c(x,y)) + \lambda \frac{1}{N_{pos}} \sum_{(x,y)} \mathbb{1}_{\{c(x,y)>0\}} L_{reg}(\hat{t}(x,y), t(x,y))$$

여기서 $L_{cls}$에는 Focal Loss를, $L_{reg}$에는 IoU Loss를 사용하여 분류 정확도와 박스 회귀 정밀도를 동시에 최적화한다.

### Unsupervised Point to Mask

Mask 수준의 정답지가 없으므로, 다음과 같은 세 가지 손실 함수의 조합으로 학습을 진행한다.

**1. Unsupervised Projection Loss ($L_{proj}$)**
예측된 mask를 x축과 y축으로 투영(projection)했을 때의 최대값들이 실제 bounding box의 경계와 일치하도록 유도한다.
$$L_{proj} = \text{Dice}(\max_x(\hat{w}), \max_x(t)) + \text{Dice}(\max_y(\hat{w}), \max_y(t))$$
여기서 $\hat{w}$는 예측된 mask이며 $t$는 bounding box annotation이다.

**2. Key-pixels Association Loss ($L_{ass}$)**
소수의 key pixel $I(i,j)$와 주변 픽셀 $I(x,y)$ 간의 유사도 $A$를 다음과 같이 정의한다.
$$A \{(i,j), (x,y)\} = \hat{p}_{i,j} \cdot \hat{p}_{x,y} + (1 - \hat{p}_{i,j}) \cdot (1 - \hat{p}_{x,y})$$
이 유사도가 임계값 $\lambda$보다 크면 동일한 라벨을 가진 것으로 간주하여 pseudo-label $\hat{y}(x,y)$를 생성하고, 이를 기반으로 cross-entropy 형태의 손실 함수를 적용한다.
$$L_{ass} = -\frac{1}{N} \sum_{(x,y) \in bbox} \hat{y}(x,y) \log p(x,y) + (1 - \hat{y}(x,y)) \log p(x,y)$$

**3. Key-pixels Distribution Loss ($L_{dis}$)**
모델의 응답 값이 낮을 때 key point 주변의 전파가 어려운 문제를 해결하기 위해, key pixel에 Gaussian kernel $G(x,y) = \exp(-\frac{x^2+y^2}{2\sigma^2})$을 적용하여 생성한 heatmap $\Phi(h(x,y))$와 예측된 probability map $p(x,y)$ 사이의 $L_1$ 거리를 최소화한다.
$$L_{dis} = \sum_{(x,y) \in bbox} \|p(x,y) - \Phi(h(x,y))\|_{L_1}$$

최종적인 segmentation 손실 함수는 다음과 같이 합산된다.
$$L_{seg} = L_{proj} + \lambda_1 L_{ass} + \lambda_2 L_{dis}$$

## 📊 Results

### 실험 설정

- **데이터셋**: 자체 구축한 신규 수술 기구 데이터셋(1,280 $\times$ 960 해상도, 10,000개 이상의 기구 주석 포함) 및 공개 데이터셋인 HOSPI-Tools를 사용하였다.
- **구현 상세**: PyTorch 프레임워크와 ResNet-50 백본을 사용하였으며, SGD 최적화 알고리즘을 통해 Nvidia GeForce RTX 4090 GPU에서 학습하였다. 학습률은 0.0001에서 시작하여 LinearLR 및 MultiStepLR 스케줄러를 통해 조정되었다.

### 정량적 결과

- **제안 데이터셋**: PWISeg은 Segmentation mAP 23.9%, Detection mAP 64.2%를 달성하였다. 특히 $\text{mAP}_{50}$ 기준으로는 Segmentation 66.3%, Detection 96.8%라는 높은 수치를 기록하며 BoxInst, Discobox, BoxLevelSet 등 기존 방법론들을 능가하였다.
- **HOSPI-Tools 데이터셋**: Segmentation mAP 30.6%를 기록하여 비교 대상 중 가장 높은 성능을 보였으며, Detection mAP는 73.2%로 Discobox(74.2%)보다는 약간 낮았으나 $\text{mAP}_{50}$에서는 95.2%로 최상위권의 성능을 보였다.

## 🧠 Insights & Discussion

본 연구는 매우 제한적인 주석(bounding box와 소수의 key point)만으로도 수준 높은 instance segmentation이 가능하다는 것을 입증하였다. 특히 단순한 box 기반의 weakly-supervised 방식에 key pixel이라는 최소한의 포인트 정보를 추가함으로써, mask-level annotation 없이도 segmentation 성능을 유의미하게 끌어올린 점이 강점이다.

다만, 전체 mAP 수치가 full supervision 방식에 비해 여전히 낮게 형성되어 있다는 점은 weakly-supervised learning의 본질적인 한계로 보인다. 또한, 제안된 데이터셋의 규모가 training set 기준 1,788장으로 비교적 작아, 더 대규모의 데이터셋에서 일반화 성능을 검증할 필요가 있다. 하지만 수술 기구라는 특수한 도메인에서 데이터 획득의 어려움을 고려할 때, 본 방법론은 실용적인 관점에서 매우 효율적인 접근 방식이라고 판단된다.

## 📌 TL;DR

본 논문은 수술 기구의 정밀한 계수를 위해 bounding box와 소수의 key point만을 사용하는 weakly-supervised instance segmentation 모델인 **PWISeg**을 제안하였다. Projection Loss, Association Loss, Distribution Loss라는 세 가지 손실 함수를 통해 mask-level annotation 없이도 효과적인 분할 성능을 구현하였으며, 새롭게 구축한 수술 기구 데이터셋과 HOSPI-Tools 데이터셋 모두에서 기존 box-supervised 모델들보다 우수한 성능을 입증하였다. 이 연구는 데이터 라벨링 비용을 획기적으로 줄이면서도 수술 보조 시스템의 정확도를 높일 수 있는 가능성을 제시한다.
