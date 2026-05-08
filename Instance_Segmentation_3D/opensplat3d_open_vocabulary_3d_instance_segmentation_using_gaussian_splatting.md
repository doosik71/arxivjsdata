# OpenSplat3D: Open-Vocabulary 3D Instance Segmentation using Gaussian Splatting

Jens Piekenbrinck, Christian Schmidt, Alexander Hermans, Narunas Vaskevicius, Timm Linder, Bastian Leibe (2025)

## 🧩 Problem to Solve

전통적인 3D 인스턴스 분할(Instance Segmentation) 방법들은 수동으로 레이블링된 데이터셋에 크게 의존한다. 이러한 과정은 노동 집약적일 뿐만 아니라, 미리 정의된 객체 카테고리에만 국한된다는 근본적인 한계가 있다. 이를 해결하기 위해 고정된 레이블 없이 임의의 객체를 분할하고 식별할 수 있는 Open-Vocabulary 3D 장면 이해에 대한 수요가 증가하고 있다.

또한, 기존의 많은 3D 인스턴스 분할 방법들은 3D 포인트 클라우드(Point Cloud) 데이터셋에 의존하는데, 이는 특수한 기록 장비가 필요하여 데이터셋의 확장성을 제한한다. 포인트 클라우드는 희소성(Sparsity)으로 인해 특징(Feature) 커버리지가 낮고, 표면의 가려짐(Occlusion) 처리에 취약하며, 2D에서 3D로의 밀집되고 미분 가능한 매핑이 어렵다는 단점이 있다. 본 논문의 목표는 이러한 제약을 극복하여 수동 레이블링 없이도 자연어 설명을 통해 3D 장면 내 임의의 객체를 식별하고 분할할 수 있는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 3D Gaussian Splatting(3DGS)의 명시적 표현력과 미분 가능한 렌더링 능력을 확장하여, 2D 파운데이션 모델의 풍부한 시맨틱 정보를 3D 공간에 효율적으로 전이하는 것이다. 주요 기여 사항은 다음과 같다.

첫째, SAM(Segment Anything Model)의 2D 인스턴스 마스크와 Vision-Language Model(VLM)의 언어 임베딩을 결합하여, 수동 3D 주석 없이도 Zero-shot Open-Vocabulary 3D 인스턴스 분할을 가능하게 하는 OpenSplat3D 프레임워크를 제안한다.

둘째, 인스턴스 간의 명확한 구분을 위해 마진 기반의 대조 손실(Margin-based Contrastive Loss)을 도입하여, 서로 다른 인스턴스의 특징 임베딩이 일정 거리 이상으로 떨어지도록 강제한다.

셋째, 알파 컴포지팅(Alpha-compositing) 렌더링 과정에서 발생하는 특징 블렌딩(Feature Blending) 문제를 해결하기 위해 새로운 분산 정규화 손실(Variance Regularization Loss)을 제안한다. 이는 렌더링 광선(Ray)을 따라 특징 임베딩의 분산을 최소화하여 3D 표현의 일관성을 높인다.

마지막으로, 제안 방법은 기존의 OpenGaussian보다 훨씬 적은 반복 횟수로 수렴하며, 최적화 속도를 대폭 향상시켰다.

## 📎 Related Works

본 논문은 크게 세 가지 관련 연구 분야를 다룬다.

먼저, CLIP과 SAM 같은 2D 파운데이션 모델들은 제로샷 인식 및 분할 능력이 뛰어나지만, 이를 3D 도메인으로 전이하는 방법이 필요하다. 기존 연구들은 2D 특징을 3D 포인트 클라우드에 투영하는 방식을 취했으나, 앞서 언급한 포인트 클라우드의 희소성 문제로 인해 한계가 있었다.

둘째, NeRF 기반의 LERF나 OpenNeRF는 언어 임베딩을 3D 볼륨에 증류하여 텍스트 기반 쿼리를 가능하게 했다. 최근에는 3DGS를 활용한 LangSplat, Gaussian Grouping 등이 등장하여 더 빠른 렌더링과 해석 가능한 포인트 기반 구조를 제공하며 시맨틱 쿼리 및 인스턴스 그룹화를 수행하고 있다.

셋째, Open-Vocabulary 3D 인스턴스 분할 분야에서는 OpenMask3D나 Segment3D 등이 제안되었다. 특히 OpenGaussian은 인스턴스 레벨 특징과 언어 임베딩을 통합한 최근 연구이다. 본 논문의 OpenSplat3D는 OpenGaussian과 유사한 목적을 가지나, 부정이 쌍(Negative pairs)을 처리할 때 역거리 가중치 대신 마진 기반 손실을 사용하며, 이산적인 코드북(Codebook) 대신 연속적인 3D 공간에서의 분산 정규화를 통해 시맨틱 일관성을 유지한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

OpenSplat3D는 입력된 다시점 RGB 이미지들로부터 3D 지오메트리를 최적화함과 동시에, 각 가우시안 프리미티브에 인스턴스 임베딩을 추가하여 최적화한다. 이후 클러스터링을 통해 3D 인스턴스를 생성하고, VLM을 통해 각 인스턴스에 언어 특징을 할당하는 파이프라인으로 구성된다.

### 1. 인스턴스 학습 (Instance Learning)

각 가우시안 $n$에 뷰-독립적인 인스턴스 특징 임베딩 $f_n \in \mathbb{R}^d$를 추가한다. 렌더링된 특징 맵 $F(p)$는 다음과 같이 계산된다.
$$F(p) = \sum_{n=1}^{N} f_n \alpha_n \prod_{j=1}^{n-1} (1-\alpha_j)$$
여기서 $\alpha_n$은 가우시안의 기여도이다. 2D SAM 마스크 $M_{v,i}$를 사용하여 각 인스턴스의 프로토타입 특징 $z_i$를 계산하고, 이를 통해 두 가지 대조 손실을 적용한다.

- **Positive Loss ($L_{pos}$):** 동일 인스턴스 내의 픽셀 특징들을 프로토타입 $z_i$로 끌어당긴다.
$$L_{pos} = \frac{1}{N_v} \sum_{i=1}^{N_v} \frac{1}{|M_{v,i}|} \sum_{p \in M_{v,i}} \|F(p) - z_i\|_2^2$$
- **Negative Loss ($L_{neg}$):** 서로 다른 인스턴스의 프로토타입 간의 거리를 마진 $\gamma$ 이상으로 밀어낸다.
$$L_{neg} = \frac{2}{N_v(N_v-1)} \sum_{i=1}^{N_v} \sum_{j>i}^{N_v} \text{ReLU}(\gamma - \|z_i - z_j\|_2^2)$$
최종 인스턴스 손실은 $L_{inst2D} = w_p L_{pos} + w_n L_{neg}$로 정의되며, 최적화 후 HDBSCAN을 통해 가우시안들을 클러스터링하여 3D 인스턴스를 확정한다.

### 2. 언어 정렬 (Language Alignment)

각 가우시안마다 고차원 언어 임베딩을 최적화하는 것은 비효율적이므로, 인스턴스당 하나의 임베딩을 생성하는 전략을 취한다. 가시성 점수(Visibility Score) $s_{v,i}$를 계산하여 가장 정보량이 많은 상위 $K$개의 뷰를 선택한다.
$$s_{v,i} = \frac{|M_{v,i}|}{|I_v|} \cdot \frac{|G_{v,i}|}{|G_i|}$$
선택된 뷰에서 인스턴스 중심의 계층적 크롭(Hierarchical Crops)을 추출하고, 이를 MasQCLIP(VLM)의 이미지 인코더에 통과시켜 얻은 임베딩 $l_{i,k,l}$들의 평균을 최종 인스턴스 언어 임베딩 $l_i$로 사용한다.
$$l_i = \frac{1}{KL} \sum_{k=1}^{K} \sum_{l=1}^{L} l_{i,k,l}$$

### 3. 인스턴스 특징 정규화 (Instance Feature Regularization)

알파 컴포지팅 렌더링 시 여러 가우시안이 겹치면서 특징이 섞이는 문제가 발생한다. 이를 해결하기 위해 렌더링 광선을 따라 특징 임베딩의 분산을 최소화하는 분산 정규화 손실 $L_{var}$를 도입한다. 픽셀 $p$에서의 분산 $\text{Var}(F(p))$는 다음과 같이 계산된다.
$$\text{Var}(F(p)) = \left( \sum_{n=1}^{N} f_n^2 \alpha_n \prod_{j=1}^{n-1} (1-\alpha_j) \right) - F(p)^2$$
최종 손실 함수는 RGB 재구성 손실, 인스턴스 손실, 분산 정규화 손실의 합으로 정의된다.
$$L = L_{RGB} + \lambda_{inst2d} L_{inst2d} + \lambda_{var} L_{var}$$

## 📊 Results

### 실험 설정

- **데이터셋:** LERF-mask, LERF-OVS (Open-vocabulary 평가), ScanNet++ v1 (3D 인스턴스 분할 평가).
- **지표:** mIoU, mBIoU, mAcc (LERF), AP, $AP_{50}$, $AP_{25}$ (ScanNet++).
- **환경:** RTX 3090 GPU, 인스턴스 최적화에 약 20~45분 소요.

### 주요 결과

1. **Open-Vocabulary Segmentation:** LERF-mask 데이터셋에서 OpenSplat3D는 대부분의 장면에서 타 방법론(Gaussian Grouping, CGC 등)보다 우수한 mIoU 및 mBIoU 성능을 보였다. LERF-OVS의 3D 객체 선택 작업에서도 OpenGaussian을 포함한 기존 방법들을 크게 앞서며 가장 높은 정확도를 달성했다.
2. **3D Instance Segmentation:** ScanNet++ 검증 셋에서 클래스 불가지론적(Class-agnostic) 분할 성능을 측정한 결과, Segment3D보다 월등한 성능을 보였으며, 특히 후처리에 대한 의존도가 낮아졌음을 확인했다.
3. **Open-Vocabulary 성능:** ScanNet++에서 Open-vocabulary 설정으로 평가했을 때, Segment3D 대비 $AP_{50}$ 지표에서 11.2 포인트 높은 성능을 기록하며, 완전 지도 학습 기반의 SGIFormer와의 격차를 상당히 좁혔다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 3DGS의 효율적인 렌더링 파이프라인에 분산 정규화 손실을 결합하여, 2D-3D 간의 특징 일관성을 성공적으로 확보했다는 점이다. 특히 Ablation Study를 통해 $\lambda_{var}$가 매우 작은 값일 때조차 분산 손실이 없을 때보다 성능이 비약적으로 향상됨을 보여주었으며, 이는 렌더링 과정에서의 특징 블렌딩 현상이 3D 인스턴스 학습에 심각한 저해 요인임을 입증한다.

다만, 한계점으로는 SAM 모델 자체가 장면을 과도하게 분할(Over-segmentation)하는 경향이 있어, 이로 인해 완전히 지도 학습된 모델(SGIFormer)과의 성능 격차가 여전히 존재한다는 점이 언급된다. 또한, 현재는 정적인 장면에 최적화되어 있어 동적인 환경으로의 확장이 향후 과제로 남아 있다.

비판적으로 해석하자면, 제안된 방법은 VLM의 고정된 임베딩을 사후에 할당하는 방식이므로, 3D 공간에서의 시맨틱 특징을 직접 최적화하는 방식에 비해 유연성이 떨어질 수 있다. 하지만 이는 계산 효율성을 얻기 위한 트레이드오프이며, 실제 결과에서 충분한 성능 향상을 입증하였다.

## 📌 TL;DR

OpenSplat3D는 3D Gaussian Splatting을 기반으로 하여, SAM의 인스턴스 마스크와 VLM의 언어 임베딩을 통합한 Zero-shot Open-Vocabulary 3D 인스턴스 분할 프레임워크이다. 특히 렌더링 시 발생하는 특징 섞임 문제를 해결하는 분산 정규화 손실(Variance Regularization Loss)과 마진 기반 대조 손실을 도입하여, 기존 방법론 대비 빠른 수렴 속도와 높은 분할 정확도를 달성하였다. 이 연구는 별도의 3D 수동 레이블링 없이도 복잡한 실내외 장면에서 임의의 객체를 텍스트 쿼리로 정밀하게 분할할 수 있음을 보여주며, 향후 로보틱스나 AR/VR 분야의 3D 장면 이해에 중요한 역할을 할 것으로 기대된다.
