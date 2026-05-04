# Zero-shot Unsupervised Transfer Instance Segmentation

Gyungin Shin, Samuel Albanie, Weidi Xie (2023)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전의 핵심 과제인 세그멘테이션(Segmentation)에서 발생하는 고비용의 픽셀 수준 어노테이션(Pixel-level annotation) 문제를 해결하고자 한다. 기존의 세그멘테이션 모델들은 대규모의 정밀한 데이터셋을 필요로 하며, 이는 현실적으로 막대한 비용과 시간을 소모하게 만든다.

특히, 최근 언어-이미지 사전학습(Language-Image Pretraining)을 활용한 비지도 시맨틱 세그멘테이션(Unsupervised Semantic Segmentation, USSLIP) 연구들이 성과를 거두고 있으나, 이들은 동일한 시맨틱 카테고리 내에서 개별 객체를 구분하는 **인스턴스 세그멘테이션(Instance Segmentation)** 능력이 결여되어 있다는 한계가 있다.

따라서 본 연구의 목표는 수동 어노테이션이나 타겟 데이터 분포에 대한 접근 없이도 이미지 내의 인스턴스를 분할하고 그 시맨틱 클래스를 추론할 수 있는 **Zero-shot Unsupervised Transfer Instance Segmentation (ZUTIS)** 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

ZUTIS의 핵심 아이디어는 사전학습된 시각-언어 모델(VLM, 예: CLIP)의 제로샷 전이 능력과 쿼리 기반의 트랜스포머 디코더(Query-based Transformer Decoder)를 결합하여, 별도의 정답 라벨 없이도 시맨틱 세그멘테이션과 인스턴스 세그멘테이션을 동시에 수행하는 통합 프레임워크를 구축하는 것이다.

주요 기여 사항은 다음과 같다:

1. **새로운 태스크 정의**: 인간의 감독이나 타겟 데이터 분포 없이 객체 인스턴스를 분할하는 'Zero-shot Unsupervised Transfer Instance Segmentation'이라는 도전적인 과제를 제시하였다.
2. **통합 프레임워크 제안**: 기존 USSLIP 방식을 확장하여 시맨틱 세그멘테이션뿐만 아니라 인스턴스 세그멘테이션을 동시에 가능하게 하는 ZUTIS를 제안하였다.
3. **성능 입증**: COCO-20K 및 ImageNet-S와 같은 표준 비지도 세그멘테이션 벤치마크에서 기존 SOTA 방식 대비 유의미한 성능 향상을 달성하였다.

## 📎 Related Works

논문은 다음과 같은 관련 연구들을 분석하고 ZUTIS와의 차별점을 제시한다.

- **Zero-shot Semantic/Instance Segmentation**: 주로 학습 시 본 적 없는 클래스로 일반화하는 것을 목표로 하며, 기존 방식들은 학습 단계에서 일부 클래스에 대한 픽셀 수준의 정답 라벨이 필요하다. 반면, ZUTIS는 학습 과정에서 어떠한 수동 어노테이션도 사용하지 않는다.
- **Unsupervised Semantic Segmentation**: 프록시 태스크를 통해 학습하며, 추후 예측 결과와 클래스 이름을 매칭하기 위해 헝가리안 매칭(Hungarian matching)과 같은 추가 단계가 필요하다. ZUTIS는 텍스트 인코더를 직접 활용하므로 이러한 매칭 단계가 필요 없다.
- **USSLIP (Unsupervised Semantic Segmentation with Language-Image Pretraining)**: CLIP과 같은 VLM을 활용해 클래스 이름을 부여한다. 하지만 기존 USSLIP 방식들은 인스턴스 구분 능력이 없으며, 새로운 카테고리마다 모델을 새로 학습시켜야 하는 경직성이 있다.
- **Class-agnostic Unsupervised Instance Segmentation**: FreeSOLO와 같은 연구들은 클래스 구분 없이 객체만 찾아내지만, ZUTIS는 VLM을 결합하여 클래스 인지(Class-aware) 인스턴스 세그멘테이션을 수행한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 문제 정의

ZUTIS는 이미지 인코더 $\Phi_{enc}^{I}$, 이미지 디코더 $\Phi_{dec}^{I}$, 그리고 텍스트 인코더 $\Phi^{T}$로 구성된다. 입력 이미지 $x$와 개념 집합 $C$에 대해 다음과 같이 출력한다:

$$
\Phi_{seg}(x, C) =
\begin{cases}
\Phi^{T}(C)W\Phi_{enc}^{I}(x) \in \{0,1\}^{|C|\times H\times W} & \text{(Semantic Segmentation)} \\
\Phi_{dec}^{I} \circ \Phi_{enc}^{I}(x) \in \{0,1\}^{n\times H\times W} & \text{(Instance Segmentation)}
\end{cases}
$$

여기서 $W$는 이미지 특징을 텍스트 임베딩 공간으로 투영하는 행렬이며, $n$은 예측할 마스크 제안(mask proposals)의 수이다.

### 2. 의사 라벨(Pseudo-label) 생성 과정

수동 라벨 없이 학습하기 위해 다음과 같은 절차로 의사 라벨을 생성한다.

- **아카이브 구축 (Archive Construction)**: CLIP 모델을 사용하여 텍스트 임베딩과 유사도가 높은 이미지들을 인덱스 데이터셋에서 검색하여 카테고리별 이미지 아카이브를 구축한다.
- **비지도 돌출 객체 탐지 (Unsupervised Saliency Detection)**: SelfMask와 같은 비지도 탐지기를 사용하여 검색된 이미지들에서 카테고리 불가지론적인(category-agnostic) 돌출 마스크 $S_i$를 생성한다.
- **Copy-Paste 증강**: 단일 객체 이미지들을 복사-붙여넣기 하여 여러 객체가 포함된 합성 이미지를 생성함으로써, 모델이 다중 객체를 분할하는 능력을 갖추게 한다.

### 3. 상세 아키텍처 및 학습 절차

#### 시맨틱 세그멘테이션 (Semantic Segmentation)

- 이미지 인코더에서 추출된 Dense Feature $\psi^{I}(x_i)$를 투영 행렬 $W$를 통해 텍스트 공간으로 보낸다.
- 고정된(frozen) 텍스트 인코더의 임베딩 $\psi^{T}(C)$와 내적(dot-product)을 수행하고 Softmax를 적용하여 확률 맵 $P_i$를 생성한다.
- 손실 함수로는 의사 마스크와의 차이를 줄이는 교차 엔트로피 손실 $\mathcal{L}_{ce}$를 사용한다.

#### 인스턴스 세그멘테이션 (Instance Segmentation)

- **마스크 제안**: Dense Feature를 FFN(Feed-Forward Network)에 통과시킨 후, 쿼리 기반 트랜스포머 디코더를 통해 $n_q$개의 마스크 제안 $M$을 생성한다.
- **학습 목표**: 생성된 제안과 의사 마스크 간의 이분 매칭 손실(Bipartite matching loss) $\mathcal{L}_{mask}$를 사용하며, 이는 Dice coefficient와 Binary Cross-Entropy의 합으로 구성된다.
- **추론 단계**: 각 마스크 영역의 평균 이미지 임베딩을 계산하고, 이를 텍스트 임베딩과 내적하여 가장 유사도가 높은 클래스를 할당한다.

#### 전체 손실 함수 및 학습 제약

전체 손실 함수는 다음과 같다:
$$\mathcal{L} = \mathcal{L}_{ce} + \lambda_{mask}\mathcal{L}_{mask}$$
특히, **트랜스포머 디코더에서 이미지 인코더로 흐르는 그래디언트를 차단(Stop-gradient)**하는 것이 매우 중요하다. 이를 통해 인스턴스 분할을 위한 학습이 VLM의 시맨틱 정렬(semantic alignment)을 해치는 것을 방지한다.

## 📊 Results

### 실험 설정

- **데이터셋**: COCO-2017 val, PASCAL VOC2012, CoCA, ImageNet-S, CUB-200-2011.
- **지표**: 시맨틱 세그멘테이션은 mIoU, 인스턴스 세그멘테이션은 COCO 스타일의 Mask AP ($AP^{mk}$)를 사용한다.
- **제로샷 전이 설정**: 타겟 벤치마크의 학습 데이터에 접근하지 않고, ImageNet1K 및 PASS 데이터셋에서 검색된 이미지로만 학습한다.

### 주요 결과

- **인스턴스 세그멘테이션**: COCO-20K 데이터셋에서 ZUTIS(ViT-B/16)는 기존 비지도 방식 및 MaskCLIP 대비 우수한 성능을 보였으며, 특히 MaskCLIP 대비 $AP^{mk}_{50}$ 기준 상당한 향상을 보였다.
- **시맨틱 세그멘테이션**: COCO 및 CoCA 벤치마크에서 MaskCLIP 대비 mIoU가 각각 12.2, 12.5 포인트 향상되었다. ImageNet-S(919개 클래스)에서는 SOTA 비지도 모델인 NamedMask 대비 최대 14.5 mIoU의 이득을 얻었다.
- **새로운 카테고리 일반화**:
  - **계층적 전이**: '새(bird)'라는 상위 개념으로 학습한 후, 학습 시 보지 못한 200종의 세부 새 품종(CUB-200-2011)에 대해 성공적으로 세그멘테이션을 수행하였다.
  - **완전 미학습 클래스**: COCO 데이터셋의 15개 미학습 클래스에 대해서도 MaskCLIP 대비 높은 $AP^{mk}_{50}$를 기록하여 강력한 제로샷 능력을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **Stop-gradient의 중요성**: 실험을 통해 디코더의 그래디언트가 인코더로 흐를 경우, 모델이 객체 분할 능력은 갖게 되나 클래스 분류 능력이 급격히 저하됨을 확인하였다. 이는 인스턴스 손실 함수가 VLM의 시각-언어 정렬을 파괴하기 때문이며, 이를 차단함으로써 두 태스크의 균형을 맞출 수 있었다.
- **효율적인 통합**: 하나의 아키텍처로 시맨틱과 인스턴스 세그멘테이션을 동시에 수행하며, 텍스트 인코더를 분류기로 사용함으로써 Open-vocabulary 확장이 용이하다.

### 한계 및 비판적 해석

- **VLM 의존성**: CLIP의 사전학습 데이터에 존재하지 않는 매우 희귀한 개념은 세그멘테이션할 수 없다는 근본적인 한계가 있다.
- **의사 라벨의 노이즈**: VLM을 통한 이미지 검색 시, 타겟 객체와 항상 함께 등장하는 주변 객체(예: 스케이트보드와 그것을 타는 사람)가 함께 검색되어 의사 마스크에 노이즈가 섞일 가능성이 크다. 이는 추론 시 오탐(False Positive)의 원인이 된다.

## 📌 TL;DR

ZUTIS는 픽셀 수준의 정답 라벨이나 타겟 데이터셋 없이도 작동하는 **최초의 제로샷 비지도 통합 세그멘테이션 프레임워크**이다. CLIP의 텍스트-이미지 정렬 능력과 트랜스포머 디코더의 마스크 생성 능력을 결합하였으며, 특히 Stop-gradient 기법을 통해 시맨틱 정렬을 보존하며 인스턴스 분할 능력을 획득하였다. 이 연구는 어노테이션 비용이 매우 높은 의료 영상이나 특수 산업 분야의 세그멘테이션 자동화에 중요한 가능성을 제시한다.
