# Instance-aware Self-supervised Learning for Nuclei Segmentation

Xinpeng Xie, Jiawei Chen, Yuexiang Li, Linlin Shen, Kai Ma, and Yefeng Zheng (2020)

## 🧩 Problem to Solve

본 논문은 계산 병리(Computational Pathology) 분야에서 매우 도전적인 과제 중 하나인 핵 인스턴스 분할(Nuclei Instance Segmentation) 문제를 다룬다. 세포핵은 이미지 내에 광범위하게 존재하며 형태적 변동성(Morphological Variances)이 매우 크기 때문에 정확한 분할이 어렵다.

특히, 핵 인스턴스 분할을 위한 학습 데이터를 구축하기 위해서는 숙련된 병리학자가 직접 세포핵의 윤곽선을 그려야 하므로, 어노테이션 과정에 막대한 비용과 시간이 소요된다. 이로 인해 고품질의 레이블링된 데이터가 부족한 현상이 발생하며, 이는 대량의 데이터를 필요로 하는 딥러닝 기반 분할 모델들의 성능 발휘를 저해하는 핵심적인 병목 현상이 된다.

따라서 본 연구의 목표는 레이블이 없는 원시 데이터(Raw data)를 최대한 활용하여 신경망이 스스로 인스턴스에 대한 특징을 학습하게 함으로써, 적은 양의 어노테이션 데이터만으로도 높은 성능의 핵 인스턴스 분할을 달성하는 Self-supervised Learning(SSL) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 신경망이 세포핵의 크기(Size)와 수량(Quantity)이라는 사전 지식(Prior-knowledge)을 암시적으로 학습하도록 유도하는 **Instance-aware Self-supervised Learning** 프레임워크를 설계한 것이다. 

이를 위해 저자들은 두 가지 보조 작업(Proxy tasks)인 **Scale-wise Triplet Learning**과 **Count Ranking**을 제안하였다. 이는 기존의 일반적인 SSL 방식들이 단순한 이미지 변형이나 전역적 특징에 집중했던 것과 달리, 인스턴스 분할의 핵심인 '개별 객체의 특성'을 학습하도록 설계되었다는 점에서 차별성을 가진다.

## 📎 Related Works

기존의 핵 분할 연구들은 Mask R-CNN을 이용한 직접적인 국소화 및 분할 방식이나, BESNet과 같이 세포 경계와 전체 세포를 각각 분할하는 듀얼 디코더 구조, 그리고 CIA-Net과 같은 다단계 정보 집계 모듈을 통해 정확도를 높이려 하였다.

또한, 의료 영상 분야에서 뇌 영역 분할이나 장기 분할 등을 위해 SSL이 적용된 사례가 있었으나, 대부분은 시맨틱 분할(Semantic Segmentation)에 국한되어 있었다. 인스턴스 분할은 픽셀이 어떤 카테고리에 속하는지뿐만 아니라 어떤 개별 객체(Instance)에 속하는지를 구분해야 하므로, 기존의 시맨틱 기반 SSL 방식으로는 인스턴스 구분을 위한 특징을 학습하기에 한계가 있었다.

## 🛠️ Methodology

### 1. 이미지 조작 (Image Manipulation)
신경망이 핵의 크기와 수량을 학습할 수 있도록 다음과 같이 트리플렛(Triplet) 샘플을 생성한다.
- **Anchor ($A$):** 원본 이미지에서 $768 \times 768$ 크기로 크롭한 패치이다.
- **Positive ($P$):** Anchor와 인접한 영역에서 동일한 크기($768 \times 768$)로 크롭한 패치로, Anchor와 유사한 크기의 핵들을 포함한다.
- **Negative ($N$):** Positive 샘플 내에서 무작위로 작은 영역($512 \times 512$ ~ $64 \times 64$)을 크롭한 후, 이를 다시 $768 \times 768$ 크기로 확대(Resize)한 샘플이다. 이 과정에서 Negative 샘플 내의 핵들은 Anchor/Positive보다 상대적으로 더 크게 나타나게 된다.

### 2. Self-supervised Proxy Tasks
세 개의 공유 가중치 인코더(Shared-weight Encoders)를 통해 샘플들을 128차원의 잠재 특징 공간(Latent Feature Space) $Z$로 임베딩한다. $E_A: A \to z_a, E_P: P \to z_p, E_N: N \to z_n$으로 정의한다.

#### Proxy Task 1: Scale-wise Triplet Learning
핵의 크기 정보를 학습하기 위해 Triplet Loss를 사용한다. 동일한 스케일의 샘플은 가깝게, 다른 스케일(확대된 샘플)은 멀게 배치한다.
$$L_{ST}(z_a, z_p, z_n) = \sum \max(0, d(z_a, z_p) - d(z_a, z_n) + m_1)$$
여기서 $d(\cdot)$는 제곱된 $L_2$ 거리이며, $m_1$은 마진(1.0)이다.

#### Proxy Task 2: Count Ranking
Positive 샘플은 Negative 샘플(Positive의 부분 집합)보다 항상 더 많은 수의 핵을 포함한다는 점을 이용한다. 특징 벡터 $z$를 스칼라 값으로 변환하는 매핑 함수 $f$(Fully Convolutional Layer)를 도입하여 수량의 상대적 순위를 학습시킨다.
$$L_{CR} = \sum \max(0, f(z_n) - f(z_p) + m_2)$$
여기서 $m_2$는 마진(1.0)이며, 이 손실 함수는 $f(z_p)$가 $f(z_n)$보다 크게 만들어 네트워크가 핵의 밀도와 수량을 인식하게 한다.

#### 전체 목적 함수 및 학습 절차
전체 SSL 손실 함수는 다음과 같다.
$$L = L_{ST} + L_{CR}$$
학습은 크게 두 단계로 진행된다.
1. **Pre-training:** 위 목적 함수를 사용하여 ResNet-101 인코더를 사전 학습시킨다.
2. **Fine-tuning:** 사전 학습된 인코더를 U-shape 구조의 **ResUNet-101** 프레임워크에 통합하고, 랜덤 초기화된 디코더와 함께 타겟 작업(핵 body, boundary, background의 3클래스 분할)을 수행하도록 미세 조정한다.

## 📊 Results

### 실험 설정
- **데이터셋:** MoNuSeg 2018 (7개 장기 이미지, 학습셋 30장, 테스트셋 14장) 및 CPM 데이터셋.
- **평가 지표:** Aggregated Jaccard Index (AJI) 및 Dice score. AJI는 객체 수준의 분할 성능을 측정하는 데 더 적합한 지표이다.
- **비교 대상:** Train-from-scratch, ImageNet Pre-trained, 그리고 기존 SSL 방식(Jigsaw Puzzles, RotNet, ColorMe).

### 주요 결과
- **SOTA 달성:** 제안된 SSL 기반 ResUNet-101은 MoNuSeg 테스트셋에서 **70.63%의 AJI**를 기록하며 새로운 State-of-the-art를 달성하였다. 이는 기존 1위 모델인 CIA-Net(69.07%)보다 높은 수치이다.
- **데이터 효율성:** 레이블 데이터의 양을 줄였을 때 성능 향상 폭이 더 두드러졌다. 특히 데이터의 10%만 사용했을 때, Train-from-scratch 대비 AJI가 **11.43%** 향상되었다.
- **SSL 방식 간 비교:** 일반적인 SSL(RotNet 등)보다 높은 성능을 보였는데, 이는 인스턴스 특성(크기, 수량)을 직접적으로 학습했기 때문으로 분석된다.
- **Ablation Study:** $L_{ST}$ 단독(+4.35%)이나 $L_{CR}$ 단독(+4.80%)보다 두 손실 함수를 함께 사용했을 때(+5.34%) 가장 높은 성능 향상을 보였다.
- **외부 검증:** CPM 데이터셋에서도 평균 Dice score 86.36%, AJI 76.19%를 기록하며 강건함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 단순한 데이터 증강이나 일반적인 프리트레이닝보다, **도메인 특화된 사전 지식(Domain-specific Prior)**을 SSL 설계에 반영하는 것이 훨씬 효과적임을 보여주었다. 특히 ImageNet으로 사전 학습된 모델이 의료 영상에서 미미한 성능 향상(+0.54%)만을 보인 것은 자연 영상과 의료 영상 사이의 거대한 도메인 간극(Domain Gap)을 시사한다.

또한, 제안된 방법론은 복잡한 듀얼 디코더나 DenseNet 기반의 무거운 구조를 사용하는 CIA-Net보다 모델 파라미터 수가 적고 계산 복잡도가 낮음에도 불구하고 더 높은 성능을 냈다는 점에서 효율성이 매우 높다.

다만, 본 논문에서 제안한 이미지 조작 방식(크롭 및 확대)이 모든 종류의 병리 이미지에서 보편적으로 적용 가능한지에 대한 일반화 가능성 논의는 부족하며, 다른 세포 종류나 다른 염색법이 적용된 이미지에서도 동일한 효과가 나타날지에 대한 추가 검증이 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 레이블 부족 문제를 해결하기 위해 핵의 크기와 수량 정보를 학습하는 **Instance-aware SSL** 프레임워크를 제안하였다. Scale-wise Triplet Learning과 Count Ranking이라는 두 가지 보조 작업을 통해 인코더를 사전 학습시켰으며, 이를 통해 MoNuSeg 데이터셋에서 **70.63% AJI**라는 새로운 SOTA 성능을 달성하였다. 이 연구는 의료 영상 인스턴스 분할에서 도메인 특화 SSL이 데이터 효율성과 정확도를 획기적으로 높일 수 있음을 증명하였으며, 향후 데이터 획득 비용이 높은 다른 의료 영상 분석 작업으로 확장될 가능성이 크다.