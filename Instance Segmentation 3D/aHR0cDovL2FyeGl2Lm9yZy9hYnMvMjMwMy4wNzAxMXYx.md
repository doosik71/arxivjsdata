# OSIS: Efficient One-stage Network for 3D Instance Segmentation

Chuan Tang, Xi Yang (2023)

## 🧩 Problem to Solve

본 논문은 3D 포인트 클라우드 데이터에서 개별 객체를 분리해내는 3D Instance Segmentation의 효율성 문제를 해결하고자 한다. 기존의 3D 인스턴스 분할 모델들은 일반적으로 다음과 같은 다단계(multi-stage) 방식을 사용한다:

1. **Proposal-based 방법**: 3D 객체 검출기를 통해 영역을 먼저 제안한 뒤 마스크를 예측한다. 이 방식은 무거운 검출 프로세스에 의존하여 추론 속도가 느리다.
2. **Cluster-based 방법**: 포인트들을 클러스터링하여 인스턴스를 추출한다. 이 방식은 클러스터링 반경과 같은 하이퍼파라미터 설정에 매우 민감하며, 수동으로 설계된(hand-crafted) 후처리 과정에 의존한다. 또한, 여러 개의 연결되지 않은 부분으로 구성된 객체에 대해 과분할(over-segmentation) 문제가 발생하기 쉽다.

결과적으로 이러한 다단계 접근 방식은 추론 속도를 제한하며, 실시간 시나리오의 요구 사항을 충족하지 못한다. 따라서 본 논문의 목표는 복잡한 후처리 과정 없이 네트워크에서 직접 인스턴스를 분할하는 효율적인 One-stage 네트워크인 OSIS를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Instance Decoder**와 **Bipartite Matching**을 통해 3D 인스턴스 분할을 단일 단계(One-stage)로 처리하는 것이다.

- **One-stage 아키텍처**: 클러스터링이나 Non-Maximum Suppression (NMS)과 같은 계산 비용이 큰 후처리 단계 없이, 신경망의 출력에서 직접 인스턴스 마스크를 생성한다.
- **Dynamic Convolution 기반의 인스턴스 디코더**: 포인트별 특징(point-wise features)으로부터 인스턴스 특징을 추출하고, 이를 다시 동적 컨볼루션의 커널 파라미터로 사용하여 정밀한 인스턴스 마스크를 생성하는 구조를 제안한다.
- **Bipartite Matching 학습**: 학습 단계에서 예측된 후보 인스턴스와 Ground Truth(GT) 간의 일대일 매핑을 형성함으로써, 추론 시 NMS 없이도 중복 예측을 효과적으로 제거할 수 있도록 한다.

## 📎 Related Works

### 3D Instance Segmentation
기존 연구는 크게 Proposal-based와 Cluster-based로 나뉜다. PointGroup, SoftGroup, HAIS 등은 클러스터링 전략을 통해 성능을 높였으나, 하이퍼파라미터 의존성과 느린 추론 속도가 한계로 지적되었다. 특히 OccuSeg나 SSTNet은 Super-voxel 전처리를 사용하지만, 이 과정 자체가 매우 많은 시간을 소모한다.

### Dynamic Convolution
SOLO v2, CondInst 등의 2D 인스턴스 분할 모델들은 마스크 예측을 동적 커널과 컨볼루션 특징으로 분리하여 빠르게 처리했다. 3D 분야에서는 DyCo3D가 동적 컨볼루션을 처음 도입했으나, 여전히 클러스터링 과정에 의존하여 완전한 One-stage 추론을 달성하지 못했다.

### Bipartite Matching
객체 검출 및 분할 모델에서 예측값과 GT를 매칭하기 위해 Hungarian 알고리즘과 같은 이분 매칭(Bipartite Matching)을 사용하는 추세이다. 이는 추론 시 중복 예측을 줄이는 효과가 있으며, 본 논문은 이를 3D 인스턴스 분할 학습에 적용하여 NMS 의존성을 제거했다.

## 🛠️ Methodology

### 전체 파이프라인
OSIS는 크게 **Point-wise Feature Extraction** 모듈과 **Instance Decoder** 모듈로 구성된다.

### 1. Point-wise Feature Extraction
입력 포인트 클라우드를 Voxel 형태로 변환한 뒤, Sparse Convolution 기반의 U-Net 백본을 통해 특징을 추출한다. 이후 다음의 네 가지 병렬 브랜치로 출력된다:
- **Offset branch**: 각 포인트에서 해당 인스턴스 중심점까지의 상대적 거리 $o_i$를 회귀한다.
- **Feature branch**: 포인트별 특징 $f_i^{origin}$을 생성한다.
- **Mask branch**: $k$개의 초기 인스턴스 마스크 $m_j^{origin}$을 생성한다.
- **Semantic branch**: 포인트별 시맨틱 예측 $S$를 수행하여 백본의 인지 능력을 돕는다.

이때, 중심점 정보에 대한 인식을 높이기 위해 Positional Encoding(PE)을 사용하여 특징을 융합한다:
$$f_i = f_i^{origin} + PE(p_i + o_i)$$

### 2. Instance Decoder
인스턴스 디코더는 **Mask-to-Feature**와 **Feature-to-Instance** 단계로 이루어진다.

- **Mask-to-Feature**: 초기 마스크 $m_j^{origin}$에 Sigmoid 활성화 함수와 임계값 $\tau$를 적용하여 $\hat{m}_j$를 얻고, 이를 가중치로 사용하여 포인트별 특징 $f_i$로부터 인스턴스 특징 $f_j^{ins}$를 추출한다.
$$\hat{m}_j = (1-[\sigma(m_j^{origin}) < \tau]) \cdot \sigma(m_j^{origin})$$
$$f_j^{ins} = \frac{\sum_{i=1}^{N} \hat{m}_j^T \cdot f_i}{N}$$
- **Feature-to-Instance**: 추출된 $f_j^{ins}$를 $1 \times 1$ Dynamic Convolution의 커널 파라미터로 사용하여 최종 인스턴스 마스크 $m_j$를 생성한다. 또한, Max pooling을 통한 전역 특징 융합 후 선형 레이어를 통해 인스턴스의 시맨틱 카테고리를 예측한다.

### 3. Bipartite Matching 및 학습
학습 시 예측된 후보 인스턴스와 GT 인스턴스 간의 유사도 $Q(i, j)$를 계산하여 Hungarian 알고리즘으로 일대일 매칭을 수행한다.
$$Q(i, j) = \hat{s}_{i,j} + \alpha \text{Dice}(m_i, m_{j}^{gt})$$
여기서 $\text{Dice}$ 계수는 다음과 같이 정의된다:
$$\text{Dice}(m_i, m_j^{gt}) = \frac{2m_i^T \cdot m_j^{gt}}{m_i^T \cdot m_i + m_j^{gt} \cdot m_j^{gt}}$$

전체 손실 함수는 마스크 손실(Dice + Focal loss), 시맨틱 분류 손실, 그리고 포인트별 손실(중심점 오프셋 및 시맨틱 손실)의 합으로 구성된다:
$$L = L_{mask} + L_{cls} + L_{point}$$

### 추론 절차
Voxel화된 데이터를 네트워크에 입력하여 후보 인스턴스들을 얻는다. 배경 클래스와 낮은 신뢰도의 인스턴스를 제거하는 것만으로 최종 결과를 도출하며, NMS나 클러스터링 같은 복잡한 후처리를 완전히 생략한다.

## 📊 Results

### 실험 설정
- **데이터셋**: ScanNet v2 (실내 장면 데이터셋)
- **지표**: Average Precision (AP), $AP_{50}$
- **구현**: PyTorch, spconv 라이브러리 사용, Adam 옵티마이저 적용

### 정량적 결과
- **추론 속도**: OSIS의 평균 추론 속도는 장면당 **138ms**로, 기존 SOTA 모델인 SoftGroup 대비 약 150% 이상의 속도 향상을 보였다. (Table I 참조)
- **분할 정확도**: ScanNet v2 테스트 셋에서 3D-MPA 대비 AP가 각각 검증셋과 테스트셋에서 4.5, 3.7 증가하였다. 특히 대부분의 카테고리에서 경쟁력 있는 성능을 보여주었다. (Table II, III 참조)

### 정성적 결과 및 분석
- **과분할 및 과소분할 해결**: SoftGroup과 같은 클러스터링 기반 방식은 커튼이나 긴 책상처럼 연결되지 않은 부분들로 구성된 객체를 여러 개로 쪼개어 예측(과분할)하는 경향이 있다. 반면, OSIS는 이를 하나의 인스턴스로 정확하게 묶어내는 능력이 뛰어남을 확인하였다. (Fig 3 참조)
- **NMS 영향도**: Ablation Study 결과, NMS를 적용했을 때와 적용하지 않았을 때의 성능 차이가 거의 없음을 확인하여, Bipartite Matching을 통한 학습이 중복 예측 제거에 효과적임을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 3D 인스턴스 분할에서 고질적인 문제였던 '추론 속도'와 '후처리 의존성'을 One-stage 구조로 해결하였다. 특히 Dynamic Convolution을 이용해 포인트 특징에서 마스크를 직접 생성하는 방식은 클러스터링의 한계인 '비연결 객체 분할' 문제를 효과적으로 극복했다.

다만, 논문에서 제시된 성능 수치가 일부 최신 모델(SSTNet, SoftGroup 등)의 AP보다 약간 낮게 나타나는 지점이 있으나, 이는 추론 속도의 압도적인 이득(138ms)과 트레이드-오프 관계에 있다고 볼 수 있다. 또한, 특정 하이퍼파라미터 $\alpha$나 임계값 $\tau$에 대한 민감도 분석이 더 상세히 제공되었다면 모델의 일반화 능력을 더 잘 평가할 수 있었을 것이다.

## 📌 TL;DR

OSIS는 복잡한 클러스터링과 NMS 후처리를 제거하고, Dynamic Convolution과 Bipartite Matching을 통해 인스턴스를 직접 예측하는 **One-stage 3D 인스턴스 분할 네트워크**이다. 장면당 **138ms**라는 매우 빠른 추론 속도를 달성하면서도, 특히 비연결 객체에 대해 기존 방식보다 우수한 분할 성능을 보여 실시간 3D 장면 이해를 위한 중요한 베이스라인을 제시하였다.