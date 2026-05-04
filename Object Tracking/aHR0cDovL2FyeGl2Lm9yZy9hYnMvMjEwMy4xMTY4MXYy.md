# Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking

Ning Wang, Wengang Zhou, Jie Wang, Houqiang Li (2021)

## 🧩 Problem to Solve

비디오 객체 추적(Visual Object Tracking)에서 연속된 프레임 사이에는 풍부한 시간적 문맥(Temporal Context)이 존재하지만, 기존의 많은 추적기들은 이를 충분히 활용하지 못하고 있다. 대다수의 추적 패러다임은 추적 작업을 프레임별 객체 검출(Per-frame object detection) 문제로 취급하며, 이로 인해 프레임 간의 시간적 관계가 무시되는 경향이 있다.

특히 Siamese 추적기의 경우 초기 템플릿만을 사용하거나, 단순한 모션 프라이어(Motion prior)만을 가정한다. 모델 업데이트 메커니즘이 있는 경우에도 각 프레임을 상호 추론이 없는 독립적인 개체로 취급한다. 이러한 접근 방식은 가려짐(Occlusion), 블러(Blur) 등 노이즈가 포함된 프레임이 템플릿으로 사용될 때 모델 업데이트를 방해하고, 검색 프레임으로 사용될 때 추적 성능을 저하시키는 문제를 야기한다. 따라서 본 논문의 목표는 Transformer 아키텍처를 도입하여 고립된 비디오 프레임들을 연결하고, 시간적 문맥을 효과적으로 전파함으로써 강건한(Robust) 객체 추적을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 인코더와 디코더를 Siamese 형태의 추적 파이프라인 내에서 두 개의 병렬 브랜치로 분리하여 배치하는 것이다.

1. **Transformer Encoder를 통한 템플릿 강화**: 여러 템플릿 특징들 간의 Self-attention을 통해 상호 보완적인 표현을 학습함으로써, 더 정교하고 고품질의 추적 모델을 생성한다.
2. **Transformer Decoder를 통한 문맥 전파**: 이전 템플릿들로부터 얻은 특징(Feature)과 공간적 마스크(Spatial Mask)를 현재 검색 영역(Search patch)으로 전파하여, 객체 탐색 과정을 용이하게 만든다.
3. **범용적 통합**: 제안한 Transformer 구조를 단순한 Siamese 매칭 방식(TrSiam)과 정교한 판별적 상관 필터(Discriminative Correlation Filter, TrDiMP) 기반 파이프라인 모두에 통합하여 그 범용성과 성능 향상을 입증하였다.

## 📎 Related Works

**Visual Tracking**
최근의 추적 연구는 템플릿 매칭 기반의 Siamese 네트워크와 Fourier 도메인에서 릿지 회귀(Ridge regression)를 푸는 DCF 기반 방법으로 나뉜다. Siamese 추적기는 효율성이 높고, DCF 기반 방법(예: DiMP)은 판별 능력이 뛰어나다. 하지만 두 방식 모두 대부분의 경우 추적을 프레임별 검출 문제로 접근하여 시간적 특성을 제대로 활용하지 못한다. 일부 연구에서 GNN이나 광학 흐름(Optical flow)을 사용했으나, 본 논문은 이를 Transformer를 통해 보다 깔끔하게 해결하고자 한다.

**Transformer**
Transformer는 원래 NLP 분야의 기계 번역을 위해 제안되었으며, Self-attention 메커니즘을 통해 입력 시퀀스 전체의 정보를 집계한다. 최근 컴퓨터 비전의 객체 검출(DETR 등)에도 도입되었으나, 비디오 추적과 같이 프레임 간의 시간적 정보 전파가 중요한 시나리오에 Transformer의 인코더-디코더 구조를 적용한 연구는 드물었다.

## 🛠️ Methodology

### 전체 시스템 구조

본 프레임워크는 Siamese-like 구조를 따르며, 상단 브랜치에서는 Transformer Encoder를 통해 추적 모델을 생성하고, 하단 브랜치에서는 Transformer Decoder를 통해 검색 영역을 강화한다.

### 주요 구성 요소 및 상세 설명

#### 1. Transformer 구조의 수정 사항

추적 작업의 특성에 맞게 기존 Transformer를 다음과 같이 수정하였다.

- **인코더-디코더 분리**: 직렬 구조가 아닌 병렬 브랜치로 구성하여 Siamese 파이프라인에 적합하게 만들었다.
- **블록 가중치 공유(Weight-sharing)**: 인코더와 디코더의 Self-attention 블록이 가중치를 공유하여 템플릿과 검색 영역의 임베딩이 동일한 특징 공간에 존재하도록 하였다.
- **인스턴스 정규화(Instance Normalization)**: NLP의 Layer Norm 대신 이미지 패치 레벨의 Instance Norm을 사용하여 이미지의 진폭 정보를 유지하였다.
- **경량화 설계(Slimming)**: 효율성을 위해 Fully-connected feed-forward 층을 제거하고 단일 헤드 어텐션(Single-head attention)을 사용하였다.

#### 2. Transformer Encoder

인코더는 여러 템플릿 특징들의 집합 $T = \text{Concat}(T_1, \dots, T_n)$을 입력으로 받는다. Self-attention을 통해 템플릿 간의 상호 관계를 학습하며, 다음과 같은 방정식으로 인코딩된 특징 $\hat{T}$를 생성한다.

$$\hat{T} = \text{Ins. Norm}(A^{T \to T}T' + T')$$

여기서 $A^{T \to T}$는 템플릿 간의 유사도 행렬이며, 이를 통해 개별 템플릿 특징들이 서로를 강화하여 더 조밀한 타겟 표현을 획득하게 된다.

#### 3. Transformer Decoder

디코더는 검색 패치 특징 $S$를 입력으로 받으며, 두 가지 전파 과정을 거친다.

**가. 마스크 전파 (Mask Transformation)**
타겟의 위치 정보를 담은 가우시안 마스크 $M$을 생성하고, 검색 특징 $\hat{S}$와 인코딩된 템플릿 특징 $\hat{T}$ 사이의 Cross-attention 행렬 $A^{T \to S}$를 계산하여 마스크를 전파한다.

$$\hat{S}_{\text{mask}} = \text{Ins. Norm}(A^{T \to S}M' \otimes \hat{S})$$

이 과정은 이전 프레임의 공간적 주의 집중(Spatial attention)을 현재 프레임으로 옮겨 타겟 후보 영역을 강조한다.

**나. 특징 전파 (Feature Transformation)**
배경 노이즈를 억제하기 위해 템플릿 특징에 마스크를 곱한 후, Cross-attention을 통해 특징을 전파한다.

$$\hat{S}_{\text{feat}} = \text{Ins. Norm}(A^{T \to S}(\hat{T} \otimes M') + \hat{S})$$

최종적으로 두 결과를 결합하여 강화된 검색 특징 $\hat{S}_{\text{final}}$을 생성한다.

$$\hat{S}_{\text{final}} = \text{Ins. Norm}(\hat{S}_{\text{feat}} + \hat{S}_{\text{mask}})$$

### 추론 및 학습 절차

- **학습**: 백본 네트워크, Transformer, 추적 모델을 엔드투엔드(End-to-end) 방식으로 함께 학습시킨다.
- **온라인 추적**: 템플릿 집합 $T$를 동적으로 업데이트한다. 5프레임마다 가장 오래된 템플릿을 버리고 현재 프레임을 추가하며, 최대 20개의 템플릿을 유지한다.
- **파이프라인 적용**:
  - **TrSiam**: $\hat{T}$를 CNN 커널로 사용하여 $\hat{S}_{\text{final}}$과 교차 상관(Cross-correlation)을 수행한다.
  - **TrDiMP**: $\hat{T}$를 이용하여 판별적 CNN 커널을 학습시키고 $\hat{S}_{\text{final}}$과 컨볼루션을 수행한다.

## 📊 Results

### 실험 설정

- **백본**: ResNet-50
- **데이터셋**: LaSOT, TrackingNet, GOT-10k, UAV123, NfS, OTB-2015, VOT2018/2019
- **지표**: Precision, Success (AUC), Average Overlap (AO), EAO 등

### 주요 결과

1. **성능 향상**: TrDiMP와 TrSiam 모두 기존 SOTA 추적기들을 능가하는 결과를 보였다. 특히 TrackingNet에서 TrDiMP는 Normalized Precision 83.3%, Success 78.4%를 기록하였다.
2. **단순 모델의 재발견**: 놀라운 점은 단순한 Siamese 매칭 기반의 TrSiam이 Transformer 도입만으로 복잡한 DiMP 모델에 근접하는 성능을 냈다는 것이다. 이는 시간적 문맥 활용의 중요성을 입증한다.
3. **강건성 검증**: LaSOT의 속성 분석 결과, 모션 블러, 배경 혼란, 뷰포인트 변경 등 다양한 까다로운 시나리오에서 성능이 크게 향상됨을 확인하였다.
4. **효율성**: TrSiam은 약 35 FPS, TrDiMP는 약 26 FPS의 속도로 동작하여 실시간 추적이 가능하다.

## 🧠 Insights & Discussion

### 강점

본 연구는 Transformer를 단순한 특징 추출기가 아닌, **프레임 간의 정보를 연결하는 브리지(Bridge)**로 활용하였다. 특히 특징 전파와 마스크 전파를 동시에 수행함으로써, 외관 변화에 유연하게 대응하고 배경의 방해 요소(Distractors)를 효과적으로 억제할 수 있음을 보여주었다.

### 한계 및 비판적 해석

- **가려짐 문제**: 논문의 Failure case 분석에서 언급되었듯, 타겟이 완전히 가려지거나 화면 밖으로 나가는 경우 Cross-attention 맵이 부정확해져 추적에 실패한다. 이는 Transformer가 '현재 존재하는' 특징들 간의 관계를 맺는 도구이지, 완전히 사라진 객체를 예측하는 생성 모델은 아니기 때문이다.
- **계산 비용**: Attention matrix의 크기로 인해 메모리 사용량이 증가하는 문제가 있으며, 이는 고해상도 이미지 처리 시 병목 현상이 될 수 있다.
- **가정**: 템플릿 업데이트 주기(5프레임)와 최대 저장 개수(20개)라는 하이퍼파라미터에 의존하고 있어, 비디오의 속도나 변화율에 따라 최적값이 달라질 수 있다.

## 📌 TL;DR

본 논문은 비디오 추적에서 무시되었던 시간적 문맥을 활용하기 위해 **Transformer Encoder-Decoder 구조를 Siamese 추적 파이프라인에 통합**한 연구이다. 인코더는 템플릿들을 상호 강화하고, 디코더는 이전 프레임의 특징과 공간 마스크를 현재 검색 영역으로 전파한다. 이를 통해 단순한 Siamese 모델조차 최신 고성능 추적기에 필적하는 성능을 낼 수 있음을 증명했으며, 여러 벤치마크에서 SOTA를 달성하였다. 이 연구는 향후 비디오 기반 작업에서 프레임 간의 관계 모델링이 핵심적인 역할을 할 것임을 시사한다.
