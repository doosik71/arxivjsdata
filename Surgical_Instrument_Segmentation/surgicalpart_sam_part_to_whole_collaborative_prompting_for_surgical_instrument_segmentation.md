# SurgicalPart-SAM: Part-to-Whole Collaborative Prompting for Surgical Instrument Segmentation

Wenxi Yue, Jing Zhang, Kun Hu, Qiuxia Wu, Zongyuan Ge, Yong Xia, Jiebo Luo, and Zhiyong Wang (2024)

## 🧩 Problem to Solve

본 논문은 수술 도구 분할(Surgical Instrument Segmentation, SIS) 작업에서 기존 모델들이 가진 한계를 해결하고자 한다. 수술 도구 분할은 수술 계획, 로봇 내비게이션, 기술 평가 등 다양한 다운스트림 애플리케이션의 기초가 되는 중요한 작업이다.

기존 접근 방식은 크게 두 가지 문제점을 가지고 있다. 첫째, 특정 도구에 특화된 Specialist 모델들은 방대한 파라미터를 학습시켜야 하므로 개발 비용이 매우 높다. 둘째, Segment Anything Model(SAM)과 같은 파운데이션 모델을 Zero-shot으로 적용할 경우, 자연 이미지와 의료 이미지 사이의 큰 도메인 격차(Domain Gap)와 수술 도구 특유의 복잡한 구조 및 세부 디테일로 인해 일반화 성능이 현저히 떨어진다.

또한, 기존의 SAM 효율적 튜닝 방식(예: SurgicalSAM)은 도구 마스크를 단순한 단일 엔티티로 처리하여 도구의 복잡한 구조적 지식을 무시하며, 카테고리 기반 프롬프트는 수술 도구의 세부 구조를 설명하기에 유연성과 정보량이 부족하다는 문제가 있다. 따라서 본 논문은 전문가의 도구 구조 지식을 통합하여 텍스트 프롬프트 기반으로 정밀한 수술 도구 분할을 수행하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 도구가 '샤프트(Shaft)', '손목(Wrist)', '팁(Tip)'과 같은 부분(Part)들로 구성되어 있다는 전문가의 구조적 지식을 SAM의 튜닝 과정에 명시적으로 통합하는 것이다.

이를 위해 **Part-to-Whole Collaborative Prompting**이라는 새로운 프레임워크를 제안한다. 이는 단순히 도구의 이름만 사용하는 것이 아니라, 카테고리 수준과 부분 수준의 텍스트를 결합한 협업 프롬프트를 통해 SAM이 도구의 전체 구조와 세부 디테일을 동시에 이해하도록 유도한다. 결과적으로 최소한의 학습 파라미터만으로도 수술 도구의 복잡한 구조를 정확하게 파악하고 세밀한 영역까지 분할할 수 있는 능력을 확보한다.

## 📎 Related Works

### 1. 수술 도구 분할 (SIS)

기존 연구들은 주로 U-Net 기반의 semantic segmentation이나 Mask R-CNN 기반의 instance segmentation을 사용하는 Specialist 모델에 집중해 왔다. 하지만 이러한 모델들은 전체 파라미터를 학습시켜야 하므로 비용이 많이 든다. 최근에는 SAM을 활용한 효율적 튜닝 방식이 제안되었으나, 도구를 단일 객체로 취급하여 세부 구조를 간과하는 한계가 있다.

### 2. 텍스트 프롬프트 가능 분할 (Text Promptable Segmentation)

자연어 프롬프트를 사용하는 방식은 더 풍부한 문맥 정보와 유연성을 제공한다. TP-SIS와 같이 CLIP을 활용한 시도가 있었으나, 부분 마스크를 단순히 감독 신호로만 사용했을 뿐 부분 간의 구조적 의존성을 고려하지 않았으며, CLIP 이미지 인코더 전체를 튜닝해야 하므로 학습 비용이 높다.

### 3. Segment Anything Model (SAM)

SAM은 강력한 일반화 능력을 갖추었지만, 의료 도메인에서는 성능이 저하된다. 또한 수술 현장에서 매 프레임마다 포인트나 박스 프롬프트를 수동으로 입력하는 것은 불가능에 가깝다. 기존의 적응(Adaptation) 방법들은 상호작용성이 부족하거나 유연하지 못한 카테고리 ID 기반 프롬프트에 의존하는 경향이 있다.

## 🛠️ Methodology

SP-SAM은 크게 네 가지 주요 구성 요소로 이루어져 있다. 전체 시스템은 frozen 상태인 SAM Image Encoder와 CLIP Text Encoder를 포함하며, 나머지 가벼운 모듈들만을 튜닝하는 Efficient-tuning 방식을 취한다.

### 1. Collaborative Prompts 및 Cross-Modal Prompt Encoder

단순한 카테고리 이름 대신, $\text{\{[part name] of [instrument category name]\}}$ 형태의 **Collaborative Prompts**를 정의한다. 예를 들어, 'Large Needle Driver'의 경우 'Shaft of Large Needle Driver', 'Wrist of Large Needle Driver' 등의 세트를 구성한다.

- **특징 추출:** CLIP Text Encoder를 통해 텍스트 임베딩을 추출하고, SAM의 임베딩 공간과 맞추기 위해 학습 가능한 $\text{Transfer MLP}$를 통과시켜 부분 임베딩 $T_{part} \in \mathbb{R}^{P \times d}$를 생성한다.
- **교차 모달 인코딩:** 이미지 임베딩 $F_I$와 $T_{part}$ 사이의 공간적 어텐션(Spatial Attention)을 통해 유사도 맵(Similarity map) $S$를 계산한다.
  $$S = T_{part} \times F_I^\top \in \mathbb{R}^{P \times h \times w}$$
- 이 $S$를 이용하여 이미지 임베딩을 활성화시킨 후($F'_I = S \circ F_I + F_I$), MLP와 CNN을 통해 각 부분에 대한 **Part Sparse Embeddings**($F_{part}^S$)와 **Part Dense Embeddings**($F_{part}^D$)를 생성한다.

### 2. Part-to-Whole Adaptive Fusion

부분별 임베딩을 통합하여 전체 도구 임베딩 $\{F_S, F_D\}$를 생성하는 단계이다. 이때 도구마다 구성 부품이 다르고(Category-specific), 이미지마다 가려진 부분(Occlusion)이 다르다는 점을 해결하기 위해 두 가지 어텐션 메커니즘을 사용한다.

- **Category Part Attention:** 전문가 지식 기반의 카테고리-부분 관계 행렬 $D_{CP} \in \{0,1\}^{C \times P}$를 활용하여 해당 카테고리에 존재하는 부분에 가중치를 부여한다.
- **Image Part Attention:** Global CNN을 통해 이미지 전체 기술자(Global descriptor) $F_G$를 학습하고, 이를 부분 임베딩 $T_{part}$와 곱해 이미지 특이적 가중치 $W = F_G \times T_{part}^\top$를 계산한다.
- **최종 융합 방정식:**
  $$F_S = F_{part}^S \circ \text{ReLU}(D_{c^*}) \in \mathbb{R}^{P \times n \times d}$$
  $$F'_D = F_{part}^D \circ D_{c^*} \circ W \in \mathbb{R}^{P \times h \times w \times d}$$
  $$F_D = \sum_{p=1}^{P} F'_D \in \mathbb{R}^{h \times w \times d}$$

### 3. Hierarchical Decoding 및 손실 함수

전체 수준과 부분 수준의 임베딩을 모두 SAM Decoder에 입력하여 **계층적 디코딩(Hierarchical Decoding)**을 수행한다. 이는 모델이 전체 구조와 세부 부품 특성을 동시에 학습하게 한다. 손실 함수는 전체 마스크와 각 부분 마스크에 대한 Dice Loss의 합으로 정의된다.
$$L = L_D(M^{(c)}, G^{(c)}) + \sum_{p=1}^{P} d_{cp} L_D(M_p^{(c)}, G_p^{(c)})$$
여기서 $L_D$는 예측값 $m_i$와 정답 $g_i$ 사이의 Dice Loss를 계산하는 식이다.

## 📊 Results

### 실험 설정

- **데이터셋:** EndoVis2017 및 EndoVis2018 데이터셋 사용.
- **지표:** Challenge IoU, IoU, mean class IoU (mc IoU)를 사용하여 평가.
- **비교 대상:** TernausNet, ISINet, TP-SIS와 같은 Specialist 모델 및 SurgicalSAM, Zero-shot SAM 기반 모델들.

### 주요 결과

- **정량적 성능:** SP-SAM은 두 데이터셋 모두에서 SOTA 성능을 달성하였다. 특히 EndoVis2018에서 Challenge IoU 기준으로 SurgicalSAM 대비 약 3.91 상승하였으며, mc IoU에서도 큰 폭의 향상을 보였다. 이는 도구 간의 변별력이 높아졌음을 의미한다.
- **효율성:** TP-SIS가 약 131M의 파라미터를 튜닝하는 반면, SP-SAM은 단 **8.62M**의 파라미터만 튜닝하고도 더 우수한 성능을 기록하였다.
- **오라클 비교:** Ground-truth(GT) Centroid를 사용한 SAM보다 월등히 좋으며, GT Bounding Box를 사용한 오라클 설정에 근접하는 성능을 보였다. 이는 수동 프롬프트 입력 없이도 텍스트만으로 높은 정밀도를 얻었음을 시사한다.

## 🧠 Insights & Discussion

### 강점

SP-SAM의 가장 큰 강점은 단순한 데이터 학습을 넘어 '수술 도구는 특정 부분들의 조합으로 이루어져 있다'는 **전문가 지식을 아키텍처에 명시적으로 녹여냈다**는 점이다. 이를 통해 SAM이 의료 도메인의 복잡한 구조를 더 빠르게 이해하게 했으며, 특히 도구의 팁(Tip)이나 가장자리(Edge)와 같은 세밀한 영역의 분할 성능을 획기적으로 높였다. 이는 실제 수술 환경에서 안전성을 보장하는 데 매우 중요한 요소이다.

### 한계 및 논의사항

논문에서는 텍스트 프롬프트를 통해 정적인 이미지 분할 성능을 높였으나, 실제 수술 영상은 연속적인 프레임으로 구성된다. 따라서 향후 연구에서는 **시계열 정보(Temporal cues)**를 통합하여 일관성을 높일 필요가 있다. 또한, 현재는 수술 도구에만 집중하고 있으나, 실제 환경에서는 인체 조직(Human tissues)과 같은 배경 타겟과의 상호작용 및 분할 문제도 함께 다뤄져야 할 것이다.

## 📌 TL;DR

본 논문은 수술 도구의 구조적 특성(부분-전체 관계)을 활용한 SAM 효율적 튜닝 방법론인 **SurgicalPart-SAM(SP-SAM)**을 제안한다. **Collaborative Prompts**와 **Part-to-Whole Adaptive Fusion**을 통해 도구의 세부 부품 지식을 통합함으로써, 매우 적은 파라미터 튜닝만으로도 기존의 Specialist 모델과 SAM 기반 모델들을 뛰어넘는 정밀한 수술 도구 분할 성능을 달성하였다. 이 연구는 파운데이션 모델을 매우 전문적인 도메인에 적응시킬 때, 도메인 지식을 어떻게 구조적으로 통합할 수 있는지에 대한 중요한 방향성을 제시한다.
