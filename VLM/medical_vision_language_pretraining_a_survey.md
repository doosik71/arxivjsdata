# Medical Vision Language Pretraining: A survey

Prashant Shrestha, Sanskar Amgain, Bidur Khanal, Cristian A. Linte, Binod Bhattarai (2023)

## 🧩 Problem to Solve

본 논문은 의료 분야에서 딥러닝 모델 학습의 가장 큰 병목 현상인 **레이블링된 데이터의 부족(scarcity of labeled data)** 문제를 해결하기 위한 방법론으로 Medical Vision Language Pretraining (VLP)에 주목한다. 의료 데이터는 특성상 전문가의 정밀한 레이블링이 필요하며, 개인정보 보호 문제로 인해 대규모 데이터셋 구축이 매우 어렵다.

이에 대한 해결책으로, 레이블이 없는 대규모의 이미지-텍스트 쌍(paired/unpaired data)을 활용하는 **자기지도학습(Self-Supervised Learning, SSL)** 기반의 VLP가 제안되었다. VLP는 사전 학습(Pretraining) 단계에서 방대한 의료 지식과 강건한 특징 표현(robust feature representations)을 학습하고, 이를 적은 양의 레이블된 데이터만으로 하위 작업(downstream tasks)에 적응시키는 것을 목표로 한다.

그동안 의료 VLP에 대한 개별 연구는 많았으나, 사전 학습 목적 함수, 아키텍처, 데이터셋 및 평가 작업 등을 체계적으로 분석하고 정리한 포괄적인 서베이 논문이 부재했다는 점이 본 연구의 핵심 동기이다.

## ✨ Key Contributions

본 논문의 주요 기여는 의료 VLP 분야의 최신 연구들을 체계적으로 분류하고 분석한 최초의 종합 서베이 보고서를 제공한다는 점이다. 구체적인 기여 사항은 다음과 같다.

- **사전 학습 목적 함수의 체계적 분류**: Masked Prediction, Contrastive Learning, Matching Prediction, 그리고 이들의 Hybrid 형태로 구분하여 수학적 정의와 함께 상세히 분석하였다.
- **모델 아키텍처 분석**: 인코더 구조(CNN, ViT)와 모달리티 융합(Modality Fusion) 방식(Early, Late, No Fusion)에 따른 차이점을 규명하였다.
- **하위 작업 및 데이터셋 매핑**: 분류, 세그멘테이션, 객체 탐지, 리트리벌, 리포트 생성, VQA 등 다양한 하위 작업과 이에 활용되는 주요 데이터셋을 정리하였다.
- **의료 도메인 특화 챌린지 제시**: 일반 도메인 VLP와 달리 의료 VLP가 직면한 데이터 불균형, 의료 리포트의 특이성(부정 표현 등), 계산 병리(Computational Pathology)의 거대 이미지 처리 문제 등을 심도 있게 논의하였다.

## 📎 Related Works

논문은 기존의 바이오메디컬 이미지-텍스트 학습 관련 연구(예: [28])가 존재하지만, 대부분 평가 작업에 치중되어 있으며 사전 학습 방법론에 대한 기술적 깊이와 포괄적인 개요가 부족했음을 지적한다.

기존의 일반 도메인 VLP(예: CLIP)가 대규모 데이터셋을 통해 뛰어난 성능을 보였으나, 의료 도메인에 그대로 적용하기에는 다음과 같은 차별점이 존재한다.

- **데이터의 특성**: 의료 이미지는 일반 이미지보다 미세한 차이로 질병 유무가 갈리며, 의료 리포트는 전문 용어, 약어, 부정 표현(negation)이 빈번하게 등장한다.
- **데이터 규모**: 수억 개의 쌍을 사용하는 일반 VLP와 달리, 의료 VLP는 수십만 개 수준의 데이터셋(예: MIMIC-CXR)에 의존하는 경우가 많다.

## 🛠️ Methodology

본 논문은 의료 VLP의 핵심 구성 요소를 네 가지 관점에서 분석한다.

### 1. 사전 학습 목적 함수 (Pretraining Objectives)

**A. Masked Prediction**
입력 데이터의 일부를 마스킹하고, 나머지 부분을 통해 마스킹된 영역을 복원하는 방식이다.

- **MLM (Masked Language Modeling)**: 텍스트 토큰을 마스킹하고 복원한다.
- **MIM (Masked Image Modeling)**: 이미지 패치를 마스킹하고 복원한다.
- **수식**: 복원 손실 $L_{recon}$을 사용하여 다음과 같이 정의한다.
$$L_{mp} = \sum_{i \in (v,t)} L_{recon}(M_\theta(\hat{X}_v, \hat{X}_t), X_i)$$
여기서 $\hat{X}$는 마스킹된 입력, $X$는 원본 입력이다.

**B. Contrastive Learning**
쌍을 이루는 데이터(positive pair) 간의 유사도는 최대화하고, 서로 다른 데이터(negative pair) 간의 유사도는 최소화한다.

- **Global Alignment**: 이미지 전체와 리포트 전체의 임베딩을 정렬한다.
$$L_{con,a \to b} = -\sum_{i} \log \frac{e^{sim(Z_{a_i}, Z_{+b_i})}}{e^{sim(Z_{a_i}, Z_{+b_i})} + \sum_{j=1, j \neq i}^{m} e^{sim(Z_{a_i}, Z_{-b_j})}}$$
- **Local Alignment**: 단어 수준의 텍스트 특징과 그에 대응하는 이미지 지역(region) 특징을 정렬하여 세밀한 특징을 학습한다.

**C. Matching Prediction**
두 모달리티가 서로 일치하는지 여부를 이진 분류(Binary Classification)하는 방식이며, 주로 Binary Cross-Entropy (BCE) 손실을 사용한다.
$$L(X_v, X_t) = \mathbb{E}[\hat{y} \cdot \log(M_\theta(X_v, X_t)) + (1-\hat{y}) \cdot \log(1-M_\theta(X_v, X_t))]$$

**D. Hybrid Objectives**
위의 목적 함수들을 가중 합산하여 상호 보완적인 효과를 얻는다.
$$L = \sum_{i=1}^{M} \lambda_i L_i$$

### 2. 모델 아키텍처 (Architecture)

- **Encoder**: 이미지 인코더로는 ResNet(CNN)과 ViT가 주로 사용되며, 텍스트 인코더로는 BERT 기반의 Transformer 구조가 표준적으로 사용된다.
- **Modality Fusion**:
  - **No Fusion**: 단순 정렬만 수행하며, 계산 효율이 높고 유니모달 작업에 유리하다.
  - **Early Fusion**: 이미지 토큰과 텍스트 토큰을 하나의 통합된 Transformer 인코더에 입력하는 Single-stream 구조이다.
  - **Late Fusion**: 각각의 인코더를 거친 후 Cross-Attention 등을 통해 융합하는 Dual-stream 구조이다.

### 3. 데이터 증강 (Data Augmentation)

의료 데이터의 부족을 해결하기 위해 사용된다. 다만, 의료 이미지의 경우 무분별한 Crop이나 Flip이 텍스트의 공간적 설명(예: "왼쪽 폐에 병변이 있음")과 불일치(misalignment)를 일으킬 수 있으므로 주의가 필요함을 강조한다.

### 4. 하위 작업 평가 (Downstream Evaluation)

학습된 모델은 다음과 같은 방식으로 활용된다.

- **Zero-shot Classification**: 학습되지 않은 클래스에 대해 텍스트 프롬프트와 이미지 임베딩의 유사도를 측정하여 분류한다.
- **Linear Probing**: 인코더를 고정하고 선형 분류기(Linear Head)만 학습시킨다.
- **Fine-tuning**: 전체 모델을 하위 작업 데이터셋에 맞춰 미세 조정한다.

## 📊 Results

본 논문은 개별 실험 결과보다는 74편의 논문을 분석한 종합적인 경향성을 제시한다.

- **목적 함수와 성능의 상관관계**: Contrastive Learning은 Zero-shot 성능을 비약적으로 향상시키며, Local Alignment는 세그멘테이션이나 객체 탐지와 같은 국소적(localized) 작업에서 필수적이다.
- **융합 방식의 영향**: Late Fusion과 Early Fusion은 VQA나 리포트 생성과 같은 복잡한 멀티모달 상호작용이 필요한 작업에서 No Fusion 방식보다 우수한 성능을 보인다.
- **데이터셋의 영향**: MIMIC-CXR과 같은 대규모 데이터셋으로 사전 학습한 모델이 하위 작업에서 일반화 성능이 훨씬 뛰어남이 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 핵심 통찰

본 논문은 의료 VLP가 단순히 일반 VLP의 적용이 아니라, **의료 도메인만의 특수성**을 반영해야 함을 역설한다. 특히, 단순한 텍스트-이미지 정렬을 넘어 의료 지식 베이스(UMLS 등)를 통합하거나, 리포트의 계층적 구조(Findings vs Impressions)를 고려한 정렬 방식이 성능 향상의 핵심임을 밝혀냈다.

### 한계 및 비판적 해석

- **데이터 불균형 문제**: 의료 데이터의 롱테일(Long-tail) 분포로 인해 희귀 질환에 대한 표현 학습이 어렵다. 이는 Contrastive Learning에서 유사한 질병을 서로 다른 샘플로 취급하는 **False Negative** 문제를 야기하며, 이를 해결하기 위한 소프트 라벨링이나 클러스터링 기법의 도입이 필요하다.
- **계산 병리(WSI)의 한계**: 기가픽셀 단위의 Whole Slide Image(WSI)를 처리하기 위한 메모리 효율적인 VLP 아키텍처가 여전히 부족하며, 현재의 패치 기반 접근법은 전체 맥락을 놓칠 위험이 있다.
- **실제 임상 적용의 괴리**: 기술적인 성능 지표(BLEU, F1-score)와 실제 임상적 유효성(Clinical Efficacy) 사이의 간극이 존재한다. 단순한 텍스트 유사도보다는 진단 정확도가 더 중요하다는 점이 지적된다.

## 📌 TL;DR

본 논문은 의료 분야의 레이블 데이터 부족 문제를 해결하기 위한 **Medical Vision Language Pretraining (VLP)**의 최신 동향을 집대성한 최초의 종합 서베이이다. Masked Prediction, Contrastive Learning, Matching Prediction이라는 세 가지 핵심 목적 함수와 다양한 융합 아키텍처를 체계적으로 분석하였으며, 의료 데이터 특유의 부정 표현, 데이터 불균형, 거대 이미지 처리 등의 챌린지를 제시하였다. 이 연구는 향후 의료용 파운데이션 모델(Foundation Model) 구축과 임상 현장 적용을 위한 기술적 이정표를 제공한다는 점에서 매우 중요한 역할을 할 것으로 기대된다.
