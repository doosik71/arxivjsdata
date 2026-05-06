# Pushing the Limits of Vision-Language Models in Remote Sensing without Human Annotations

Keumgang Cha, Donggeun Yu, Junghoon Seo (2024)

## 🧩 Problem to Solve

본 논문은 원격 탐사(Remote Sensing, RS) 분야에서 Vision-Language Model(VLM)을 구축하기 위한 대규모 데이터셋의 부족 문제를 해결하고자 한다. 자연 이미지 도메인에서는 웹 크롤링을 통해 방대한 양의 이미지-텍스트 쌍을 확보할 수 있어 강력한 파운데이션 모델(Foundation Model) 구축이 가능하지만, 원격 탐사 도메인은 이러한 데이터의 가용성이 매우 낮다.

기존의 원격 탐사용 Vision-Language 데이터셋은 규모가 작아 견고한 파운데이션 모델을 학습시키기에 부족하며, 인간이 직접 레이블링하는 방식은 비용과 시간이 지나치게 많이 소요된다는 한계가 있다. 따라서 본 연구의 목표는 인간의 개입 없이 기계 학습 모델을 이용하여 대규모의 고품질 Vision-Language 데이터셋을 구축하고, 이를 통해 원격 탐사 도메인에 최적화된 VLM인 RSCLIP을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 대규모 언어 모델(LLM) 기반의 이미지 디코딩 모델인 InstructBLIP을 활용하여, 기존의 원격 탐사 이미지 데이터셋으로부터 자동으로 텍스트 설명을 생성함으로써 대규모의 Vision-Language 쌍을 구축하는 것이다.

단순히 하나의 캡션을 생성하는 것이 아니라, 서로 다른 두 가지 프롬프트("Write a short description for the image." 및 "Describe the image in detail")를 사용하여 짧은 요약문과 상세한 설명문을 동시에 생성함으로써 언어적 다양성과 품질을 확보하였다. 이를 통해 구축된 약 960만 개의 데이터셋으로 학습된 RSCLIP은 인간의 주석 없이도 기존의 합성 레이블 기반 모델보다 우수한 성능을 보이며, 인간 주석 기반 모델과 대등한 수준의 성능을 달성하였다.

## 📎 Related Works

원격 탐사 커뮤니티에서는 Masked Image Modeling(MIM) 방식의 파운데이션 모델 연구가 활발히 진행되어 왔으나, 이러한 모델들은 핵심 컴퓨터 비전 작업을 수행할 때 여전히 지도 학습 기반의 미세 조정(Supervised Fine-tuning)에 의존해야 한다는 한계가 있다.

이를 해결하기 위해 CLIP과 같은 대조 학습(Contrastive Learning) 기반의 Vision-Language 모델이 주목받고 있다. 기존 연구인 RS5M은 BLIP-2를 사용하여 데이터셋을 구축하였고, RemoteCLIP은 전통적인 데이터셋을 Vision-Language 형식으로 변환하려 시도하였다. 그러나 본 논문은 InstructBLIP을 통해 더 정교하고 다양한 텍스트 설명을 생성함으로써 데이터의 품질을 높였으며, 더 방대한 양의 이미지 코퍼스를 활용하여 비전 인코더의 성능을 극대화했다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. 대규모 Vision-Language 데이터셋 생성

본 연구는 InstructBLIP을 이용하여 개별 이미지에서 Vision-Language 쌍을 추출한다. 텍스트의 다양성을 위해 이미지당 두 가지 프롬프트를 적용하여 간결한 캡션과 상세한 캡션을 각각 생성하였다.

- **데이터 소스**: fMoW, Million-AID, DFC2019, DFC2021, DeepGlobe, DIOR, HRSC, Inria 및 RS5M의 일부를 사용하였다.
- **전처리**: 모든 이미지는 InstructBLIP의 입력 규격에 맞게 $512 \times 512$ 픽셀 크기로 리사이징 및 크롭되었다.
- **최종 규모**: 총 9,686,720개의 Vision-Language 쌍이 구축되었으며, 이 중 약 627만 개는 InstructBLIP으로 생성되었고 약 340만 개는 RS5M에서 가져왔다.

### 2. RSCLIP 모델 아키텍처 및 학습 절차

RSCLIP은 표준적인 CLIP 프레임워크를 따르며, 이미지와 텍스트의 표현 공간을 일치시키는 것을 목표로 한다.

- **모델 구조**:
  - **Vision Encoder**: Vision Transformer(ViT)를 기반으로 하며, 패치 사이즈 16, 히든 사이즈 768, MLP 사이즈 3072, 12개의 헤드와 12개의 레이어로 구성된다.
  - **Text Encoder**: BERT-base 모델을 사용하며, 히든 사이즈 768, MLP 사이즈 3072, 12개의 헤드와 12개의 레이어로 구성된다.
  - 두 인코더 모두 처음부터 학습시키지 않고, Million-AID 데이터셋으로 학습된 MAE(Masked AutoEncoder)와 BERT-base의 사전 학습 가중치를 초기값으로 사용하였다.

- **학습 목표 및 손실 함수**:
    유사한 의미를 가진 이미지-텍스트 쌍은 가깝게, 서로 다른 쌍은 멀게 배치하도록 InfoNCE loss를 사용하여 최적화한다.
    $$ \mathcal{L} = -\log \frac{\exp(\text{sim}(v_i, t_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j) / \tau)} $$
    여기서 $\text{sim}(\cdot)$은 코사인 유사도를 의미하며, $\tau$는 Temperature 파라미터(0.07)이다.

- **학습 세부 설정**:
  - **데이터 증강**: 텍스트 설명에 방향, 위치, 색상 정보가 포함되어 있으므로 강한 증강은 피하고, $0.8 \sim 1.0$ 범위의 Resized Random Cropping만 적용하였다.
  - **최적화**: AdamW 옵티마이저와 Cosine Decay 스케줄러를 사용하였으며, 16대의 GPU에서 배치 사이즈 112(GPU당)로 10 에포크 동안 학습하였다. 입력 이미지 크기는 448 픽셀이다.

## 📊 Results

### 1. 주요 실험 (Main Experiments)

다운스트림 태스크 데이터셋을 사전 학습에 사용하지 않은 상태에서 모델의 일반화 능력을 평가하였다.

- **Image-Text Retrieval**: RSICD 및 RSITMD 데이터셋에서 평가하였으며, R@1, R@5, R@10 및 Mean Recall(mR)을 측정하였다. RSCLIP은 RSITMD의 R@1을 제외한 거의 모든 지표에서 기존 모델들을 능가하는 성능을 보였다.
- **Zero-shot Classification**: AID 및 RESISC45 데이터셋에서 "a satellite image of [class name]"이라는 템플릿 프롬프트를 사용하여 성능을 측정하였다. RSCLIP은 두 데이터셋 모두에서 가장 높은 Top-1 Accuracy를 기록하였다.
- **Semantic Localization**: AIR-SLT 데이터셋을 통해 이미지 내 특정 의미 영역을 찾아내는 능력을 측정하였다. $R_{su}$ (중요 영역 비율), $R_{as}$ (중심점 편차), $R_{da}$ (어텐션 분산도) 등을 측정하였으며, 종합 지표인 $R_{mi}$에서 최적의 성능을 기록하였다.

### 2. 추가 실험 (Additional Experiments)

비전 인코더 단독 성능 및 직접적인 VL-pair를 사용한 모델(RemoteCLIP, S-CLIP 등)과의 비교를 수행하였다.

- **Few-shot Classification**: 1, 4, 8, 16, 32-shot 설정에서 평가하였다. RSCLIP은 RemoteCLIP보다 월등히 높은 정확도를 보였는데, 이는 RSCLIP이 훨씬 더 방대한 이미지 코퍼스로 사전 학습되었기 때문으로 분석된다.
- **Linear Probing 및 k-NN Classification**: 모든 학습 데이터를 사용한 Linear Probing과 k-NN 분류 실험에서도 RSCLIP은 대부분의 데이터셋에서 최고 성능을 기록하였다. 특히 Vision Encoder만 사용하는 작업에서 강력한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 대형 언어 모델을 이용한 이미지 디코딩 방식이 인간의 주석 없이도 고품질의 원격 탐사 데이터셋을 구축할 수 있는 매우 효과적인 방법임을 입증하였다.

**강점 및 분석**:

- **데이터 규모의 승리**: RSCLIP이 Few-shot이나 Linear Probing과 같은 비전 단독 태스크에서 RemoteCLIP을 압도한 이유는 학습에 사용된 이미지 데이터의 양이 훨씬 많았기 때문이다. 이는 VLM의 성능이 단순히 텍스트의 정확도뿐만 아니라, 학습에 사용된 시각적 데이터의 다양성과 양에 크게 의존함을 시사한다.
- **합성 레이블의 효용성**: 직접적인 다운스트림 언어 분포를 학습한 모델보다는 약간 뒤처질 수 있으나, InstructBLIP으로 생성한 합성 레이블만으로도 충분히 경쟁력 있는 성능을 낼 수 있음을 확인하였다.

**한계 및 논의**:

- **텍스트 정밀도**: 합성된 텍스트가 실제 지형의 매우 세부적인 특성을 완벽하게 포착했는지에 대한 정성적 분석은 부족하다.
- **추가 모달리티**: 원격 탐사 이미지에는 다중 분광(Multi-spectral) 정보 등 다양한 모달리티가 존재하지만, 본 연구는 RGB 기반의 접근 방식을 취하였다. 향후 연구에서 이러한 다양한 모달리티를 언어와 결합하는 방향으로 확장이 필요하다.

## 📌 TL;DR

본 연구는 인간의 레이블링 없이 InstructBLIP을 통해 약 960만 개의 원격 탐사 이미지-텍스트 쌍을 자동으로 구축하고, 이를 통해 파운데이션 모델인 **RSCLIP**을 제안하였다. RSCLIP은 Zero-shot 분류, 이미지-텍스트 검색 및 시맨틱 로컬라이제이션에서 기존 모델들을 능가하거나 대등한 성능을 보였으며, 특히 방대한 데이터 학습을 통해 매우 강력한 비전 인코더 성능을 확보하였다. 이 연구는 데이터 수집 비용이 높은 특수 도메인에서 LLM을 활용한 데이터 증강 전략이 매우 유효함을 보여주며, 향후 원격 탐사 분야의 VLM 연구에 중요한 기초가 될 것으로 보인다.
