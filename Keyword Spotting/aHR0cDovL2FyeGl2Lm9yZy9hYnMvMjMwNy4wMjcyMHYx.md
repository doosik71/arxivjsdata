# On-Device Constrained Self-Supervised Speech Representation Learning for Keyword Spotting via Knowledge Distillation

Gene-Ping Yang, Yue Gu, Qingming Tang, Dongsu Du, Yuzong Liu (2023)

## 🧩 Problem to Solve

본 연구는 온디바이스(On-device) 환경에서 키워드 검출(Keyword Spotting, KWS)을 수행하기 위한 효율적인 자기지도 음성 표현 학습(Self-Supervised Speech Representation Learning, S3RL) 모델을 구축하는 것을 목표로 한다. 해결하고자 하는 핵심 문제는 크게 두 가지이다.

첫째, **모델 크기와 연산 비용의 제약**이다. Wav2vec 2.0과 같은 최신 S3RL 모델은 매우 강력한 특징 추출 능력을 갖추고 있으나, 수천만 개의 파라미터를 가진 거대 모델(예: 12-layer Transformer, 95M 파라미터)인 경우가 많아 메모리와 전력이 제한적인 온디바이스 환경에 그대로 배포하는 것이 불가능하다.

둘째, **데이터 편향(Data Bias) 문제**이다. 산업 규모의 키워드 데이터셋은 특정 키워드가 포함된 발화에 치우쳐 있는 경향이 있다. 이러한 편향된 데이터로 대조 학습(Contrastive Learning)을 수행할 경우, 모델이 다양성이 부족한 데이터 내의 가짜 노이즈(Spurious noise)를 인코딩하여 과적합(Overfitting)되는 문제가 발생한다.

따라서 본 논문은 지식 증류(Knowledge Distillation, KD)를 통해 거대 모델의 지식을 경량 모델로 전이함으로써, 제한된 자원 내에서 데이터 편향에 강건한 KWS 모델을 구현하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 거대 모델(Teacher)의 풍부한 표현 능력을 경량 모델(Student)로 효율적으로 전이하기 위해 두 가지 새로운 지식 증류 기법을 제안하는 것이다.

1. **Dual-View Cross-Correlation Distillation**: 단순히 프레임 단위의 특징을 복제하는 기존 방식에서 벗어나, 배치-뷰(Batch-view)와 특징-뷰(Feature-view)라는 두 가지 관점에서 교차 상관 행렬(Cross-correlation matrix)을 정규화한다. 이를 통해 특징 차원의 중복성을 줄이고 샘플 간의 대조 능력을 강화한다.
2. **Teacher Codebook Distillation**: 편향된 도메인 데이터로 학생 모델을 학습시킬 때 발생하는 다양성 부족 문제를 해결하기 위해, 이미 대규모의 다양한 데이터(LibriSpeech)로 학습된 교사 모델의 코드북(Codebook)을 직접 활용하여 학습 목표를 설정한다.

## 📎 Related Works

기존의 S3RL 연구들은 주로 SUPERB와 같은 공개 벤치마크에서 성능을 높이는 데 집중했으며, Wav2vec 2.0, HuBERT 등이 대표적이다. 최근에는 모델 경량화를 위해 DistilHuBERT나 LightHuBERT와 같은 지식 증류 기반의 접근법이 제시되었다.

그러나 기존의 거리 기반 지식 증류(Distance-based distillation) 방식들은 주로 개별 프레임(Single frames)의 유사도에 집중했다. 이는 개별 프레임의 변동성에 취약할 수 있으며, 특히 KWS와 같이 발화 전체의 맥락이 중요한 작업에서는 한계가 있다. 본 논문은 이를 해결하기 위해 발화 단위(Utterance-wise)의 표현을 증류하고, 단순 거리 측정이 아닌 상관관계 기반의 Dual-view 접근 방식을 채택하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

본 시스템은 교사-학생(Teacher-Student) 프레임워크를 따른다. 교사 모델로는 LibriSpeech 960시간 데이터로 학습된 Wav2vec 2.0(95M 파라미터)을 사용하며, 학생 모델로는 3-layer Transformer 기반의 경량 모델(21M 또는 1.6M 파라미터)을 사용한다.

### 1. Dual-View Cross-Correlation Distillation

이 기법은 특징 차원의 중복을 제거하고 샘플 간의 변별력을 높이기 위해 두 가지 뷰의 상관 행렬을 최적화한다. 입력 발화 $X$에 대해 교사 모델의 특징 행렬을 $H \in \mathbb{R}^{b \times d}$, 학생 모델의 특징 행렬을 $O \in \mathbb{R}^{b \times d}$라고 정의한다 (여기서 $b$는 배치 크기, $d$는 특징 차원).

**A. 특징-뷰 (Feature-view): Redundancy Reduction**
특징 차원 간의 상관관계를 계산하여 중복성을 줄인다. 교차 상관 행렬 $C \in \mathbb{R}^{d \times d}$의 각 원소 $C_{ij}$는 다음과 같이 계산된다.

$$C_{ij} = \frac{\sum_{b} H_{bi} O_{bj}}{\sqrt{\sum_{b} (H_{bi})^2} \sqrt{\sum_{b} (O_{bj})^2}}$$

이 행렬 $C$가 단위 행렬(Identity matrix)이 되도록 유도하여, 각 특징 차원이 독립적이고 컴팩트한 정보를 갖게 한다. 손실 함수 $L_C$는 다음과 같다.

$$L_C = \sum_{i} (C_{ii}-1)^2 + \alpha \sum_{i,j \neq i} C_{ij}^2$$

**B. 배치-뷰 (Batch-view): Contrast Operation**
배치 내 샘플 간의 관계를 계산하여 같은 샘플의 특징은 유사하게, 다른 샘플의 특징은 다르게 만든다. 상관 행렬 $G \in \mathbb{R}^{b \times b}$의 각 원소 $G_{ij}$는 다음과 같다.

$$G_{ij} = \frac{\sum_{d} H_{id} O_{jd}}{\sqrt{\sum_{d} (H_{id})^2} \sqrt{\sum_{d} (O_{jd})^2}}$$

손실 함수 $L_G$ 역시 $G$가 단위 행렬이 되도록 유도한다.

$$L_G = \sum_{i} (G_{ii}-1)^2 + \beta \sum_{i,j \neq i} G_{ij}^2$$

최종 DVCC 손실 함수는 Stop-gradient($sg$)를 사용하여 다음과 같이 동적으로 결합된다.

$$L_{DVCC} = L_C / sg(L_C) + L_G / sg(L_G)$$

### 2. Teacher Codebook Distillation

학습 데이터의 편향성을 극복하기 위해, 교사 모델의 양자화된 코드북을 정답셋으로 활용한다. 학생 모델의 예측값 $o_t$가 교사 모델의 양자화 벡터 $k_t$와 가깝게 학습되도록 하는 대조 손실 함수를 정의한다.

$$L_{t-code} = -\sum_{t} \log \frac{\exp(\cos(o_t, k_t))}{\sum_{\tilde{k} \in K_t} \exp(\cos(o_t, \tilde{k}))}$$

여기서 $K_t$는 하나의 양성 샘플 $k_t$와 $N$개의 음성 샘플로 구성된 집합이다.

### 3. 통합 목적 함수 (Combined Objective)

최종적으로 학생 모델은 다음의 통합 손실 함수를 통해 학습된다.

$$L_{combined} = L_{DVCC} + \gamma L_{t-code}$$

## 📊 Results

### 실험 설정

- **데이터셋**: 16,600시간의 내부 Alexa KWS 데이터셋.
- **비교 대상**: 지식 증류를 적용하지 않은 Baseline, DistilHuBERT 기반 모델.
- **지표**: 고정된 오인식률(False Rejection Rate, FRR)에서의 상대적 오인식률(Relative False Acceptance Rate, FAR). 값이 낮을수록 성능이 우수하다.
- **조건**: 일반 환경(Normal)과 노이즈가 섞인 재생 환경(Playback).

### 주요 결과

1. **성능 향상**: 제안된 DVCC 기반 지식 증류 방식은 Baseline 대비 Normal 조건에서 14.6%, Playback 조건에서 21.3%의 상대적 FAR 감소를 보였다.
2. **경량 모델의 효율성**: 1.6M 파라미터의 초경량 모델이 지식 증류를 통해 21M 파라미터의 Baseline 모델과 대등하거나 더 나은 성능을 달성하였다.
3. **Dual-View의 효과**: Ablation study 결과, Normal 조건에서는 Feature-view가 약간 우세했으나, Noisy(Playback) 조건에서는 Dual-view가 단일 뷰 방식들보다 약 2.6%~3% 더 우수한 성능을 보였다. 이는 배치-뷰의 대조 학습 성분이 강건성(Robustness) 확보에 필수적임을 시사한다.
4. **Teacher Codebook의 기여**: 교사 모델의 코드북을 활용했을 때, 특히 노이즈 환경에서 성능이 크게 향상되었다. 이는 다양한 데이터로 학습된 교사 모델의 코드북이 데이터 편향 문제를 완화하는 효과적인 가이드 역할을 했음을 의미한다.
5. **교사 모델 레이어 선택**: 교사 모델의 모든 레이어를 사용하는 것보다 5~8번 레이어의 특징만을 추출하여 증류했을 때 가장 뛰어난 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 온디바이스 KWS라는 제약 사항 하에서 S3RL 모델을 최적화하는 효과적인 방법을 제시하였다.

**강점 및 통찰**:

- **상관관계 기반 증류**: 단순한 거리 측정($L1$, Cosine distance)보다 상관 행렬을 이용한 정규화가 특징 차원의 중복을 줄이고 모델의 일반화 능력을 높이는 데 더 효과적임을 입증하였다.
- **데이터 편향 해결**: 도메인 특화 데이터셋의 편향성 문제를 해결하기 위해 외부 대규모 데이터로 사전 학습된 모델의 '코드북'이라는 정적인 지식을 활용한 점이 매우 전략적이다.
- **레이어 특성 파악**: 모든 레이어가 동일한 가치를 가지지 않으며, KWS 작업에 최적화된 특정 중간 레이어(5-8번)가 존재한다는 점을 발견하여 증류 효율을 높였다.

**한계 및 논의**:

- **범용성 검증 부족**: 본 연구는 오직 온디바이스 KWS 작업에만 집중하여 실험을 진행하였다. 제안된 DVCC 및 코드북 증류 방식이 다른 음성 처리 작업(예: ASR, 화자 인식)에서도 동일한 효과를 낼지는 추가적인 검증이 필요하다.
- **하이퍼파라미터 의존성**: $\alpha, \beta, \gamma$와 같은 가중치 파라미터가 성능에 영향을 줄 수 있으나, 이에 대한 정밀한 튜닝 과정에 대한 상세 설명은 부족하다.

## 📌 TL;DR

본 논문은 온디바이스 키워드 검출(KWS)을 위해 거대 S3RL 모델의 지식을 경량 모델로 전이하는 **Dual-View Cross-Correlation** 및 **Teacher Codebook Distillation** 기법을 제안하였다. 이 방법은 데이터 편향 문제를 해결하고 모델 크기를 획기적으로 줄이면서도(최대 98% 감소), 특히 노이즈 환경에서 매우 강건한 성능을 보여준다. 이는 향후 저전력/저사양 기기에서의 고성능 음성 인터페이스 구현에 중요한 기여를 할 것으로 기대된다.
