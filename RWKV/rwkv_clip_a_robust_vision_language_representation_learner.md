# RWKV-CLIP: A Robust Vision-Language Representation Learner

Tiancheng Gu, Kaicheng Yang, Xiang An, Ziyong Feng, Dongnan Liu, Weidong Cai, Jiankang Deng (2024)

## 🧩 Problem to Solve

본 논문은 대규모 웹 데이터셋을 활용하는 Contrastive Language-Image Pre-training (CLIP)의 두 가지 핵심적인 한계점을 해결하고자 한다.

첫째는 **데이터의 품질 문제**이다. 웹에서 수집된 이미지-텍스트 쌍은 노이즈가 많고, 텍스트 표현이 추상적이거나 이미지와 텍스트 간의 세만틱 불일치(semantic discrepancies)가 발생하는 경우가 많다. 기존의 합성 캡션 생성 모델(예: OFA)은 훈련 데이터의 한계로 인해 거친 수준(coarse-grained)의 객체 범주만 식별할 수 있다는 한계가 있다.

둘째는 **모델 아키텍처의 효율성 문제**이다. 현재 대부분의 시각-언어 표현 학습 모델은 Transformer 아키텍처를 기반으로 한다. 그러나 Transformer의 내재적인 2차 복잡도(quadratic computational complexity)는 고해상도 이미지나 긴 시퀀스를 처리할 때 계산 비용을 급격히 증가시켜 범용적인 적용에 제약을 준다.

따라서 본 연구의 목표는 LLM을 활용해 고품질의 정교한 텍스트 설명을 생성하는 프레임워크를 구축하고, Transformer의 병렬 학습 효율성과 RNN의 추론 효율성을 동시에 갖춘 RWKV 아키텍처를 시각-언어 모델에 최초로 도입하여 강건하고 효율적인 표현 학습기를 구현하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 데이터 정제 파이프라인과 새로운 모델 아키텍처의 결합에 있다.

1. **다양한 설명 생성 프레임워크(Diverse Description Generation Framework)**: 웹 기반의 원본 텍스트(raw text), 합성 캡션(synthetic captions), 그리고 RAM++를 통한 개방형 세그먼트 탐지 태그(detection tags)를 LLM(LLaMA3)으로 통합 및 정제하여, 단순한 캡션을 넘어 세밀하고 정확한 의미 정보를 담은 설명을 생성한다.
2. **RWKV-CLIP 제안**: RNN의 효율적인 추론 속도와 Transformer의 병렬 학습 능력을 결합한 RWKV 아키텍처를 시각-언어 표현 학습에 최초로 적용하였다.
3. **강건성 및 효율성 입증**: 다양한 모델 규모와 데이터셋에 걸친 광범위한 실험을 통해, RWKV-CLIP이 기존 ViT 기반의 CLIP 모델들보다 우수한 성능과 강건성을 가짐을 증명하였다.

## 📎 Related Works

### 시각-언어 표현 학습 (Vision-Language Representation Learning)

CLIP 이후 SLIP, DeCLIP, FILIP, UniCLIP, HiCLIP, ALIP 등 다양한 개선 연구가 진행되었다. 이들은 주로 대조 학습(contrastive loss)의 효율성을 높이거나, 다중 뷰 감독, 계층적 어텐션, 합성 데이터 활용 등을 통해 성능을 향상시켰다. 본 논문은 이러한 기존 연구들이 주로 데이터 필터링이나 손실 함수 수정에 집중한 것과 달리, 데이터 생성 단계의 LLM 활용과 모델의 근본적인 아키텍처 변경(RWKV)을 동시에 시도했다는 점에서 차별화된다.

### 텍스트 증강 (Text Augmentation)

LaCLIP이나 CapsFusion과 같이 LLM을 사용하여 텍스트 설명을 재작성하는 시도가 있었다. 그러나 LLM의 환각(hallucination) 문제와 제한된 샘플 의존성으로 인해 여전히 노이즈가 발생할 가능성이 컸다. 본 논문은 이를 해결하기 위해 RAM++의 탐지 태그를 도입하여 LLM이 이미지의 구체적인 객체 정보를 바탕으로 텍스트를 생성하도록 제약함으로써 환각 현상을 완화하였다.

### Receptance Weighted Key Value (RWKV)

RWKV는 NLP 분야에서 Transformer의 메모리 병목과 계산 복잡도를 해결하기 위해 제안되었다. 최근 Vision-RWKV, PointRWKV, Diffusion-RWKV 등이 등장하며 시각적 인지 및 생성 작업에서 ViT보다 효율적임이 입증되었으나, 시각-언어 표현 학습(VL representation learning) 분야에서의 잠재력은 아직 검증되지 않은 상태였다.

## 🛠️ Methodology

### 1. 다양한 설명 생성 (Diverse Description Generation)

노이즈가 많은 웹 데이터셋을 개선하기 위해 다음과 같은 파이프라인을 구축한다.

- **입력 데이터 수집**: 원본 텍스트($T_r$), OFA-base 모델로 생성한 합성 캡션($T_s$), 그리고 RAM++ 모델로 추출한 세밀한 탐지 태그(detection tags)를 준비한다.
- **LLM 기반 정제**: 초기에는 ChatGPT-3.5-turbo를 사용하여 이 세 가지 정보를 통합하는 지시어(instruction) 데이터셋을 구축한다.
- **모델 최적화**: 구축된 데이터셋으로 LLaMA3-8B 모델을 미세 조정(fine-tuning)하여, 대규모 데이터셋에 대해 효율적이고 정확한 '다양한 설명($T_g$)'을 추론 생성한다.

### 2. RWKV-CLIP 아키텍처

RWKV-CLIP은 이미지 인코더($E_I$)와 텍스트 인코더($E_T$)로 구성된 듀얼 타워 구조를 가진다. 각 인코더는 Spatial Mixing과 Channel Mixing 블록의 스택으로 이루어져 있다.

#### 2.1 입력 증강 (Input Augmentation)

모델의 강건성을 위해 텍스트 입력 시 원본 텍스트, 합성 캡션, 생성된 설명 중 하나를 무작위로 선택하여 사용한다.
$$\text{aug}(T) = \text{Sample}([T_r, T_s, T_g])$$

#### 2.2 Spatial Mixing (공간 믹싱)

선형 복잡도의 글로벌 어텐션을 수행하며, $\text{Lerp}$(Linear Interpolation)를 통해 4개의 병렬 선형 층에서 $G^s_x, R^s_x, K^s_x, V^s_x$ 벡터를 얻는다.

- **Q-Lerp (Image)**: 이미지의 4방향 시프트 벡터를 사용하여 특징 상호작용을 강화한다.
- **B-Lerp (Text)**: 텍스트의 양방향 시프트 벡터를 사용하여 추가 연산 비용 없이 전후 토큰 간 상호작용을 보장한다.

시간에 따라 변하는 감쇠 계수(time-varying decay) $w^s_x$는 다음과 같이 계산된다.
$$\phi(x) = \lambda + \tanh(x \cdot M_i) \cdot M_j$$
$$\hat{w}^s_x = x + (1 - \phi(\text{Lerp}_w(x))) \cdot x^\star$$
$$w^s_x = \exp(-\exp(\tilde{w}^s_x))$$

최종적으로 Bi-WKV 메커니즘을 통해 양방향 어텐션 결과 $wkv_t$를 계산하고, 게이트 메커니즘 $\sigma(G^s_x)$를 곱해 출력 $O^s_x$를 산출한다.

#### 2.3 Channel Mixing (채널 믹싱)

Spatial Mixing 이후 수행되며, $\text{Lerp}$를 통해 $R^c_x, K^c_x$를 얻고 선형 투영과 게이트 메커니즘을 거쳐 최종 출력을 생성한다.
$$O^c_x = (\sigma(R^c_x) \odot \rho(K^c_x)) \cdot w^c_o$$
여기서 $\rho$는 squaredReLU 함수이다.

#### 2.4 학습 목표 및 손실 함수

이미지 임베딩 $\hat{I}$와 텍스트 임베딩 $\hat{T}$ 사이의 거리를 최소화하기 위해 대칭적 교차 엔트로피 기반의 대조 손실(Contrastive Loss) 함수를 사용한다.
$$L = -\sum_{i=1}^{N} \left[ \log \frac{e^{\hat{I}_i^\top \hat{T}_i / \tau}}{\sum_{j} e^{\hat{I}_i^\top \hat{T}_j / \tau}} + \log \frac{e^{\hat{I}_i^\top \hat{T}_i / \tau}}{\sum_{j} e^{\hat{I}_j^\top \hat{T}_i / \tau}} \right]$$

## 📊 Results

### 실험 설정

- **데이터셋**: YFCC15M 및 LAION400M의 부분 집합(10M, 30M)을 사용하였다.
- **비교 모델**: CLIP-ViT-B/32, DeCLIP, HiCLIP, ALIP 등.
- **평가 지표**: Linear Probe Accuracy, Zero-shot Image-Text Retrieval (Recall@1, 5, 10), Zero-shot Classification Accuracy.

### 주요 결과

1. **Linear Probe**: 10개의 하위 데이터셋에서 평균 1.9%~11.1%의 성능 향상을 보였으며, 특히 10개 중 8개 데이터셋에서 ALIP를 능가하였다.
2. **Zero-shot 이미지-텍스트 검색**: Flickr30k와 MSCOCO 모두에서 SOTA를 달성하였다. 특히 Flickr30k의 R@1 지표에서 I2T(이미지$\rightarrow$텍스트)는 76.0%, T2I(텍스트$\rightarrow$이미지)는 57.6%를 기록하여 ALIP 대비 각각 5.5%, 8.7% 향상되었다.
3. **Zero-shot 분류**: 11개 데이터셋에서 평균 2.6%~14.4% 향상되었으며, 특히 Food101과 ImageNet 같은 인스턴스 판별 데이터셋에서 큰 폭의 개선이 있었다.
4. **강건성 평가**: ImageNet-V2, ImageNet-A 등 강건성 테스트셋에서도 ALIP 대비 평균 2.0% 높은 성능을 보여 RWKV 기반 모델의 강건함을 입증하였다.

### 절제 연구 (Ablation Study)

- **데이터 및 모델 스케일링**: LAION 10M, 30M 실험 결과, 모델 규모와 데이터 양이 증가함에 따라 RWKV-CLIP이 일관되게 우수한 성능을 유지함을 확인하였다.
- **텍스트 유형별 영향**: 합성 캡션($T_s$)과 생성된 설명($T_g$)을 사용했을 때 원본 텍스트($T_r$)보다 Linear Probe 성능이 높았다. 이는 원본 데이터의 미스매치(mismatch) 문제를 해결했음을 의미한다.
- **아키텍처 조합**: RWKV 이미지 인코더와 RWKV 텍스트 인코더를 함께 사용했을 때 가장 높은 성능을 보였으며, RWKV와 Transformer를 혼합했을 때는 호환성 문제로 성능이 하락하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문의 가장 큰 성과는 **데이터 정제와 모델 아키텍처의 시너지**를 확인한 것이다. UMAP 시각화 결과, RWKV-CLIP은 ALIP보다 이미지-텍스트 모달리티 간의 거리가 더 가깝고, 동일 모달리티 내에서의 변별력이 더 뛰어남을 보였다. 이는 RWKV 아키텍처가 시각-언어 표현 학습에서 Transformer보다 더 효율적인 교차 모달 정렬(cross-modal alignment)을 수행할 수 있음을 시사한다.

또한, 탐지 태그(detection tags)를 LLM의 입력으로 제공함으로써 LLM이 이미지에 없는 내용을 생성하는 환각 현상을 효과적으로 억제하고, 더욱 세밀한 시각적 정보를 텍스트에 반영할 수 있었다.

### 한계 및 미해결 질문

1. **외부 모델 의존성**: 제안된 프레임워크가 기존의 캡션 생성 모델(OFA)과 탐지 태그 모델(RAM++)에 의존하고 있어, 이들 모델의 품질이 최종 결과물에 직접적인 영향을 미친다.
2. **데이터 규모의 제한**: 계산 자원의 한계로 인해 수천만 단위의 데이터셋까지만 실험이 진행되었다. 실제 CLIP과 같은 수십억 단위(billion-scale) 데이터셋에서도 동일한 효율성과 성능 향상이 나타날지는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 LLM과 탐지 태그를 결합해 웹 데이터의 노이즈를 제거하는 **다양한 설명 생성 프레임워크**와, Transformer의 학습 효율과 RNN의 추론 효율을 결합한 **RWKV-CLIP 아키텍처**를 제안한다. 실험 결과, RWKV-CLIP은 기존 ViT 기반 모델들보다 선형 프로빙, 제로샷 분류 및 검색 작업에서 우수한 성능과 강건성을 보였으며, 특히 효율적인 연산 비용으로 고성능의 시각-언어 표현 학습이 가능함을 입증하였다. 이는 향후 초거대 시각-언어 모델의 효율적인 학습 및 추론 방향에 중요한 통찰을 제공한다.
