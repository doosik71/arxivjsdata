# ViLA: Efficient Video-Language Alignment for Video Question Answering

Xijun Wang, Junbang Liang, Chun-Kai Wang, Kenan Deng, Yu Lou, Ming C. Lin, and Shan Yang (2024)

## 🧩 Problem to Solve

비디오 질의응답(Video Question Answering, VideoQA)은 이미지 질의응답과 달리 추가적인 시간적 차원(temporal dimension)이 존재하여 훨씬 더 도전적인 과제이다. 특히 제한된 컴퓨팅 자원 내에서 비디오의 수많은 프레임 중 질문과 가장 관련이 있는 핵심 프레임을 어떻게 효율적이고 효과적으로 샘플링하느냐가 핵심적인 난제이다.

기존의 많은 비디오-언어 모델들은 사전 학습된 이미지-언어 모델을 기반으로 하며, 비디오 프레임을 무작위 또는 균일하게 샘플링하는 방식을 사용한다. 그러나 이러한 전략은 비디오의 길이가 길거나 정보가 불균일하게 분포된 경우 결정적인 정보를 놓칠 가능성이 크다. 따라서 본 논문은 추론 지연 시간(inference latency)을 줄이면서도, 입력된 질문에 최적화된 핵심 프레임을 선택하여 비디오-언어 정렬(video-language alignment)의 정확도를 높이는 것을 목표로 한다.

## ✨ Key Contributions

본 논문은 효율적인 비디오-언어 정렬을 위해 **ViLA** 네트워크를 제안하며, 핵심 아이디어는 다음과 같다.

1. **Text-guided Frame-Prompter**: 질문 텍스트의 영향을 받아 가장 중요한 프레임을 선택하는 학습 가능한 모듈이다. 가볍게 설계되어 효율성을 유지하면서도 VQA 손실 함수를 통해 텍스트 기반의 기울기(gradient)를 전달받아 질문 관련 프레임을 선택하도록 학습된다.
2. **QFormer-Distiller**: 비디오 정보를 LLM의 입력 도메인으로 효율적으로 전송하기 위해 QFormer 위에 증류(distillation) 메커니즘을 추가한 모듈이다. 교사-학생(Teacher-Student) 구조를 통해 적은 수의 프레임만으로도 풍부한 시각적 정보를 표현할 수 있도록 돕는다.

## 📎 Related Works

### Visual-Language Alignment

최근 BLIP-2와 같은 모델들은 QFormer를 사용하여 동결된(frozen) 시각 인코더와 LLM 사이의 도메인 간극을 메우는 방식을 취하고 있다. 하지만 대부분의 연구가 이미지-텍스트 정렬에 집중되어 있으며, 비디오로의 확장 시 시간적 모델링을 어떻게 효율적으로 수행할 것인가에 대한 논의는 상대적으로 부족했다.

### Image-to-Video Transfer Learning

사전 학습된 이미지 모델(ViT, CLIP 등)의 지식을 비디오 태스크로 전이하려는 시도가 많았다. 최근 SeViLA는 언어 인식 프레임 로컬라이저(language-aware frame localizer)를 제안했으나, 별도의 로컬라이저를 학습시켜야 하므로 실시간 추론에 불리하고 파라미터 수가 증가한다는 한계가 있다.

### Knowledge Distillation 및 Frame Selection

지식 증류는 거대 모델의 지식을 작은 모델로 전이하는 데 널리 사용되어 왔으며, 비디오 QA에서도 고정된 간격의 밀집 샘플링(dense sampling)의 비용 문제를 해결하기 위해 적응형 프레임 샘플링(adaptive frame sampling)이나 어텐션 기반의 핵심 프레임 식별 방식이 연구되어 왔다.

## 🛠️ Methodology

### 전체 시스템 구조

ViLA는 크게 네 가지 구성 요소로 이루어져 있다: 사전 학습된 동결 시각 인코더 $E_v$, Frame-Prompter $F_p$, QFormer-Distiller $Q_d$, 그리고 사전 학습된 동결 LLM이다. 전체 파이프라인은 비디오 프레임을 인코딩한 후, 질문에 최적화된 프레임을 샘플링하고, 이를 QFormer를 통해 LLM이 이해할 수 있는 형태로 변환하여 정답을 생성하는 흐름을 가진다.

### Text-guided Frame-Prompter

프레임 샘플링의 효율성을 위해 설계된 이 모듈은 미분 가능한 프레임 선택기이다.

1. **특징 추출 및 전처리**: 입력 비디오 프레임 $\{f_1, \dots, f_T\}$는 시각 인코더를 통해 특징 $X = \{x_i | x_i = E_v(f_i), i \in [1, T]\}$로 변환된다. 이후 채널별 평균 풀링(mean-pooling)을 수행하여 차원을 축소하고, 데이터를 $S$개의 세그먼트로 나눈다.
2. **학습 가능한 임베딩**: 평균 풀링된 특징은 다음과 같은 FC 레이어와 레이어 정규화(LN)를 거쳐 임베딩 $\hat{x}_i$로 변환된다.
    $$\hat{x}_i = W_2 * \text{ReLU}(\text{LN}(W_1 * \text{Mean}(x_i)))$$
3. **Gumbel-Softmax 기반 선택**: 미분 가능성을 보장하기 위해 Gumbel-Softmax를 사용하여 세그먼트 마스크 $M$을 생성한다. 본 논문은 시간적 다양성을 확보하기 위해 세그먼트별 마스킹(segment-wise masking) 방식을 채택하였다.
4. **텍스트 가이드**: 선택된 프레임($X \cdot M$)과 질문 텍스트 $X_t$는 Cross-Attention을 통해 정렬되며, 최종 VQA 손실 함수 $\mathcal{L}_{vqa}$를 통해 질문과 관련된 프레임을 선택하도록 역전파되는 기울기가 Frame-Prompter에 전달된다.

### QFormer-Distiller

비디오 정보를 LLM 도메인으로 효율적으로 전송하기 위해 교사-학생 구조의 증류 방식을 사용한다.

1. **Teacher-QFormer**: 모든 프레임 특징을 사용하여 넓은 수용 영역(receptive field)에서 학습된다.
2. **Student-QFormer**: Frame-Prompter가 선택한 일부의 핵심 프레임만을 입력으로 받는다.
3. **증류 과정**: 학생 모델의 출력을 디코더 $D$(간순 FC + LN 구조)를 통해 변환하여 교사 모델의 특징과 일치하도록 학습시킨다. 증류 손실 함수는 다음과 같다.
    $$\mathcal{L}_{distill} = \text{MSE}(D(X'_{student}), X'_{teacher})$$
이 과정을 통해 학생 모델은 적은 수의 프레임만으로도 교사 모델이 가진 풍부한 정보량을 모사하게 된다.

### 학습 목표 및 절차

전체 네트워크는 다음 두 가지 손실 함수의 합으로 학습된다.
$$\text{Total Loss} = \mathcal{L}_{distill} + \mathcal{L}_{vqa}$$
여기서 $\mathcal{L}_{vqa}$는 다지선다형 질문에 대한 교차 엔트로피(Cross-Entropy) 손실을 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: NExT-QA, STAR, How2QA, TVQA, VLEP 등 5개 벤치마크를 사용하였다.
- **모델**: 시각 인코더로 ViT-G(1B), LLM으로 Flan-T5 XL(3B)를 사용하였다.
- **지표**: 정답 선택 정확도(Accuracy)와 추론 시간(Inference Time)을 측정하였다.

### 정량적 결과

ViLA는 거의 모든 벤치마크에서 기존 SOTA 모델들을 능가하며, 특히 추론 속도에서 압도적인 성능 향상을 보였다.

- **NExT-QA**: SeViLA 대비 평균 정확도를 1.0% 향상시켰으며, 특히 Temporal 질문에서 3.3% 향상과 함께 $3.04\times$의 속도 향상을 달성하였다.
- **STAR**: Interaction 질문에서 SeViLA 대비 4.6% 향상되었으며, 전체 평균에서 2.2% 높은 정확도를 기록하면서 동시에 $3.04\times$ 빠른 추론 속도를 보였다.
- **VLEP**: 2개의 프레임만 사용한 ViLA가 4개의 프레임을 사용한 SeViLA보다 0.3% 더 높은 성능을 냈으며, 속도는 $4.2\times$ 빨랐다.

### 정성적 결과 및 분석

시각화 분석 결과, ViLA는 SeViLA에 비해 질문의 의도에 더 부합하는 프레임을 정확하게 포착하는 것으로 나타났다. 예를 들어, "자동차와 흙길"에 대한 질문에서 SeViLA는 단순한 "도로"에 집중한 반면, ViLA는 질문에서 요구하는 구체적인 대상이 포함된 프레임을 선택하였다.

### Ablation Study

Frame-Prompter(FP)와 QFormer-Distiller(QFD) 각각의 기여도를 분석한 결과, 두 모듈 모두 성능 향상에 핵심적인 역할을 함이 확인되었다. 특히 STAR 데이터셋에서 QFD는 정확도를 2.9% 높였고, FP는 추가로 1.6%를 높였다. 또한 QFD를 적용했을 때 적은 프레임(4개)을 사용하는 모델의 프레임 선택 결과가 더 많은 프레임(8개)을 사용하는 모델의 선택 결과와 92.3% 일치함을 확인하여, QFD가 핵심 프레임 선택 능력을 강화함을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 연구는 비디오 QA에서 단순히 많은 프레임을 사용하는 것이 아니라, **질문에 기반하여 '무엇을 볼 것인가'를 결정하는 효율적인 샘플링 메커니즘**의 중요성을 증명하였다. 특히 동결된 LLM의 능력을 최대한 활용하면서도, 가벼운 Frame-Prompter와 증류 기반의 QFormer-Distiller를 통해 추론 효율성과 정확도를 동시에 잡았다는 점이 고무적이다.

### 한계 및 논의사항

1. **LLM 크기의 제한**: 리소스 제약으로 인해 파라미터 수가 13B 이하인 LLM에 대해서만 평가가 이루어졌다. 더 거대한 모델에서의 확장성(scalability)에 대한 검증이 필요하다.
2. **긴 비디오 처리**: 본 논문은 효율적인 샘플링을 다루었지만, 매우 긴 비디오 세그먼트에서의 정렬 문제에 대해서는 향후 연구 과제로 남겨두었다.
3. **LLM 파인튜닝**: 본문에서는 LLM을 동결시킨 상태로 성능을 측정했지만, LoRA를 통해 LLM을 파인튜닝했을 때 NExT-QA에서 75.1%의 더 높은 정확도를 기록했다. 이는 ViLA 구조가 LLM의 가중치 업데이트와도 잘 결합될 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 비디오 QA의 효율성을 극대화하기 위해 **질문 기반 프레임 샘플링 모듈(Frame-Prompter)**과 **시각-언어 정렬 강화 모듈(QFormer-Distiller)**을 제안한 ViLA 네트워크를 소개한다. ViLA는 기존 SOTA 모델 대비 훨씬 적은 수의 프레임만을 사용하고도 더 높은 정확도를 기록했으며, 특히 추론 속도를 최대 $4.2\times$까지 향상시켜 실시간 비디오-언어 태스크 적용 가능성을 높였다. 이 연구는 향후 효율적인 비디오 이해 모델 설계 및 거대 언어 모델의 시각적 프롬프팅 연구에 중요한 기초가 될 것으로 보인다.
