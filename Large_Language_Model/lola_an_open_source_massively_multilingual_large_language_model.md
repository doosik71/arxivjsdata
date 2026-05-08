# LOLA – An Open-Source Massively Multilingual Large Language Model

Nikit Srivastava, Denis Kuchelev, Tatiana Moteu Ngoli, Kshitij Shetty, Michael Röder, Hamada M. Zahera, Diego Moussallem, Axel-Cyrille Ngonga Ngomo (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)의 다국어 확장성 및 접근성 문제를 해결하고자 한다. 현재 대부분의 고성능 LLM은 영어 중심적이며, 다국어 능력을 갖추기 위해서는 모델의 용량을 대폭 확장해야만 성능 저하를 막을 수 있는 '다국어의 저주(curse of multilinguality)' 현상이 발생한다. 또한, 많은 최신 모델들이 폐쇄적인 라이선스 정책을 취하거나 훈련 데이터 및 상세 구현 내용을 공개하지 않아 학술적 재현성과 연구 접근성이 낮다는 문제가 있다. 따라서 본 연구의 목표는 효율적인 아키텍처를 통해 160개 이상의 언어를 지원하면서도, 훈련 및 추론 비용을 최적화하고 모델의 가중치와 코드를 완전히 공개하는 오픈소스 다국어 모델 LOLA를 구축하는 것이다.

## ✨ Key Contributions

LOLA의 핵심 아이디어는 sparse Mixture-of-Experts (MoE) 아키텍처를 활용하여 모델의 전체 용량(Capacity)은 크게 늘리되, 실제 연산에 참여하는 활성 파라미터(Active Parameters) 수를 낮게 유지함으로써 효율성과 성능의 균형을 잡는 것이다. 특히, 언어 가족(Language Family) 간의 계통학적(Phylogenetic) 유사성을 MoE의 전문가(Expert) 라우팅 메커니즘이 암시적으로 학습할 수 있다는 직관에 기반하여, 저리소스 언어에 대한 교차 언어 전이 학습(Cross-linguistic transfer learning) 능력을 강화하고자 하였다.

## 📎 Related Works

기존의 다국어 LLM 연구는 mBERT, XLM-R, BLOOM과 같이 단일 모델로 여러 언어를 처리하는 방향으로 발전해 왔다. 최근에는 영어로 사전 훈련된 모델을 다국어로 확장하는 방식(Tower, SeaLLM 등)이 사용되고 있으나, 여전히 언어 간 성능 균형과 훈련 비용의 최적화라는 과제가 남아 있다.

MoE 구조는 파라미터의 일부만 활성화하여 계산 비용을 줄이면서 모델 크기를 확장할 수 있는 대안으로 제시되었으며, GShard, Switch Transformers 등이 대표적이다. 특히 기계 번역(MT) 분야의 NLLB와 같은 모델들은 언어 그룹화를 위해 MoE를 활용한 바 있다. 하지만 일반적인 사전 훈련 기반의 다국어 LLM에서 MoE가 어떻게 언어적 구조를 학습하고 다국어의 저주를 완화하는지에 대한 심층적인 연구는 상대적으로 부족한 실정이다.

## 🛠️ Methodology

### 전체 시스템 구조

LOLA는 GPT 스타일의 Decoder-only Transformer 아키텍처를 기반으로 한다. 총 24개의 디코더 레이어로 구성되며, 표준 Feed-Forward Network(FFN) 레이어를 매 두 번째 레이어마다 MoE 레이어로 교체하여 배치하였다. 모델의 hidden dimension과 embedding dimension은 2048이며, 16개의 attention head를 가진다. 최대 시퀀스 길이는 2048 토큰이다. 전체 파라미터는 7.4B개이지만, sparse 구조 덕분에 토큰당 활성화되는 파라미터는 1.3B개로 제한되어, 연산 비용은 1.3B dense 모델과 유사한 수준이다.

### 라우팅 메커니즘 (Routing Mechanism)

MoE 레이어 내에는 16개의 전문가(Expert)가 존재하며, Switch Transformer의 영향을 받은 top-1 gating 메커니즘을 사용한다. 입력 $x$에 대해 게이팅 가중치 행렬 $W_g$를 곱해 로짓(logit)을 계산하고, Softmax를 통해 각 전문가에 할당될 확률을 구한다.

$$h(x) = W_g \cdot x$$
$$G_i(x) = \frac{\exp(h(x)_i)}{\sum_{j=1}^{N} \exp(h(x)_j)}$$

이후 가장 높은 확률을 가진 전문가 $i^*$를 선택하여 최종 출력을 계산한다.

$$i^* = \arg \max_i G_i(x)$$
$$\text{MoE}(x) = G_{i^*}(x) \cdot E_{i^*}(x)$$

### 학습 목표 및 손실 함수

모델은 Causal Language Modeling(CLM) 태스크로 사전 훈련되었으며, 최종 손실 함수 $L_{\text{final}}$은 교차 엔트로피 손실($L_{\text{CE}}$)과 전문가 간 부하 균형을 맞추기 위한 보조 손실($l_{\text{aux}}$)의 합으로 정의된다.

$$L_{\text{final}} = L_{\text{CE}} + \alpha \cdot l_{\text{aux}}$$

여기서 $\alpha$는 $10^{-2}$로 설정되었으며, 보조 손실 $l_{\text{aux}}$는 각 전문가에게 할당된 가중치의 평균 $P$와 실제 할당된 토큰의 비율 $f$의 내적으로 계산하여 특정 전문가에게만 토큰이 쏠리는 현상을 방지한다.

$$l_{\text{aux}} = N \cdot \sum_{i=1}^{N} P_i \cdot f_i$$

### 학습 절차 및 데이터

CulturaX 데이터셋에서 추출한 167개 언어의 텍스트(약 4,650억 개의 토큰)를 사용하였다. SentencePiece 토크나이저(어휘 사전 크기 100,000)를 사용했으며, 96대의 NVIDIA A100 GPU를 이용하여 약 19일간 훈련을 진행하였다.

## 📊 Results

### 실험 설정

LOLA의 성능을 평가하기 위해 활성 파라미터 수에 따라 17개의 비교 모델을 3가지 카테고리로 분류(K-Means clustering)하였다. 평가는 Q&A, Reasoning, NLI, Reading Comprehension의 4개 영역, 총 13개의 다국어 벤치마크(ARC, MMLU, XNLI, Belebele 등)에서 수행되었다. 측정 지표로는 Accuracy와 Exact Match(MGSM 태스크용)를 사용하였다.

### 주요 결과

1. **모델 크기별 비교**: LOLA는 카테고리 1(소형) 및 카테고리 2(중형) 모델들을 일관되게 압도하는 성능을 보였다. 특히 카테고리 2 모델들은 LOLA보다 활성 파라미터가 평균 2.8배 더 많고 더 많은 토큰으로 학습되었음에도 불구하고 LOLA가 더 우수한 성능을 기록하여, MoE 구조의 효율성을 입증하였다. 다만, 활성 파라미터가 5배 이상 많은 카테고리 3 모델들에 비해서는 성능이 낮았다.
2. **태스크별 성능**:
   - **강점**: NLI, Reasoning, Reading Comprehension 태스크에서 매우 강력한 성능을 보였으며, 특히 NLI에서는 카테고리 3 모델들과 경쟁 가능한 수준의 성능을 나타냈다.
   - **약점**: 팩트 기반의 Q&A 및 수학적 추론(MGSM)에서는 성능이 상대적으로 낮았다. 또한, 2048 토큰이라는 시퀀스 길이 제한으로 인해 Few-shot 설정에서의 성능이 Zero/One-shot 설정보다 떨어지는 경향을 보였다.

## 🧠 Insights & Discussion

### 전문가 라우팅과 언어 계통학의 상관관계

본 논문의 가장 흥미로운 분석은 MoE의 전문가 선택 패턴이 실제 언어 가족(Language Family)의 구조와 상관관계가 있다는 점이다. 언어별 전문가 활성화 벡터를 생성하여 유클리드 거리를 측정하고 이를 URIEL 프로젝트의 계통학적 거리와 비교한 결과, 양의 상관관계가 발견되었다. 특히 학습 데이터가 많은 언어일수록 상관관계가 강하게 나타났으며(최대 0.55), 로망스어군(프랑스어, 스페인어, 이탈리아어 등)이나 게르만어군이 유사한 전문가 집단을 공유하는 경향이 확인되었다. 이는 MoE 구조가 명시적인 지시 없이도 언어 간의 구조적 유사성을 스스로 학습하여 효율적으로 파라미터를 분배하고 있음을 시사한다.

### 비판적 해석 및 한계

LOLA는 계산 효율성 면에서 뛰어나지만, 몇 가지 한계가 명확하다. 첫째, sparse 모델임에도 불구하고 모든 파라미터를 메모리에 적재해야 하므로 동일한 활성 파라미터를 가진 dense 모델보다 더 많은 GPU 메모리를 요구한다. 둘째, 수학 및 코딩 데이터의 부족과 전용 토크나이저의 부재로 인해 산술 연산 능력이 부족하다. 셋째, 2,000 토큰 이하의 짧은 컨텍스트 윈도우는 긴 문서 처리나 복잡한 few-shot 프롬프팅에 제약이 된다.

## 📌 TL;DR

LOLA는 160개 이상의 언어를 지원하는 오픈소스 다국어 LLM으로, 7.4B의 전체 파라미터 중 1.3B만 활성화하는 sparse MoE 아키텍처를 채택하여 효율성을 극대화하였다. 실험 결과, 유사 규모의 dense 모델들을 압도하며 특히 NLI와 추론 태스크에서 강점을 보였다. 또한, 전문가 라우팅 메커니즘이 언어 가족 간의 계통학적 유사성을 반영한다는 것을 밝혀냄으로써, MoE가 다국어 모델의 용량 확장과 전이 학습 효율을 높이는 유효한 수단임을 입증하였다. 이 연구는 향후 계산 자원이 제한된 환경에서 고성능 다국어 모델을 구축하는 데 중요한 기초 자료가 될 것이다.
