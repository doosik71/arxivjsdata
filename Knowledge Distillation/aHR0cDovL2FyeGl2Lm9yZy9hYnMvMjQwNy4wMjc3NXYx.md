# MLKD-BERT: Multi-level Knowledge Distillation for Pre-trained Language Models

Ying Zhang, Ziheng Yang, Shufan Ji (2024)

## 🧩 Problem to Solve

본 논문은 거대 사전 학습 언어 모델(Pre-trained Language Models, PLMs), 특히 BERT와 같은 모델이 가지는 방대한 파라미터 수와 긴 추론 시간으로 인해 자원이 제한된 환경이나 실시간 시나리오에서 사용하기 어렵다는 문제를 해결하고자 한다. 이를 위해 모델 압축 기법인 지식 증류(Knowledge Distillation, KD)를 활용하지만, 기존의 KD 방법론들은 다음과 같은 두 가지 한계점을 가지고 있다.

첫째, 기존 방식들은 주로 Feature-level 지식(특징 수준의 지식)을 증류하는 데 집중하며, 토큰 간의 관계나 샘플 간의 관계와 같은 Relation-level 지식(관계 수준의 지식)을 충분히 탐색하지 않는다. 이러한 관계 지식은 학생 모델의 성능을 향상시키는 데 매우 중요한 정보가 될 수 있다.

둘째, 대부분의 기존 연구는 교사 모델의 Self-attention 분포를 그대로 증류하려 하기 때문에, 학생 모델이 교사 모델과 동일한 수의 Attention head를 가져야만 하는 제약이 있다. 이는 학생 모델의 Attention head 수를 줄여 추론 시간을 단축하려는 시도를 방해한다.

따라서 본 연구의 목표는 관계 수준의 지식을 추가로 학습하고, Attention head 수를 유연하게 설정할 수 있도록 하여 성능 저하를 최소화하면서 추론 속도를 획기적으로 높이는 새로운 지식 증류 방법인 MLKD-BERT를 제안하는 것이다.

## ✨ Key Contributions

MLKD-BERT의 핵심 아이디어는 Feature-level 지식과 Relation-level 지식을 모두 포함하는 다층적(Multi-level) 지식을 두 단계(Two-stage)에 걸쳐 증류하는 것이다. 

가장 중심적인 설계는 단순히 Attention 분포를 복제하는 것이 아니라, Attention 출력 벡터 간의 유사도 관계(Self-attention relation)를 학습하도록 설계한 점이다. 이를 통해 학생 모델의 Attention head 수가 교사 모델과 다르더라도 지식 전이가 가능하게 하여, 모델의 경량화와 추론 속도 최적화를 유연하게 수행할 수 있게 하였다. 또한, 임베딩 층에서의 토큰 유사도와 예측 층에서의 샘플 간 유사도 및 대조 관계를 도입하여 모델이 데이터의 구조적 관계를 더 잘 이해하도록 유도하였다.

## 📎 Related Works

기존의 BERT 압축 연구로는 DistilBERT, BERT-PKD, TinyBERT, MobileBERT, MiniLM 등이 있다. DistilBERT는 소프트 타겟 확률과 임베딩 출력을 사용하며, BERT-PKD는 중간 레이어에서 점진적으로 지식을 추출한다. TinyBERT와 MobileBERT는 임베딩 층과 Self-attention 분포 등 더 내부적인 표현을 증류하여 성능을 높였다. MiniLM 및 MiniLMv2는 Self-attention 관계를 활용한 깊은 증류 방식을 제안하였다.

그러나 이러한 기존 방식들은 앞서 언급한 바와 같이 관계 수준의 지식 활용이 부족하며, 특히 학생 모델의 Attention head 수를 교사 모델과 동일하게 유지해야 한다는 제약이 있어 추론 효율성을 극대화하는 데 한계가 있다. MLKD-BERT는 이러한 제약을 해소하고 관계 지식을 다각도로 활용함으로써 기존 방식들과 차별화된다.

## 🛠️ Methodology

MLKD-BERT는 다운스트림 작업 예측을 위해 두 단계의 증류 절차를 거친다. 학생 모델의 레이어 수는 교사 모델보다 적으므로, TinyBERT에서 제안된 균등 매핑(Uniform mapping) 전략을 사용하여 학생 레이어를 교사 레이어에 대응시킨다.

### Stage 1: Embedding 및 Transformer-layer 증류
첫 번째 단계는 특징 표현과 변환 능력에 집중하며, 임베딩 층과 Transformer 층을 증류한다.

**1. Embedding-layer Distillation**
토큰 간의 유사도 관계를 학습한다. 교사와 학생의 토큰 임베딩 유사도 행렬 간의 KL-divergence를 최소화한다.
$$L_{EMB} = \frac{1}{|x|} \sum_{i=1}^{|x|} D_{KL}(R^T_i || R^S_i)$$
여기서 $R^T$와 $R^S$는 각각 교사와 학생의 토큰 임베딩 유사도 행렬이며, 다음과 같이 계산된다.
$$R^T = \text{softmax}\left(\frac{E^T {E^T}^T}{\sqrt{d^T_h}}\right), \quad R^S = \text{softmax}\left(\frac{E^S {E^S}^T}{\sqrt{d^S_h}}\right)$$

**2. Transformer-layer Distillation**
MHA(Multi-Head Attention)와 FFN(Feed Forward Network) 서브 레이어를 각각 다르게 증류한다.

- **MHA Distillation**: Attention head 수의 유연성을 위해 'MHA-split' 개념을 도입한다. 교사의 Attention head들을 학생의 head 수($A_s$)에 맞춰 그룹화하여 분할하고, 각 분할 내의 출력 벡터 유사도 관계를 증류한다.
$$L_{MHA} = \frac{1}{A_s |x|} \sum_{n=1}^{N} \sum_{a=1}^{A_s} \sum_{i=1}^{|x|} D_{KL}(R^T_{m,a,i} || R^S_{n,a,i})$$
이 방식을 통해 학생 모델은 교사보다 적은 수의 Attention head를 가져도 관계 지식을 학습할 수 있다.

- **FFN Distillation**: 특징 수준의 지식을 전이하기 위해 학생과 교사의 출력 은닉 상태(Hidden states) 간의 평균 제곱 오차(MSE)를 최소화한다.
$$L_{FFN} = \sum_{n=1}^{N} \text{MSE}(H^S_n W_h, H^T_m)$$
여기서 $W_h$는 학생의 차원을 교사의 차원으로 맞추기 위한 학습 가능한 선형 변환 행렬이다.

Stage 1의 최종 손실 함수는 $L_{\text{Stage 1}} = L_{EMB} + L_{MHA} + L_{FFN}$이다.

### Stage 2: Prediction-layer 증류
두 번째 단계는 샘플 예측 능력에 집중하며, 샘플 간의 관계를 활용한다.

**1. Sample Similarity Relation ($L_{SS}$)**
배치 내 샘플 간의 유사도 관계를 학습한다. $[CLS]$ 토큰의 출력 벡터를 사용하여 샘플 유사도 행렬 $R$을 생성하고 KL-divergence를 최소화한다.
$$L_{SS} = \frac{1}{b} \sum_{i=1}^{b} D_{KL}(R^T_i || R^S_i)$$

**2. Sample Contrastive Relation ($L_{SC}$)**
같은 클래스의 샘플은 가깝게, 다른 클래스의 샘플은 멀게 배치하도록 InfoNCE 손실 함수를 사용한다.
$$L_{SC} = \frac{1}{2b} \sum_{i=1}^{2b} \sum_{i \in I} \frac{1}{|P(i)|} \sum_{p \in P(i)} L_{\text{InfoNCE}}(i, p)$$

**3. Soft Label Distillation ($L_{KD}$)**
기존의 KD 방식과 같이 교사와 학생의 로짓(Logits) 값 사이의 KL-divergence를 최소화한다.
$$L_{KD} = D_{KL}(\text{softmax}(z^T/\tau) || \text{softmax}(z^S/\tau))$$

Stage 2의 최종 손실 함수는 $L_{\text{Stage 2}} = L_{SS} + L_{SC} + L_{KD}$이다.

## 📊 Results

### 실험 설정
- **데이터셋**: GLUE 벤치마크(8개 태스크) 및 추출적 질의응답(Extractive QA) 태스크인 SQuAD 1.1, 2.0을 사용하였다.
- **모델 구성**: 교사 모델은 BERT-base(109M 파라미터)를 사용하였고, 학생 모델은 4개 레이어($\text{MLKD-BERT}_4$, 14.5M)와 6개 레이어($\text{MLKD-BERT}_6$, 67.0M) 두 가지 버전을 구축하였다.
- **비교 대상**: BERT-PKD, DistilBERT, BERT-EMD, TinyBERT, MiniLMv2 등 최신 KD 방법론들과 비교하였다.

### 주요 결과
- **성능 우위**: GLUE 벤치마크에서 $\text{MLKD-BERT}_4$는 4개 레이어 모델 중 평균 성능 1위를 기록하였으며, $\text{MLKD-BERT}_6$ 역시 경쟁 모델 대비 우수한 성능을 보였다. 특히 SQuAD 태스크에서도 모든 베이스라인 모델을 능가하는 결과를 보였다.
- **압축 효율성**: $\text{MLKD-BERT}_6$는 파라미터 수와 추론 시간을 50%로 줄였음에도 불구하고 교사 모델 성능의 평균 99.5%를 유지하였다.
- **추론 속도와 Head 수의 관계**: Attention head 수를 12개에서 3개로 줄였을 때, 성능 하락은 매우 적은 반면(약 1~2% 내외), 추론 시간은 배치 사이즈가 클수록 더 드라마틱하게 감소함을 확인하였다(예: MNLI-m 태스크, 배치 64에서 약 14.11% 감소).
- **Ablation Study**: $L_{FFN}, L_{MHA}, L_{SC}, L_{SS}, L_{EMB}$ 순으로 중요도가 높게 나타났으며, 모든 손실 함수가 성능 향상에 기여함을 확인하였다. 또한, One-stage보다 Two-stage 증류 방식이 더 높은 성능을 보였는데, 이는 표현 학습(Stage 1)과 예측 학습(Stage 2)의 목표를 분리하여 강조했기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 Feature-level 지식만으로는 부족하며, Relation-level 지식이 모델의 성능을 보완하는 상호 보완적 역할을 한다는 점을 입증하였다. 특히 임베딩, MHA, 샘플 수준에서 다각도로 관계를 정의하고 이를 전이함으로써 학생 모델이 교사의 추론 메커니즘을 더 깊게 모방할 수 있게 하였다.

또한, MHA-split 설계를 통해 학생 모델의 Attention head 수를 유연하게 조정할 수 있게 한 점은 실무적으로 매우 중요한 기여이다. 이는 성능과 속도 사이의 Trade-off를 사용자가 직접 제어할 수 있게 하여, 하드웨어 제약 조건에 맞는 최적의 모델을 구성할 수 있게 한다.

다만, 두 단계로 나누어 학습하는 방식은 단일 단계 학습보다 더 많은 훈련 시간을 소모한다는 한계가 있다. 또한 본 연구는 자연어 이해(NLU) 태스크에 한정되어 있어, 생성(Generation) 태스크로의 확장 가능성에 대해서는 추가적인 검증이 필요하다.

## 📌 TL;DR

MLKD-BERT는 Feature-level과 Relation-level 지식을 모두 증류하는 2단계 프레임워크를 통해 BERT를 압축하는 방법론이다. 특히 MHA-split 구조를 도입하여 학생 모델의 Attention head 수를 자유롭게 설정할 수 있게 함으로써, 성능 저하를 최소화하면서도 추론 속도를 획기적으로 개선하였다. 이 연구는 향후 Transformer 기반 PLM의 효율적인 압축 및 실시간 배포를 위한 중요한 가이드라인을 제공한다.