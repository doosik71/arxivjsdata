# LXMERT: Learning Cross-Modality Encoder Representations from Transformers
Hao Tan, Mohit Bansal

## 🧩 Problem to Solve
시각적 개념, 언어적 의미, 그리고 이 두 양식 간의 정렬 및 관계에 대한 이해를 요구하는 시각-언어 추론(Vision-and-Language Reasoning)에서, 기존의 단일 양식(vision 또는 language) 백본 모델에 대한 대규모 사전 학습은 잘 개발되어 있었으나, 시각과 언어 양식 쌍을 위한 대규모 사전 학습 및 미세 조정 연구는 미흡했습니다. 이 논문은 이러한 교차 양식 연결을 효과적으로 학습하기 위한 프레임워크를 구축하는 것을 목표로 합니다.

## ✨ Key Contributions
*   **LXMERT 프레임워크 제안:** 시각-언어 교차 양식 표현 학습을 위한 새로운 Transformer 기반 프레임워크 LXMERT를 제안합니다. 이 프레임워크는 객체 관계 인코더, 언어 인코더, 교차 양식 인코더의 세 가지 주요 인코더로 구성됩니다.
*   **다양한 사전 학습 태스크 도입:** 시각과 언어 양식 간의 연결을 효과적으로 학습하기 위해 다음 다섯 가지의 독창적이고 다양한 사전 학습 태스크를 도입했습니다.
    *   Masked Cross-Modality Language Modeling
    *   Masked Object Prediction (RoI-Feature Regression 및 Detected-Label Classification)
    *   Cross-Modality Matching
    *   Image Question Answering (QA)
*   **최고 성능 달성:** VQA(Visual Question Answering) 및 GQA(Graphical Question Answering) 두 가지 시각 질의응답 데이터셋에서 최신 기술(State-of-the-Art, SOTA) 결과를 달성했습니다.
*   **일반화 능력 입증:** 도전적인 시각 추론 태스크인 NLVR$_2$에서 이전 최고 기록을 22% 절대치(54%에서 76%) 개선하여 사전 학습된 교차 양식 모델의 뛰어난 일반화 능력을 입증했습니다.
*   **심층적인 분석 및 시각화:** 모델의 새로운 구성 요소와 사전 학습 전략이 강력한 결과에 크게 기여함을 입증하기 위한 상세한 ablation study와 다양한 인코더에 대한 어텐션 시각화를 제공했습니다.

## 📎 Related Works
*   **모델 아키텍처:**
    *   **Bi-directional attention:** 시각-언어 태스크에 양방향 어텐션을 적용한 Lu et al. (2016) 및 독해에 적용한 BiDAF (Seo et al., 2017).
    *   **Transformer:** 기계 번역에서 처음 사용된 Transformer (Vaswani et al., 2017)를 단일 양식 인코더 및 교차 양식 인코더의 기반으로 활용했습니다.
    *   **BUTD (Bottom-Up and Top-Down attention):** 객체 RoI 특징으로 이미지를 임베딩하는 방식 (Anderson et al., 2018)을 객체 위치 임베딩 및 객체 관계 인코더로 확장했습니다.
*   **사전 학습 (Pre-training):**
    *   **단일 양식 언어 모델:** ELMo (Peters et al., 2018), GPT (Radford et al., 2018), BERT (Devlin et al., 2019)와 같은 대규모 사전 학습 언어 모델의 발전을 기반으로 합니다.
    *   **교차 양식 사전 학습:** XLM (Lample and Conneau, 2019)은 교차 언어 표현을 학습하고, VideoBert (Sun et al., 2019)는 언어와 시각 토큰의 연결에 마스킹된 LM을 적용합니다. 이 논문은 기존 BERT 스타일의 토큰 기반 사전 학습 방식과 차별화되는 새로운 모델 아키텍처와 사전 학습 태스크를 제안합니다.
*   **최근 동시 연구:** ViLBERT (Lu et al., 2019) 및 VisualBERT (Li et al., 2019)와 같은 유사한 교차 양식 사전 학습 방향의 동시 연구들과 비교하여, LXMERT는 더 상세한 다중 구성 요소 디자인과 추가적인 사전 학습 태스크(RoI-feature regression, 이미지 QA)를 통해 더 나은 성능을 달성했습니다.

## 🛠️ Methodology
LXMERT는 최근 Transformer 모델의 발전을 따라 Self-Attention 및 Cross-Attention 레이어를 기반으로 구축되었습니다.

### 1. 입력 임베딩 (Input Embeddings)
*   **단어 수준 문장 임베딩 ($h_i$):**
    *   문장을 WordPiece 토크나이저로 단어 ${w_1, ..., w_n}$로 분할합니다.
    *   각 단어 $w_i$와 그 인덱스 $i$ (문장에서의 절대 위치)를 임베딩합니다.
    $$ \hat{w_i} = \text{WordEmbed}(w_i) $$
    $$ \hat{u_i} = \text{IdxEmbed}(i) $$
    $$ h_i = \text{LayerNorm}(\hat{w_i} + \hat{u_i}) $$
*   **객체 수준 이미지 임베딩 ($v_j$):**
    *   Faster R-CNN 객체 탐지기를 사용하여 이미지에서 $m$개의 객체 ${o_1, ..., o_m}$를 탐지합니다.
    *   각 객체 $o_j$는 위치 특징 $p_j$ (경계 상자 좌표)와 2048차원 RoI 특징 $f_j$로 표현됩니다.
    *   위치 인식 임베딩 $v_j$를 학습합니다.
    $$ \hat{f_j} = \text{LayerNorm}(W_F f_j + b_F) $$
    $$ \hat{p_j} = \text{LayerNorm}(W_P p_j + b_P) $$
    $$ v_j = (\hat{f_j} + \hat{p_j}) / 2 $$

### 2. 인코더 (Encoders)
*   **단일 양식 인코더 (Single-Modality Encoders):**
    *   **언어 인코더 (Language Encoder):** $N_L$개의 Transformer 레이어(self-attention + feed-forward)로 구성되며, 언어 입력에 집중합니다.
    *   **객체 관계 인코더 (Object-Relationship Encoder):** $N_R$개의 Transformer 레이어(self-attention + feed-forward)로 구성되며, 시각 입력(객체 특징)에 집중합니다.
*   **교차 양식 인코더 (Cross-Modality Encoder):**
    *   $N_X$개의 교차 양식 레이어를 쌓아서 구현합니다.
    *   각 레이어는 두 개의 self-attention 서브 레이어, 하나의 양방향 cross-attention 서브 레이어 (언어-시각, 시각-언어), 그리고 두 개의 feed-forward 서브 레이어로 구성됩니다.
    *   양방향 cross-attention은 두 양식 간의 정보 교환 및 엔티티 정렬을 통해 공동 교차 양식 표현을 학습합니다.

### 3. 출력 표현 (Output Representations)
*   LXMERT는 언어, 시각, 교차 양식의 세 가지 출력을 가집니다.
*   언어 및 시각 출력은 교차 양식 인코더에 의해 생성된 특징 시퀀스입니다.
*   교차 양식 출력은 BERT와 유사하게 문장 단어 앞에 추가된 특별 토큰 `[CLS]`의 해당 특징 벡터를 사용합니다.

### 4. 사전 학습 태스크 (Pre-Training Tasks)
*   **Masked Cross-Modality LM:**
    *   BERT와 유사하게 단어의 15%를 무작위로 마스킹하고, 모델은 마스킹된 단어를 예측합니다.
    *   기존 BERT와 달리, LXMERT는 시각 양식에서도 마스킹된 단어를 예측하여 모호성을 해소하고 시각-언어 연결을 강화합니다.
*   **Masked Object Prediction:**
    *   객체의 15%를 무작위로 마스킹(RoI 특징을 0으로 설정)하고, 모델은 마스킹된 객체의 속성을 예측합니다.
    *   **RoI-Feature Regression:** $L_2$ 손실을 사용하여 객체 RoI 특징 $f_j$를 회귀합니다.
    *   **Detected-Label Classification:** 교차 엔트로피 손실을 사용하여 마스킹된 객체의 탐지된 레이블을 분류합니다.
*   **Cross-Modality Matching:**
    *   각 문장의 50%를 일치하지 않는 다른 문장으로 대체합니다.
    *   이미지와 문장이 서로 일치하는지 예측하는 분류기를 훈련합니다.
*   **Image Question Answering (QA):**
    *   사전 학습 데이터의 약 1/3을 이미지에 대한 질문으로 구성합니다.
    *   이미지와 질문이 일치할 때 이 이미지 관련 질문에 대한 답을 예측하도록 모델을 학습합니다.

### 5. 사전 학습 데이터 및 절차
*   **데이터:** MS COCO, Visual Genome (VG), VQA v2.0, GQA, VG-QA 등 5개 데이터셋에서 총 9.18M 개의 이미지-문장 쌍을 수집했습니다 (180K 고유 이미지, 약 100M 단어, 6.5M 이미지 객체).
*   **절차:**
    *   Adam 옵티마이저와 선형 감소 학습률 스케줄을 사용합니다 (최대 학습률 $1e^{-4}$).
    *   20 epoch 동안 모델을 사전 학습합니다 (처음 10 epoch는 QA 태스크 없이, 마지막 10 epoch는 QA 태스크 포함).
    *   총 9500개의 정답 후보를 가진 공동 정답 테이블을 사용합니다.
    *   Faster R-CNN 탐지기는 고정된 특징 추출기로 사용하며, 각 이미지에 대해 36개의 객체를 유지합니다.
    *   인코더 및 임베딩 레이어의 모든 파라미터는 무작위로 초기화됩니다 (일부 실험에서는 BERT 파라미터 로드).
    *   $N_L=9, N_X=5, N_R=5$로 레이어 수를 설정하고, hidden size는 768입니다.

## 📊 Results
LXMERT는 다양한 벤치마크에서 SOTA 성능을 달성했습니다.

*   **VQA v2.0 (test-standard):**
    *   전체 정확도(Accu): **72.5%** (이전 SOTA 대비 2.1%p 개선).
    *   'Binary', 'Other' 질문 하위 범주에서 2.4%p 개선을 보였습니다.
*   **GQA (test-standard):**
    *   전체 정확도(Accu): **60.3%** (이전 SOTA 대비 3.2%p 개선).
    *   'Open' 도메인 질문에서 4.6%p 개선을 달성했습니다.
*   **NLVR$_2$ (Test-U):**
    *   정확도(Accu): **76.2%** (이전 SOTA 대비 22%p 절대 개선, 48%p 상대 오차 감소).
    *   일관성(Cons): **42.1%** (이전 SOTA 대비 30%p 절대 개선, 3.5배 증가).
*   **Ablation Study:**
    *   BERT 기반 모델만으로는 LXMERT 사전 학습 없이 NLVR$_2$에서 약 22%p 낮은 성능을 보였습니다.
    *   이미지 QA 사전 학습 태스크는 모든 데이터셋에서 성능을 향상시켰습니다 (NLVR$_2$에서 2.1%p 개선). 이는 사전 학습에 사용되지 않은 NLVR$_2$ 데이터에서 더욱 강력한 표현을 학습했음을 보여줍니다.
    *   RoI-Feature Regression 및 Detected-Label Classification을 포함한 비전 사전 학습 태스크 모두 개별적으로 기여하며, 함께 사용했을 때 최고의 결과를 얻었습니다.

## 🧠 Insights & Discussion
*   **교차 양식 사전 학습의 중요성:** 이 논문은 시각-언어 태스크를 위한 대규모 사전 학습 프레임워크의 필요성과 효과를 강력하게 입증했습니다. 특히, BERT와 같은 언어 전용 사전 학습 모델로는 복잡한 시각-언어 추론 태스크에서 충분한 성능을 내기 어렵고, LXMERT의 교차 양식 사전 학습이 필수적임을 보여주었습니다.
*   **모델 구성 요소의 시너지:** LXMERT의 세 가지 인코더(언어, 객체 관계, 교차 양식)의 설계와 5가지의 다양한 사전 학습 태스크가 서로 시너지를 발휘하여 강력한 성능을 달성했습니다. 특히, 이미지 QA 태스크와 객체 마스킹 예측 태스크는 시각-언어 정렬 및 추론 능력 학습에 결정적인 역할을 했습니다.
*   **일반화 능력:** 사전 학습 시 사용되지 않은 NLVR$_2$와 같은 도전적인 시각 추론 태스크에서 큰 성능 향상을 보이며 모델의 강력한 일반화 능력을 입증했습니다. 이는 LXMERT가 단순히 특정 데이터셋에 과적합된 것이 아니라, 시각과 언어 간의 심층적인 관계를 학습했음을 시사합니다.
*   **어텐션 메커니즘의 해석:** 어텐션 시각화를 통해 언어 인코더가 BERT와 유사한 언어적 패턴(다음 단어, 이전 단어)을 학습하고, 객체 관계 인코더가 이미지 내 객체 간의 의미 있는 관계(장면 그래프)를 구축하며, 교차 양식 인코더가 대명사, 명사, 관사 등 정보성 높은 단어에 집중하여 시각적 정보를 텍스트에 연결하는 방식을 보여주었습니다.
*   **제한 및 향후 연구:** 모델은 Faster R-CNN에 의해 탐지된 객체 특징에 의존하므로, 탐지기의 성능에 영향을 받습니다. 향후 연구로는 이미지와 텍스트 간의 명사-명사, 명사-동사 관계를 직접 포착하는 사전 학습 태스크 활용이 고려될 수 있습니다.

## 📌 TL;DR
LXMERT는 Transformer 기반의 새로운 교차 양식 인코더 프레임워크로, 시각(객체 관계 인코더), 언어(언어 인코더), 그리고 두 양식 간의 상호작용(교차 양식 인코더)을 통합적으로 학습합니다. Masked Language Modeling, Masked Object Prediction, Cross-Modality Matching, Image QA 등 5가지의 다양한 사전 학습 태스크를 통해 대규모 이미지-문장 쌍 데이터셋에서 모델을 사전 학습했습니다. 결과적으로, VQA와 GQA에서 SOTA를 달성하고, NLVR$_2$에서는 이전 최고 기록을 22%p 개선하며 시각-언어 추론 태스크에서 탁월한 성능과 일반화 능력을 입증했습니다. 이는 교차 양식 사전 학습의 중요성과 LXMERT의 효과적인 아키텍처 및 학습 전략을 보여줍니다.