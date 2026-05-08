# BOTM: Echocardiography Segmentation via Bi-directional Optimal Token Matching

Zhihua Liu, Lei Tong, Xilin He, Che Liu, Rossella Arcucci, Chen Jin, Huiyu Zhou

## 🧩 해결할 문제

기존 심초음파(echocardiography) 영상 분할(segmentation) 방법들은 심장 구조의 모양 변화(shape variation), 부분적인 관찰(partial observation), 그리고 유사한 강도(intensity)로 인한 영역 모호성(region ambiguity)으로 인해 해부학적 불일치(anatomical inconsistency) 문제를 겪고 있습니다. 이는 낮은 신호 대 잡음비(low signal-to-noise ratio) 조건에서 해부학적으로 결함이 있는 구조를 가진 오탐지 분할(false positive segmentation)을 초래합니다. 이러한 문제들은 부정확한 경계, 모호한 위치, 그리고 위상적 결함(topological defects)으로 나타나며, 정확하고 안정적인 심초음파 분할을 어렵게 합니다.

## ✨ 주요 기여

* 심초음파 분할을 위한 새로운 토큰 매칭 프레임워크인 **BOTM (Bi-directional Optimal Token Matching)**을 제안합니다. 이 방법은 최적 수송(optimal transport)을 통해 암묵적인 해부학적 보존을 보장하는 공동 최적화된 해부학적 일관성 정규화(consistency regularization)를 포함합니다.
* 토큰 매칭을 양방향 교차 수송 어텐션 프록시(bi-directional cross-transport attention proxy)로 확장하여 공간 및 시간 도메인 모두에서 해부학적 사전 정보(anatomical priors)를 통합할 수 있도록 합니다.
* 광범위한 실험을 통해 BOTM이 다양한 시나리오에서 강력한 일반화 능력(generalization)과 높은 해석 가능성(interpretation)을 갖춘 우수하고 안정적인 분할 결과를 달성함을 입증합니다.

## 📎 관련 연구

* **기존 심초음파 분할 방법:** UNet [26], 비전 트랜스포머(Vision Transformer) [8] 및 그 변형 [2, 5, 11, 22, 30, 31]에 기반하며, 단일 프레임에 과적합되어 해부학적 불일치를 야기할 수 있습니다.
* **시간 정보 통합 방법:** 일부 연구 [1, 7, 28]는 단순한 프레임 추가 또는 연결을 통해 시간 정보를 통합하려 했으나, 해부학적 및 구조적 무결성(integrity)을 보존하는 메커니즘이 부족합니다.
* **SAM 기반 접근법:** Segment Anything Model (SAM) [14] 위에 도메인별 어댑터 모듈 [10, 17]을 도입했지만, 추가적인 계산 복잡성을 야기하고 적절한 마스크 프롬프트에 크게 의존합니다.
* **정규화 전략:** 학습 중 정규화 전략을 공동 최적화하여 심장 분할을 개선하는 연구 [4, 23, 24, 35]도 있으나, 픽셀 단위 제약(pixel-wise constraints)에 의존하여 종종 중복성을 초래합니다.
* **움직임 추적 기반 분할:** DiffuseMorph [13], GPTrack [33] 등 움직임 추적을 통해 분할하는 방법론과 비교됩니다.

## 🛠️ 방법론

BOTM은 소스 프레임 $I_s$와 타겟 프utter_frame $I_t$로 구성된 심초음파 이미지 쌍을 입력으로 받아, 심초음파 분할과 최적의 해부학적 전송(optimal anatomy transportation)을 동시에 수행합니다.

1. **토큰 임베딩 추출:** 공유 비전 트랜스포머 인코더를 사용하여 각 프레임에서 계층적 토큰 임베딩 $X_s = \{X_s^1, \dots, X_s^L\}$ 및 $X_t = \{X_t^1, \dots, X_t^L\}$을 얻습니다. 여기서 $L$은 인코딩 스테이지 수입니다.
2. **최적 토큰 매칭으로서의 해부학적 일관성:**
    * 각 스테이지 $l$에서 해부학적 일관성 추정은 최적 수송(Optimal Transport, OT) 맵 $T^\star_l$을 학습하는 것으로 공식화됩니다. $T^\star_l$의 각 항목은 토큰 임베딩 인스턴스 $X_s^l \in \mathbb{R}^{h_s^l \times w_s^l \times d_s^l}$와 $X_t^l \in \mathbb{R}^{h_t^l \times w_t^l \times d_s^l}$ 사이의 밀집 매칭 점수를 나타냅니다.
    * 코사인 유사도(cosine similarity) $M = \frac{X_s^l \cdot X_t^l}{\|X_s^l\| \cdot \|X_t^l\|}$를 사용하여 비용 행렬 $C = \mathbf{1} - M$을 정의합니다.
    * 엔트로피 정규화된 OT 문제 [6]를 풀기 위해 다음을 최소화합니다:
        $$T^\star_l = \arg \min_{T \in \mathbb{R}^{h_s^l w_s^l \times h_t^l w_t^l}} \sum_{i j} T_{i j} C_{i j} + \varepsilon H(T)$$
        여기서 $H(T)$는 엔트로피, $\varepsilon$는 온도 파라미터($0.1$)입니다. 이 문제는 반복적인 Sinkhorn 알고리즘 [25]을 통해 해결됩니다.
3. **양방향 교차 수송 어텐션(Bi-directional Cross-Transport Attention, BCTA):**
    * 획득된 최적 수송 계획 $T^\star_l$과 동적 토큰 레벨 해부학적 중요성 추정치를 통합하기 위해 BCTA 프록시 모듈이 도입됩니다.
    * **해부학적 중요성 학습:** 가벼운 MLP(Multi-Layer Perceptron)와 평균 풀링 레이어를 사용하여 각 토큰에 대한 지역적 해부학적 중요성 $Z_{k, \text{local}}^l$과 전역적 해부학적 분포 $Z_{k, \text{global}}^l$를 학습합니다:
        $$Z_k^l = \text{Softmax}(\text{MLP}(\text{Concat}[Z_{k, \text{local}}^l, Z_{k, \text{global}}^l]))$$
        $$Z_{k, \text{local}}^l = \text{MLP}(X_k^l), \quad Z_{k, \text{global}}^l = \text{AvgPool}(\text{MLP}(X_k^l)), \quad k \in \{s,t\}$$
    * **교차 수송 어텐션:** 해부학적 중요성 $Z_k^l$를 마스크 정책으로 사용하여 양방향으로 교차 수송 어텐션 임베딩을 계산하여 토큰 임베딩을 업데이트합니다:
        $$[\tilde{X}_k]_{i j} = \frac{\exp[[A_k]_{i j}](P_k^l)_{i j}}{\sum_{N_k=1} \exp[[A_k]_{i k}](P_k^l)_{i k}}, \quad \text{where } [P_k^l]_{i j} = \begin{cases} 1, & \text{if } i=j \\ [Z_k^l]_{i j}, & \text{if } i \neq j \end{cases}, \quad 1 \le i,j \le N$$
        여기서 $A_k = X_k(X_{\setminus k} \otimes T^\star_l) / \sqrt{D}$는 교차 수송 어텐션입니다. $P_k^l$는 $i \neq j$일 때 $Z_k^l$에 따라 토큰 기여도를 동적으로 조절하는 마스크 역할을 합니다.
4. **분할 마스크 생성:** 수송된 토큰 임베딩 $\tilde{X}_s, \tilde{X}_t$를 가벼운 디코더(4개의 MLP 레이어 스택)에 입력하여 최종 분할 출력(semantic segmentation mask)을 생성합니다.

## 📊 결과

* **학습 능력:** CAMUS2CH, CAMUS4CH, TED 데이터셋에서 BOTM은 mDice 및 mHD와 같은 다양한 지표에서 최첨단(state-of-the-art) 성능을 달성했습니다. 특히, CAMUS2CH LV에서 $-1.917$ mHD 감소, LA에서 $-2.915$ mHD 감소와 TED 데이터셋에서 $+1.9\%$ mDice 향상을 보였습니다. 이는 SAMUS [17] 및 CC-SAM [10]과 같은 강력한 모델과 비교해도 경쟁력 있거나 우수한 결과입니다.
* **일반화 능력:** `RandomBlur` 및 `RandomGaussianNoise`와 같은 인위적인 노이즈 조건에서 CAMUS4CH 데이터셋에 대해 테스트한 결과, BOTM은 UNet 및 TransUNet에 비해 낮은 성능 저하율을 보이며 안정적인 성능을 유지했습니다 (표 4). TED 비디오 데이터셋에 `RandomFrameDropout`을 적용한 제한된 학습 데이터 조건에서도 BOTM은 우수한 일반화 능력을 입증했습니다 (표 5).
* **정성적 비교:** BOTM은 어려운 조건에서도 심장 구조를 정확하게 분할하며, 기준선(baseline) 방법들이 보여주는 경계 단절, 불완전한 구조, 마스크 공동(cavity) 등의 문제를 해결합니다 (그림 5, 6). 마스크 소프트 로짓(soft logits) 시각화 결과, BOTM이 분할 불확실성을 효과적으로 최소화하여 보다 일관되고 신뢰할 수 있는 마스크 경계선을 생성함을 보여줍니다 (그림 7).
* **Ablation Study:**
  * **페어링된 분할의 효과:** 쌍으로 된 이미지 특징을 추가(ADD)하는 것이 단일 프레임 분할보다 성능을 약간 향상시킴으로써, 교차 프레임 해부학적 컨텍스트의 가치를 확인했습니다.
  * **최적 해부학적 수송의 효과:** 최적 수송(OT)을 통한 해부학적 일관성 도입은 mDice에서 약 $2\%$ 향상을 가져와, 형태 연속성 보존에 기여함을 입증했습니다.
  * **양방향 교차 수송 어텐션 프록시의 효과:** BCTA 모듈의 도입은 모든 기준선 및 OT만 적용된 모델보다 우수한 성능을 달성하여, BOTM의 단순하지만 효과적이고 강력한 아키텍처임을 보여주었습니다.

## 🧠 통찰 및 논의

BOTM은 토큰 매칭을 통해 해부학적 일관성을 명시적으로 모델링함으로써 기존 심초음파 분할 방법의 주요 한계를 극복합니다. 이는 특히 심장 주기 변형(cardiac cyclic deformation) 동안의 해부학적 세부 사항과 구조적 정체성을 보존하는 데 중요합니다. 최적 수송 이론을 토큰 레벨 일관성에 적용하고, 학습 가능한 해부학적 중요성 마스크와 결합한 양방향 교차 수송 어텐션은 잡음이 많은 영상과 제한된 데이터 환경에서도 모델의 강건성과 일반화 능력을 크게 향상시킵니다. 이 접근 방식은 복잡한 후처리나 수동 프롬프트 없이도 안정적이고 해석 가능한 분할 결과를 제공합니다. 이는 심초음파 분석 및 임상 지표 추정의 발전에 기여할 수 있는 유망한 아키텍처임을 시사합니다.

## 📌 요약

심초음파 분할은 형태 변화, 부분 관찰, 영역 모호성으로 인한 해부학적 불일치로 어려움을 겪습니다. 이 논문은 이러한 해부학적 일관성을 보장하기 위해 **BOTM (Bi-directional Optimal Token Matching)** 프레임워크를 제안합니다. BOTM은 이미지 쌍으로부터 토큰 임베딩을 추출한 후, **최적 수송(Optimal Transport, OT)**을 통해 토큰 간의 최적 매칭을 찾습니다. 이 매칭은 **양방향 교차 수송 어텐션(Bi-directional Cross-Transport Attention, BCTA)** 모듈을 통해 통합되어 해부학적 중요도를 고려한 동적 토큰 업데이트를 수행합니다. 실험 결과, BOTM은 다양한 데이터셋에서 기존 방법론 대비 뛰어난 분할 정확도와 안정성, 일반화 능력을 보였으며, 특히 해부학적 일관성 측면에서 우수함을 입증했습니다.
